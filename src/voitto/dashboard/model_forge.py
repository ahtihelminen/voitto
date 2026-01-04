import json
from datetime import date

import pandas as pd
import streamlit as st
from sqlmodel import Session, select

# Internal Imports
from voitto.database import engine
from voitto.engine.train_bayes import train_base_model
from voitto.engine.train_xgb import train_xgboost_model
from voitto.models import ModelArtifact, Unified

# --- UI Config ---
st.set_page_config(
    page_title="Voitto Model Forge", page_icon="🧪", layout="wide"
)

st.title("🏭 Model Forge")
st.markdown(
    """
    **Construct Prediction Components.** Train specialized models 
    (e.g., Usage Estimators, Efficiency Priors) 
    that can be assembled into a full prediction pipeline.
    """
)

# --- Tabs ---
tab_train, tab_registry = st.tabs(
    ["🔨 Train Component", "📚 Component Registry"]
)


def fetch_training_data(cutoff_date: date) -> pd.DataFrame:
    """Fetches historical game logs for training."""
    with Session(engine) as session:
        # We fetch all records with stats to allow training on any feature
        # We rely on the Unified view, but ensure we have actual stats
        statement = select(Unified).where(
            Unified.points is not None,
            Unified.game_date is not None and Unified.game_date < cutoff_date,
        )
        results = session.exec(statement).all()
        return pd.DataFrame([r.model_dump() for r in results])


# --- TAB 1: TRAIN COMPONENT ---
with tab_train:
    with st.form("new_component_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("1. Component Identity")
            model_name = st.text_input(
                "Component Name", placeholder="e.g., XGB_Usage_Predictor_v1"
            )
            model_arch = st.selectbox(
                "Architecture", ["XGBoost", "Gaussian Residual", "Poisson Base"]
            )

            target_feature = st.selectbox(
                "Target Variable",
                ["points", "fga", "minutes", "fg_pct", "usg_pct"],
                help="What is this model trying to predict?",
            )

        with col2:
            st.subheader("2. Training Config")
            history_cutoff = st.date_input(
                "Training Cutoff",
                value=date(2025, 10, 1),
                help="Train on data BEFORE this date.",
            )

            # Pack extra config into a JSON string or manage via code
            st.caption("Hyperparameters")
            learning_rate = st.number_input(
                "Learning Rate (XGB)", value=0.05, step=0.01
            )

            notes = st.text_area(
                "Notes", placeholder="Testing new rolling window features..."
            )

        submitted = st.form_submit_button("🚀 Launch Training", type="primary")

    if submitted:
        if not model_name:
            st.error("⚠️ Please name your component.")
        else:
            # 1. Validation
            with Session(engine) as session:
                exists = session.exec(
                    select(ModelArtifact).where(
                        ModelArtifact.name == model_name
                    )
                ).first()
                if exists:
                    st.error(f"❌ Component '{model_name}' already exists.")
                    st.stop()

            # 2. Status Indicator
            status_container = st.status(
                "⚙️ Fabricating Component...", expanded=True
            )

            # 3. Data Loading
            status_container.write("Fetching historical dataset...")
            df_history = fetch_training_data(history_cutoff)

            if df_history.empty:
                status_container.update(
                    label="❌ Error: No training data found.", state="error"
                )
                st.stop()

            status_container.write(f"Loaded {len(df_history)} rows of history.")

            # 4. Configuration Packing
            # We pass these to the training functions.
            config = {
                "model_name": model_name,
                "model_type": model_arch,
                "target": target_feature,
                "learning_rate": learning_rate,
                "notes": notes,
            }

            # 5. Execution
            status_container.write(
                f"Training {model_arch} on target: '{target_feature}'..."
            )

            save_path = None
            if model_arch == "XGBoost":
                save_path = train_xgboost_model(
                    df_history, config, save_dir="saved_models"
                )
            else:
                save_path = train_base_model(
                    df_history, config, save_dir="saved_models"
                )

            # 6. Registration
            status_container.write("Registering artifact in database...")
            with Session(engine) as session:
                hyperparams = json.dumps(
                    {
                        "learning_rate": learning_rate,
                        "training_cutoff": str(history_cutoff),
                    }
                )

                new_component = ModelArtifact(
                    name=model_name,
                    model_type=model_arch.lower().replace(" ", "_"),
                    target_feature=target_feature,
                    artifact_path=str(save_path),
                    hyperparameters=hyperparams,
                    feature_cols=json.dumps(
                        []
                    ),  # Could be populated by training
                    metrics=json.dumps({"notes": notes}) if notes else None,
                )
                session.add(new_component)
                session.commit()

            status_container.update(
                label="✅ Component Built Successfully!", state="complete"
            )
            st.success(f"Saved to: `{save_path}`")



# --- TAB 2: COMPONENT REGISTRY ---
with tab_registry:
    st.subheader("📦 Available Components")

    with Session(engine) as session:
        components = session.exec(
            select(ModelArtifact).order_by(ModelArtifact.created_at.desc()) # type: ignore
        ).all()

        if components:
            # Prepare display data
            display_data = [
                {
                    "ID": c.id,
                    "Name": c.name,
                    "Type": c.model_type,
                    "Target": c.target_feature,
                    "Date": c.created_at.strftime("%Y-%m-%d"),
                    "Path": c.artifact_path,
                }
                for c in components
            ]

            st.dataframe(
                pd.DataFrame(display_data),
                column_config={
                    "Path": st.column_config.TextColumn(
                        "File Path", width="medium"
                    ),
                },
                width="stretch",
                hide_index=True,
            )

            # Action: Delete
            st.divider()
            col_del, _ = st.columns([1, 3])
            with col_del:
                to_delete = st.selectbox(
                    "Select Component to Delete", [c.name for c in components]
                )
                if st.button("🗑️ Delete Component"):
                    with Session(engine) as session:
                        obj = session.exec(
                            select(ModelArtifact).where(
                                ModelArtifact.name == to_delete
                            )
                        ).first()
                        if obj:
                            session.delete(obj)
                            session.commit()
                            st.rerun()
        else:
            st.info(
                "No components registered yet. Go to the 'Train Component'" \
                " tab to build one."
            )
