import json
from datetime import date

import pandas as pd
import streamlit as st
from sqlmodel import Session, select

# Internal Imports
from voitto.database.database import engine
from voitto.database.models import ModelArtifact, Unified
from voitto.engine.train_bayes import train_base_model
from voitto.engine.train_xgb import train_xgboost_model

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
    # Architecture selector OUTSIDE form for immediate reactivity
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Component Identity")
        model_arch = st.selectbox(
            "Architecture", ["XGBoost", "Gaussian Residual", "Poisson Base"],
            key="model_arch_selector"
        )

    with col2:
        st.subheader("2. Training Config")
        st.caption("Hyperparameters adapt to selected architecture")

    st.divider()

    with st.form("new_component_form"):
        col1, col2 = st.columns(2)

        with col1:
            model_name = st.text_input(
                "Component Name", placeholder="e.g., XGB_Usage_Predictor_v1"
            )

            target_feature = st.selectbox(
                "Target Variable",
                ["points", "residuals", "fga", "minutes", "fg_pct", "usg_pct"],
                help="What is this model trying to predict?",
            )

        with col2:
            history_cutoff = st.date_input(
                "Training Cutoff",
                value=date(2025, 10, 1),
                help="Train on data BEFORE this date.",
            )

            # Adaptive hyperparameters based on model architecture
            st.caption(f"**{model_arch} Hyperparameters**")

            # Default values (matching widget defaults)
            learning_rate = 0.05
            max_depth = 6
            n_estimators = 100
            prior_sigma = 1.0
            lambda_prior = 10.0
            n_samples = 1000

            if model_arch == "XGBoost":
                learning_rate = st.number_input(
                    "Learning Rate", value=0.05, step=0.01, min_value=0.001
                )
                max_depth = st.slider(
                    "Max Depth", min_value=2, max_value=15, value=6
                )
                n_estimators = st.number_input(
                    "Number of Estimators",
                    value=100, min_value=10, max_value=1000, step=10
                )
            elif model_arch == "Gaussian Residual":
                prior_sigma = st.number_input(
                    "Prior Sigma", value=1.0, step=0.1, min_value=0.01
                )
                n_samples = st.number_input(
                    "MCMC Samples",
                    value=1000, min_value=100, max_value=10000, step=100
                )
            else:  # Poisson Base
                lambda_prior = st.number_input(
                    "Lambda Prior", value=10.0, step=1.0, min_value=0.1
                )
                n_samples = st.number_input(
                    "MCMC Samples",
                    value=1000, min_value=100, max_value=10000, step=100
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
            config: dict[str, str | float | int] = {
                "model_name": model_name,
                "model_type": model_arch,
                "target": target_feature,
                "notes": notes,
            }

            # Add architecture-specific hyperparameters
            if model_arch == "XGBoost":
                config["learning_rate"] = learning_rate
                config["max_depth"] = max_depth
                config["n_estimators"] = n_estimators
            elif model_arch == "Gaussian Residual":
                config["prior_sigma"] = prior_sigma
                config["n_samples"] = n_samples
            else:  # Poisson Base
                config["lambda_prior"] = lambda_prior
                config["n_samples"] = n_samples

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
                # Store architecture-specific hyperparameters
                hyperparam_dict: dict[str, str | float | int] = {
                    "training_cutoff": str(history_cutoff)
                }
                if model_arch == "XGBoost":
                    hyperparam_dict["learning_rate"] = learning_rate
                    hyperparam_dict["max_depth"] = max_depth
                    hyperparam_dict["n_estimators"] = n_estimators
                elif model_arch == "Gaussian Residual":
                    hyperparam_dict["prior_sigma"] = prior_sigma
                    hyperparam_dict["n_samples"] = n_samples
                else:  # Poisson Base
                    hyperparam_dict["lambda_prior"] = lambda_prior
                    hyperparam_dict["n_samples"] = n_samples
                hyperparams = json.dumps(hyperparam_dict)

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
            select(ModelArtifact).order_by(ModelArtifact.created_at.desc())  # type: ignore
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
                "No components registered yet. Go to the 'Train Component'"
                " tab to build one."
            )
