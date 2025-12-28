from datetime import date, datetime

import pandas as pd
import streamlit as st
from sqlmodel import Session, create_engine, select

from voitto.engine.train_bayes import train_base_model
from voitto.engine.train_xgb import train_xgboost_model
from voitto.models import Experiment, Unified

# --- Config ---
SQLITE_URL = "sqlite:///voitto.db"
engine = create_engine(SQLITE_URL)

st.title("🧪 Model Forge")
st.markdown("Define and train new **Base Models** (Priors) on historical data.")

# --- 1. Experiment Configuration ---
with st.form("new_experiment_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        exp_name = st.text_input(
            "Experiment Name",
            placeholder="e.g., Gaussian_Residual_v1"
        )
        model_type = st.selectbox(
            "Model Architecture",
            ["Gaussian Residual", "Poisson Base", "XGBoost"]
        )
        
    with col2:
        # Define what counts as "History" for the base model
        history_cutoff = st.date_input(
            "Training Data Cutoff", 
            value=date(2025, 10, 1),
            help="Data BEFORE this date is used to learn the priors."
        )
        description = st.text_area(
            "Notes",
            placeholder="Testing higher recency weight..."
        )

    submitted = st.form_submit_button("🔨 Train Base Model", type="primary")

if submitted:
    if not exp_name:
        st.error("Please provide an Experiment Name.")
    else:
        # Check for duplicates
        with Session(engine) as session:
            existing = session.exec(
                select(Experiment).where(Experiment.name == exp_name)
            ).first()
            if existing:
                st.error(
                    f"Experiment '{exp_name}' already exists."
                    " Choose a unique name."
                )
                st.stop()

        # --- Execution ---
        status = st.status("Training in progress...", expanded=True)
        
        # 1. Fetch Historical Data
        status.write("Fetching historical data from database...")
        with Session(engine) as session:
            statement = select(Unified).where(
                Unified.market_key == "player_points",
                Unified.points is not None,
                (
                    Unified.game_date is not None and
                    Unified.game_date < history_cutoff
                )
            )
            results = session.exec(statement).all()
            df_history = pd.DataFrame([r.model_dump() for r in results])
            
        if df_history.empty:
            status.update(
                label="❌ Failed: No historical data found.", state="error"
            )
            st.stop()
            
        # 2. Train Model
        status.write(
            f"Training Bayesian model on {len(df_history)}"
            " games (this may take a minute)..."
        )
        
        config: dict[str, str | float] = {
            "model_type": model_type,
            "experiment_name": exp_name
        }

        if model_type == "XGBoost":
            save_path = train_xgboost_model(
                df_history,
                config,
                save_dir="saved_models"
            )
        else:
            save_path = train_base_model(
                df_history,
                config,
                save_dir="saved_models"
            )
        
        # 3. Save to DB
        status.write("Saving metadata...")
        with Session(engine) as session:
            new_exp = Experiment(
                name=exp_name,
                model_type=model_type,
                recency_weight=0.5,
                base_model_path=str(save_path),
                training_cutoff=datetime.combine(
                    history_cutoff,
                    datetime.min.time()
                ),
                description=description
            )
            session.add(new_exp)
            session.commit()
        
        status.update(label="✅ Training Complete!", state="complete")
        st.success(f"Model saved to `{save_path}` and registered in database.")

# --- Existing Models Table ---
st.divider()
st.subheader("📚 Registered Models")
with Session(engine) as session:
    exps = session.exec(select(Experiment)).all()
    if exps:
        data = [{
            "ID": e.id, 
            "Name": e.name, 
            "Type": e.model_type, 
            "Cutoff": e.training_cutoff,
            "Path": e.base_model_path
        } for e in exps]
        st.dataframe(pd.DataFrame(data), width='stretch')
    else:
        st.info("No models trained yet.")