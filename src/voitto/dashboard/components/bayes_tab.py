"""Bayesian training tab component (Gaussian Residual & Poisson Base)."""
import json
from datetime import date

import pandas as pd
import streamlit as st
from sqlmodel import Session, select

from voitto.database.database import engine
from voitto.database.models import ModelArtifact, Unified
from voitto.engine.train_bayes import train_base_model


def fetch_training_data(cutoff_date: date) -> pd.DataFrame:
    """Fetches historical game logs for training."""
    with Session(engine) as session:
        statement = select(Unified).where(
            Unified.points is not None,
            Unified.game_date is not None and Unified.game_date < cutoff_date,
        )
        results = session.exec(statement).all()
        return pd.DataFrame([r.model_dump() for r in results])


def render_bayes_training() -> None:
    """Render the Bayesian model training form and handle submission."""
    
    # Architecture selector outside form for immediate reactivity
    model_arch = st.selectbox(
        "Prior Type",
        ["Gaussian Residual", "Poisson Base"],
        help="Gaussian for continuous residuals, Poisson for count data.",
    )

    with st.form("bayes_training_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📝 Model Identity")
            model_name = st.text_input(
                "Model Name", placeholder="e.g., Gaussian_Residual_v1"
            )

            target_feature = st.selectbox(
                "Target Variable",
                ["residuals", "points", "fga", "minutes"],
                help="What is this prior modeling?",
            )

            history_cutoff = st.date_input(
                "Training Cutoff",
                value=date(2025, 10, 1),
                help="Train on data BEFORE this date.",
            )

        with col2:
            st.subheader("⚙️ Hyperparameters")
            
            # Architecture-specific hyperparameters
            if model_arch == "Gaussian Residual":
                prior_sigma = st.number_input(
                    "Prior Sigma", value=1.0, step=0.1, min_value=0.01
                )
                lambda_prior = None
            else:  # Poisson Base
                lambda_prior = st.number_input(
                    "Lambda Prior", value=10.0, step=1.0, min_value=0.1
                )
                prior_sigma = None

            n_samples = st.number_input(
                "MCMC Samples",
                value=1000, min_value=100, max_value=10000, step=100
            )

            notes = st.text_area(
                "Notes", placeholder="Testing updated priors..."
            )

        submitted = st.form_submit_button(
            f"🚀 Train {model_arch} Model", type="primary"
        )

    if submitted:
        _handle_bayes_training(
            model_name=model_name,
            model_arch=model_arch,
            target_feature=target_feature,
            history_cutoff=history_cutoff,
            prior_sigma=prior_sigma,
            lambda_prior=lambda_prior,
            n_samples=n_samples,
            notes=notes,
        )


def _handle_bayes_training(
    model_name: str,
    model_arch: str,
    target_feature: str,
    history_cutoff: date,
    prior_sigma: float | None,
    lambda_prior: float | None,
    n_samples: int,
    notes: str,
) -> None:
    """Execute Bayesian training pipeline."""
    if not model_name:
        st.error("⚠️ Please name your model.")
        return

    # Check for duplicate name
    with Session(engine) as session:
        exists = session.exec(
            select(ModelArtifact).where(ModelArtifact.name == model_name)
        ).first()
        if exists:
            st.error(f"❌ Model '{model_name}' already exists.")
            return

    # Training status indicator
    status = st.status(f"⚙️ Training {model_arch} model...", expanded=True)

    # Load data
    status.write("Fetching historical dataset...")
    df_history = fetch_training_data(history_cutoff)

    if df_history.empty:
        status.update(label="❌ Error: No training data found.", state="error")
        return

    status.write(f"Loaded {len(df_history)} rows of history.")

    # Build config
    config: dict[str, str | float | int] = {
        "model_name": model_name,
        "model_type": model_arch,
        "target": target_feature,
        "notes": notes,
        "n_samples": n_samples,
    }

    if model_arch == "Gaussian Residual":
        config["prior_sigma"] = prior_sigma
    else:  # Poisson Base
        config["lambda_prior"] = lambda_prior

    # Train
    status.write(f"Training on target: '{target_feature}'...")
    save_path = train_base_model(df_history, config, save_dir="saved_models")

    # Register artifact
    status.write("Registering artifact in database...")
    with Session(engine) as session:
        hyperparam_dict: dict[str, str | float | int] = {
            "training_cutoff": str(history_cutoff),
            "n_samples": n_samples,
        }
        if model_arch == "Gaussian Residual":
            hyperparam_dict["prior_sigma"] = prior_sigma
        else:
            hyperparam_dict["lambda_prior"] = lambda_prior

        new_artifact = ModelArtifact(
            name=model_name,
            model_type=model_arch.lower().replace(" ", "_"),
            target_feature=target_feature,
            artifact_path=str(save_path),
            hyperparameters=json.dumps(hyperparam_dict),
            feature_cols=json.dumps([]),
            metrics=json.dumps({"notes": notes}) if notes else None,
        )
        session.add(new_artifact)
        session.commit()

    status.update(label=f"✅ {model_arch} Model Trained!", state="complete")
    st.success(f"Saved to: `{save_path}`")
