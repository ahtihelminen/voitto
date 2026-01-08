"""XGBoost training tab component."""

import json
from datetime import date

import pandas as pd
import streamlit as st
from sqlmodel import Session, select

from voitto.database.database import engine
from voitto.database.models import ModelArtifact, Unified
from voitto.engine.train_xgb import train_xgboost_model
from voitto.features.registry import FEATURE_REGISTRY


def fetch_training_data(cutoff_date: date) -> pd.DataFrame:
    """Fetches historical game logs for training."""
    with Session(engine) as session:
        statement = select(Unified).where(
            Unified.points is not None,
            Unified.game_date is not None and Unified.game_date < cutoff_date,
        )
        results = session.exec(statement).all()
        return pd.DataFrame([r.model_dump() for r in results])


def render_xgb_training() -> None:
    """Render the XGBoost training form and handle submission."""

    # Get available features from registry
    available_features = sorted(FEATURE_REGISTRY.keys())

    with st.form("xgb_training_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Model Identity")
            model_name = st.text_input(
                "Model Name", placeholder="e.g., XGB_Points_v1"
            )

            target_feature = st.selectbox(
                "Target Variable",
                ["points", "minutes", "fga", "fg_pct", "usg_pct", "residuals"],
                help="What is this model predicting?",
            )

            history_cutoff = st.date_input(
                "Training Cutoff",
                value=date(2025, 10, 1),
                help="Train on data BEFORE this date.",
            )

        with col2:
            st.subheader("Hyperparameters")
            learning_rate = st.number_input(
                "Learning Rate", value=0.05, step=0.01, min_value=0.001
            )
            max_depth = st.slider(
                "Max Depth", min_value=2, max_value=15, value=6
            )
            n_estimators = st.number_input(
                "Number of Estimators",
                value=100,
                min_value=10,
                max_value=1000,
                step=10,
            )

            notes = st.text_area(
                "Notes", placeholder="Testing new rolling window features..."
            )

        # Feature selection section (full width)
        st.subheader("Feature Selection")
        selected_features = st.multiselect(
            "Select Features for Training",
            options=available_features,
            default=available_features[:10]
            if len(available_features) >= 10
            else available_features,
            help=(f"Choose from {len(available_features)} available features. "
                  f"Default shows first 10."),
        )

        st.caption(f"Selected {len(selected_features)} feature(s)")

        submitted = st.form_submit_button("Train XGBoost Model", type="primary")

    if submitted:
        _handle_xgb_training(
            model_name=model_name,
            target_feature=target_feature,
            history_cutoff=history_cutoff,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            notes=notes,
            selected_features=selected_features,
        )


def _handle_xgb_training(
    model_name: str,
    target_feature: str,
    history_cutoff: date,
    learning_rate: float,
    max_depth: int,
    n_estimators: int,
    notes: str,
    selected_features: list[str],
) -> None:
    """Execute XGBoost training pipeline."""
    if not model_name:
        st.error("⚠️ Please name your model.")
        return

    if not selected_features:
        st.error("⚠️ Please select at least one feature for training.")
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
    status = st.status("⚙️ Training XGBoost model...", expanded=True)

    # Load data
    status.write("Fetching historical dataset...")
    df_history = fetch_training_data(history_cutoff)

    if df_history.empty:
        status.update(label="❌ Error: No training data found.", state="error")
        return

    status.write(f"Loaded {len(df_history)} rows of history.")

    # Build config
    config: dict[str, str | float | int | list[str]] = {
        "model_name": model_name,
        "model_type": "XGBoost",
        "target": target_feature,
        "notes": notes,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "n_estimators": n_estimators,
        "features": selected_features,
    }

    # Train
    status.write(
        f"Training on target: '{target_feature}' with {len(selected_features)} "
        f"features..."
    )
    save_path = train_xgboost_model(df_history, config, save_dir="saved_models")

    # Register artifact
    status.write("Registering artifact in database...")
    with Session(engine) as session:
        hyperparam_dict = {
            "training_cutoff": str(history_cutoff),
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "n_estimators": n_estimators,
        }

        new_artifact = ModelArtifact(
            name=model_name,
            model_type="xgboost",
            target_feature=target_feature,
            artifact_path=str(save_path),
            hyperparameters=json.dumps(hyperparam_dict),
            feature_cols=json.dumps(selected_features),
            metrics=json.dumps({"notes": notes}) if notes else None,
        )
        session.add(new_artifact)
        session.commit()

    status.update(label="✅ XGBoost Model Trained!", state="complete")
    st.success(f"Saved to: `{save_path}`")
