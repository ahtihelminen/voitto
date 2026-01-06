from pathlib import Path

import streamlit as st
from sqlmodel import Session, create_engine, select, text

from voitto.database.models import ModelArtifact

# --- Constants ---
SQLITE_URL = "sqlite:///voitto.db"
MODELS_DIR = "saved_models"

# --- UI Header ---
st.title("Voitto Model Lab")
st.markdown("### Sports Prediction & Betting Platform")
st.divider()

# --- System Check ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🛠 System Status")
    
    # 1. Check Database
    engine = create_engine(SQLITE_URL)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    st.success("Database: Connected")
    # 2. Check Models Folder
    if Path(MODELS_DIR).exists():
        count = len(
            [f for f in Path(MODELS_DIR).iterdir() if f.suffix == ".nc"]
        )
        st.success(f"Model Storage: Found ({count} traces)")
    else:
        st.warning(f"Model Storage: Missing '{MODELS_DIR}'")
        if st.button("Create Folder"):
            Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
            st.rerun()

with col2:
    st.subheader("🚀 Workflow")
    st.markdown("""
    **1. 🔍 Data Explorer**
    Browse and analyze historical player/game data.
    
    **2. 🔨 Model Training**
    Train XGBoost or Bayesian models on historical data.
    *Output: Saved model artifact*
    """)

with col3:
    st.subheader("📌 Quick Stats")
    # Placeholder for when tables exist
    with Session(engine) as session:
        # We wrap in try/except in case tables aren't created yet
        models = session.exec(select(ModelArtifact)).all()
        st.metric("Models", len(models))

# --- Footer ---
st.divider()
st.caption("Voitto Analytics v0.2.0 | Powered by Bambi & PyMC")
