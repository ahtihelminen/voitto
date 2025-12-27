from pathlib import Path

import streamlit as st
from sqlmodel import Session, create_engine, select, text

from voitto.models import Experiment

# --- Page Config (Global) ---
st.set_page_config(
    page_title="Voitto Model Lab",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
SQLITE_URL = "sqlite:///voitto.db"
MODELS_DIR = "saved_models"

# --- UI Header ---
st.title("🏀 Voitto Model Lab")
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
    **1. 🧪 Model Forge**
    Define new experiments and train base priors (23-25).
    *Output: Saved .nc trace file*
    
    **2. 📊 Backtest Lab**
    Run walk-forward simulations on saved models.
    *Output: Performance metrics & Kelly growth*
    
    **3. 💰 Live War Room**
    Generate today's predictions using the best model.
    *Output: Actionable betting slip*
    """)

with col3:
    st.subheader("📌 Quick Stats")
    # Placeholder for when tables exist
    with Session(engine) as session:
        # We wrap in try/except in case tables aren't created yet
        exps = session.exec(select(Experiment)).all()
        st.metric("Experiments", len(exps))

# --- Footer ---
st.divider()
st.caption("Voitto Analytics v0.2.0 | Powered by Bambi & PyMC")