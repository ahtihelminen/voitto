import streamlit as st

from voitto.dashboard.components.bayes_tab import render_bayes_training
from voitto.dashboard.components.xgb_tab import render_xgb_training

st.set_page_config(page_title="Model Training", page_icon="🔨", layout="wide")
st.title("Model Training Lab")

# Create the tabs
tab_xgb, tab_bayes, tab_exp = st.tabs(
    ["XGBoost (Regression)", "Bayesian (Priors)", "Experimental"]
)

# Render content inside each tab
with tab_xgb:
    st.caption(
        "Train gradient-boosted decision trees for minutes, points, etc."
    )
    render_xgb_training()  # <--- Calls the logic you wrote previously

with tab_bayes:
    st.caption("Update base priors (Gaussian/Poisson) using historical data.")
    render_bayes_training()

with tab_exp:
    st.info("New architectures coming soon...")
