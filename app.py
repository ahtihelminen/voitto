import streamlit as st

# --- Page Config (Global - only set once here) ---
st.set_page_config(
    page_title="Voitto Model Lab",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Navigation Setup ---
home = st.Page(
    "src/voitto/dashboard/home.py",
    title="Home",
    icon="🏠",
    default=True,
)
data_explorer = st.Page(
    "src/voitto/dashboard/data_explorer.py",
    title="Data Explorer",
    icon="📂",
)
model_forge = st.Page(
    "src/voitto/dashboard/model_forge.py",
    title="Model Forge",
    icon="🧪",
)
backtest_lab = st.Page(
    "src/voitto/dashboard/backtest_lab.py",
    title="Backtest Lab",
    icon="📊",
)
live_betting = st.Page(
    "src/voitto/dashboard/live_betting.py",
    title="Live Betting",
    icon="💰",
)

page = st.navigation(
    [
        home,
        data_explorer,
        model_forge,
        backtest_lab,
        live_betting,
    ]
)
page.run()
