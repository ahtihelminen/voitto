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
training = st.Page(
    "src/voitto/dashboard/training.py",
    title="Model Training",
    icon="🔨",
)

page = st.navigation(
    [
        home,
        data_explorer,
        training,
    ]
)
page.run()
