from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlmodel import Session, create_engine, select

# Import the engine logic we created above
from voitto.engine.backtest import run_backtest
from voitto.models import Unified

# --- Setup ---
st.set_page_config(page_title="Voitto Model Lab", layout="wide")
SQLITE_URL = "sqlite:///voitto.db"
engine = create_engine(SQLITE_URL)

@st.cache_data
def load_data() -> pd.DataFrame:
    with Session(engine) as session:
        statement = select(Unified).where(
            Unified.market_key == "player_points",
            Unified.points is not None ,
            Unified.game_date is not None
        )
        results = session.exec(statement).all()
        df = pd.DataFrame([r.model_dump() for r in results])
        df['game_date'] = pd.to_datetime(df['game_date'], utc=True)
        return df.sort_values('game_date')

# --- Sidebar: Experiment Controls ---
st.sidebar.header("🧪 Experiment Config")

model_type = st.sidebar.selectbox(
    "Model Strategy",
    ["Gaussian Residual", "Poisson Base"]
)
test_start = st.sidebar.date_input("Backtest Start Date", date(2025, 10, 22))
retrain_days = st.sidebar.slider("Retrain Interval (Days)", 1, 30, 7)
recency_weight = st.sidebar.slider("Prior Uncertainty (Recency)", 1.0, 5.0, 2.0)

run_btn = st.sidebar.button("🚀 Run Experiment", type="primary")

# --- Main Page ---
st.title("Voitto Model Development Lab")

if run_btn:
    df = load_data()
    
    # Progress UI
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(pct: float, msg: str) -> None:
        progress_bar.progress(pct)
        status_text.text(msg)
    
    # Run
    config = {
        "model_type": model_type,
        "test_start_date": test_start,
        "retrain_days": retrain_days,
        "recency_weight": recency_weight
    }
    
    results = run_backtest(df, config, update_progress)
    status_text.text("✅ Complete!")
    progress_bar.progress(100)
    
    # --- Analysis Section ---
    st.divider()
    
    # Metrics
    mae_model = results['error'].abs().mean()
    mae_market = (results['points'] - results['market_line']).abs().mean()
    diff = mae_market - mae_model
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Model MAE", f"{mae_model:.4f}")
    col2.metric("Market MAE", f"{mae_market:.4f}")
    col3.metric("Edge vs Market", f"{diff:.4f}", delta_color="normal")
    
    # Save to Leaderboard
    lb_file = "leaderboard.csv"
    entry = {
        "Timestamp": pd.Timestamp.now(),
        "Model": model_type,
        "Retrain Days": retrain_days,
        "Edge": diff,
        "Model MAE": mae_model
    }
    lb_df = pd.DataFrame([entry])
    if Path(lb_file).exists():
        lb_df.to_csv(lb_file, mode='a', header=False, index=False)
    else:
        lb_df.to_csv(lb_file, index=False)

    # Charts
    st.subheader("Performance Analysis")
    
    # 1. Cumulative Edge Over Time
    results['abs_error_model'] = results['error'].abs()
    results['abs_error_market'] = (
        results['points'] - results['market_line']
    ).abs()
    results['daily_edge'] = (results['abs_error_market'] -
                            results['abs_error_model'])
    results['cumulative_edge'] = results['daily_edge'].cumsum()
    
    fig = px.line(results, x='game_date', y='cumulative_edge', 
                    title="Cumulative Edge (Higher is Better)",
                    labels={
                        'cumulative_edge': 'Cumulative Points Saved vs Market'
                    })
    st.plotly_chart(fig, width="stretch")



# --- Leaderboard Section ---
st.divider()
st.subheader("🏆 Leaderboard")
if Path("leaderboard.csv").exists():
    lb = pd.read_csv("leaderboard.csv").sort_values("Edge", ascending=False)
    st.dataframe(lb, use_container_width=True)
else:
    st.info("No experiments run yet.")