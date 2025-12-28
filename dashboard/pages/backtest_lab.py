import pandas as pd
import plotly.express as px
import streamlit as st
from sqlmodel import Session, create_engine, select

from voitto.engine.backtest import run_backtest
from voitto.engine.train_xgb import run_xgboost_backtest
from voitto.models import Experiment, Unified

st.set_page_config(page_title="Backtest Lab", page_icon="📊", layout="wide")
SQLITE_URL = "sqlite:///voitto.db"
engine = create_engine(SQLITE_URL)

st.title("📊 Backtest Lab")

# --- 1. Select Experiment ---
with Session(engine) as session:
    experiments = session.exec(select(Experiment)).all()

if not experiments:
    st.warning("No trained models found. Go to 'Model Forge' first.")
    st.stop()

exp_names = [e.name for e in experiments]
selected_exp_name = st.selectbox("Select Base Model", exp_names)
selected_exp = next(e for e in experiments if e.name == selected_exp_name)

# --- 2. Simulation Config ---
col1, col2, col3 = st.columns(3)
with col1:
    test_start = st.date_input(
        "Simulation Start Date", value=selected_exp.training_cutoff
    )
with col2:
    recency = st.slider(
        "Recency Weight (Uncertainty)",
        1.0,
        5.0,
        2.0,
        help="Higher = looser priors",
    )
with col3:
    retrain_days = st.slider("Retrain Interval (Days)", 1, 30, 7)

if st.button("🚀 Run Walk-Forward Simulation", type="primary"):
    # UI Setup
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_ui(pct: float, msg: str) -> None:
        progress_bar.progress(pct)
        status_text.text(msg)

    # 1. Load Data (Full Dataset)
    with Session(engine) as session:
        statement = (
            select(Unified)
            .where(
                Unified.market_key == "player_points",
                Unified.points is not None,
                Unified.game_date is not None,
            )
            .order_by(Unified.game_date)  # type: ignore
        )
        results = session.exec(statement).all()
        df_full = pd.DataFrame([r.model_dump() for r in results])
        df_full["game_date"] = pd.to_datetime(
            df_full["game_date"],
            utc=True,
        )

    # 2. Run Backtest
    config = {
        "start_date": "2023-10-01",  # Unused in loop but good for metadata
        "test_start_date": test_start,
        "model_type": selected_exp.model_type,
        "recency_weight": recency,
        "retrain_days": retrain_days,
    }

    if "XGBoost" in selected_exp.model_type:
        results_df = run_xgboost_backtest(df_full, config, update_ui)
    else:
        results_df = run_backtest(df_full, config, update_ui)
    status_text.text("✅ Simulation Complete!")
    progress_bar.progress(100)

    # --- 3. Analysis ---
    st.divider()

    # Calculate Error
    mae_model = results_df["error"].abs().mean()
    mae_market = (results_df["points"] - results_df["market_line"]).abs().mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Model MAE", f"{mae_model:.4f}")
    c2.metric("Market MAE", f"{mae_market:.4f}")
    c3.metric(
        "Edge",
        f"{mae_market - mae_model:.4f}",
        delta_color="normal" if (mae_market - mae_model) > 0 else "inverse",
    )

    # Visuals
    st.subheader("Cumulative Performance")

    # Edge Over Time
    results_df["daily_edge"] = (
        results_df["points"] - results_df["market_line"]
    ).abs() - results_df["error"].abs()
    results_df["cum_edge"] = results_df["daily_edge"].cumsum()

    fig = px.line(
        results_df,
        x="game_date",
        y="cum_edge",
        title="Cumulative Points Saved vs Market",
    )
    st.plotly_chart(fig, width="stretch")

    bets_df = results_df[results_df["bet_type"] != "No Bet"].copy()

    if not bets_df.empty:
        st.subheader("💰 Profitability Simulation (Unit Bets)")

        # Calculate ROI
        total_bets = len(bets_df)
        total_profit = bets_df["bet_outcome"].sum()
        roi = (total_profit / total_bets) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Bets", f"{total_bets}")
        c2.metric("Units Won", f"{total_profit:.2f}")
        c3.metric("ROI", f"{roi:.2f}%", delta_color="normal")

        # Plot Cumulative Profit
        bets_df["cum_profit"] = bets_df["bet_outcome"].cumsum()
        fig_profit = px.line(
            bets_df,
            x="game_date",
            y="cum_profit",
            title="Cumulative Units Won (@ -110 Odds)",
        )
        st.plotly_chart(fig_profit, width="stretch")
    else:
        st.warning("No bets placed with current confidence thresholds.")

    # Data View
    with st.expander("View Raw Predictions"):
        st.dataframe(
            results_df[
                [
                    "game_date",
                    "player_name",
                    "points",
                    "market_line",
                    "pred_points",
                    "error",
                ]
            ]
        )
