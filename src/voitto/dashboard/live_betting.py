import json
from datetime import datetime

import pandas as pd
import streamlit as st
from sqlmodel import Session, create_engine, select

from voitto.engine.bet import size_bets
from voitto.engine.predict import predict_daily
from voitto.models import ModelArtifact, Unified

SQLITE_URL = "sqlite:///voitto.db"
engine = create_engine(SQLITE_URL)

st.title("💰 Live War Room")

# --- Sidebar Controls ---
st.sidebar.header("Betting Config")
bankroll = st.sidebar.number_input("Bankroll ($)", 1000, 100000, 5000)
kelly_fraction = st.sidebar.slider("Kelly Fraction", 0.1, 1.0, 0.25)
min_edge = st.sidebar.slider(
    "Min Edge %",
    0.0,
    0.10,
    0.01,
    help="Only bet if win prob is this much higher than break-even.",
)

# --- 1. Select Model ---
with Session(engine) as session:
    models = session.exec(select(ModelArtifact)).all()

if not models:
    st.error("No models available.")
    st.stop()

model_names = [m.name for m in models]
selected_model_name = st.selectbox("Select Active Model", model_names)
selected_model = next(
    m for m in models if m.name == selected_model_name
)

# Parse hyperparameters
hyperparams = json.loads(selected_model.hyperparameters)

# --- 2. Data Prep ---
# In a real app, 'upcoming_games' would come from an API (NBA API)
# Here we simulate 'Today' by letting you pick a date from the DB that has no
# points yet, OR we just pick the latest date in the DB for demonstration.
target_date = st.date_input("Target Game Date", datetime.now())

if st.button("🎲 Generate Bets", type="primary"):
    with st.spinner("Crunching numbers..."):
        with Session(engine) as session:
            # A. Get Training Data (Current Season up to yesterday)
            # We define current season start based on the model hyperparams
            # or a fixed date
            training_cutoff = hyperparams.get("training_cutoff", "2025-10-01")
            season_start = pd.to_datetime(training_cutoff)

            hist_stmt = select(Unified).where(
                Unified.market_key == "player_points",
                Unified.points is not None,
                Unified.game_date >= season_start, # type: ignore
                Unified.game_date < target_date, # type: ignore
            )
            df_train = pd.DataFrame(
                [row.model_dump() for row in session.exec(hist_stmt).all()]
            )

            # B. Get Target Games (The games to bet on)
            # In production, this data won't have 'points' yet.
            target_stmt = (
                select(Unified)
                .where(
                    Unified.market_key == "player_points",
                    # Unified.game_date == target_date # Strict match
                    # For demo purposes, we might need a range if specific
                    # date has no games
                    Unified.game_date >= target_date, # type: ignore
                )
                .limit(50)
            )  # Safety limit

            df_target = pd.DataFrame(
                [r.model_dump() for r in session.exec(target_stmt).all()]
            )

        if df_target.empty:
            st.warning(f"No games found for {target_date}.")
            st.stop()

        st.info(
            f"Training on {len(df_train)} recent games to predict"
            f" {len(df_target)} upcoming props."
        )

        # C. Predict
        config = {
            "model_type": selected_model.model_type,
            "recency_weight": hyperparams.get("recency_weight", 2.0),
        }

        preds_df = predict_daily(
            current_season_data=df_train,
            upcoming_games=df_target,
            base_model_path=selected_model.artifact_path,
            config=config,
        )

        # D. Size Bets
        bets_df = size_bets(
            preds_df,
            bankroll=bankroll,
            kelly_fraction=kelly_fraction,
            min_edge=min_edge,
        )

        # --- 3. Display Results ---

        # Filter for active bets
        active_bets = bets_df[bets_df["bet_side"] != "Pass"].copy()

        st.metric("Total Bets Found", len(active_bets))

        if not active_bets.empty:
            st.subheader("🔥 Recommended Bets")

            # Format for display
            display_cols = [
                "player_name",
                "opponent_team",
                "market_line",
                "pred_points",
                "prob_over",
                "bet_side",
                "bet_amount",
                "edge_pct",
            ]
            display_df = active_bets[display_cols].sort_values(
                "edge_pct", ascending=False
            )

            display_df["prob_over"] = display_df["prob_over"].map(
                "{:.1%}".format
            )
            display_df["edge_pct"] = display_df["edge_pct"].map(
                "{:.1%}".format
            )
            display_df["bet_amount"] = display_df["bet_amount"].map(
                "${:,.2f}".format
            )
            display_df["pred_points"] = display_df["pred_points"].map(
                "{:.3f}".format
            )
            display_df["market_line"] = display_df["market_line"].map(
                "{:.3f}".format
            )

            st.dataframe(
                display_df.style.apply(
                    lambda x: [
                        "background-color: #248c1f"
                        if v == "Over"
                        else "background-color: #b32f14"
                        for v in x
                    ],
                    subset=["bet_side"],
                ),
                width='stretch',
            )

            total_wager = active_bets["bet_amount"].sum()
            st.caption(f"Total Exposure: ${total_wager:,.2f}")

        else:
            st.info("No value bets found for these risk settings.")

