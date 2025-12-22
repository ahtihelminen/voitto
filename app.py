import pandas as pd
import streamlit as st
from sqlmodel import Session, create_engine, select

from voitto.models import (  # Assuming you have a models.py file
    GameOdds,
    GameStats,
    PlayerPropOdds,
    PlayerStats,
    Unified,
)

# Set up database connection
engine = create_engine("sqlite:///voitto.db")

st.set_page_config(layout="wide", page_title="Voitto Dashboard")

st.title("Voitto: Sports Prediction Dashboard")
# Display recent entries
st.subheader("Latest Database Updates")
with Session(engine) as session:
    get_games_odds = select(
        GameOdds
    ).order_by(
        GameOdds.id.desc() # type: ignore
    ).limit(100)
    games_odds = session.exec(get_games_odds).all()
    games_odds_df = pd.DataFrame([row.model_dump() for row in games_odds])

    get_games_stats = select(
        GameStats
    ).order_by(
        GameStats.id.desc() # type: ignore
    )
    games_stats = session.exec(get_games_stats).all()
    games_stats_df = pd.DataFrame([row.model_dump() for row in games_stats])

    get_stats = select(
        PlayerStats
    ).order_by(
        PlayerStats.game_date.asc() # type: ignore
    )
    stats = session.exec(get_stats).all()
    stats_df = pd.DataFrame([row.model_dump() for row in stats])

    get_odds = select(
        PlayerPropOdds
    ).order_by(
        PlayerPropOdds.id.desc() # type: ignore
    )
    odds = session.exec(get_odds).all()
    odds_df = pd.DataFrame([row.model_dump() for row in odds])
    
    get_unified = select(
        Unified
    ).order_by(
        Unified.id.desc() # type: ignore
    )
    unified = session.exec(get_unified).all()
    unified_df = pd.DataFrame([row.model_dump() for row in unified])

    st.subheader("Recent Unified Records")
    st.dataframe(unified_df)
    st.subheader("Recent Games (odds)")
    st.dataframe(games_odds_df)
    st.subheader("Recent Games (stats)")
    st.dataframe(games_stats_df)
    st.subheader("Recent Player Stats")
    st.dataframe(stats_df)
    st.subheader("Recent Player Odds")
    st.dataframe(odds_df)



# Prediction vs Market logic
st.subheader("Value Bets: Model vs Market")
# Logic to compare model_prediction and current_odds goes here