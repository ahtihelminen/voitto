import pandas as pd
import streamlit as st
from sqlmodel import Session, create_engine, select

from voitto.models import (  # Assuming you have a models.py file
    GameOdds,
    PlayerGameStats,
    PlayerPropOdds,
)

# Set up database connection
engine = create_engine("sqlite:///voitto.db")

st.set_page_config(layout="wide", page_title="Voitto Dashboard")

st.title("Voitto: Sports Prediction Dashboard")
# Display recent entries
st.subheader("Latest Database Updates")
with Session(engine) as session:
    get_games = select(
        GameOdds
    ).order_by(
        GameOdds.id.desc() # type: ignore
    ).limit(100)
    games = session.exec(get_games).all()
    games_df = pd.DataFrame([row.model_dump() for row in games])

    get_stats = select(
        PlayerGameStats
    ).order_by(
        PlayerGameStats.game_date.asc() # type: ignore
    ).limit(100)
    stats = session.exec(get_stats).all()
    stats_df = pd.DataFrame([row.model_dump() for row in stats])

    get_odds = select(
        PlayerPropOdds
    ).order_by(
        PlayerPropOdds.id.desc() # type: ignore
    ).limit(100)
    odds = session.exec(get_odds).all()
    odds_df = pd.DataFrame([row.model_dump() for row in odds])
    

    st.dataframe(games_df)
    st.dataframe(stats_df)
    st.dataframe(odds_df)


# Prediction vs Market logic
st.subheader("Value Bets: Model vs Market")
# Logic to compare model_prediction and current_odds goes here