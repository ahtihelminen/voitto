import pandas as pd
import streamlit as st
from sqlmodel import SQLModel, select

from voitto.database import engine
from voitto.models import (
    DailyPrediction,
    GameOdds,
    GameStats,
    ModelArtifact,
    PlayerPropOdds,
    PlayerStats,
    TeamStats,
    Unified,
)

st.title("🔎 Data Explorer")

# --- 1. Table Selector ---
TABLE_MAP = {
    "Unified Data": Unified,
    "Player Stats": PlayerStats,
    "Player Prop Odds": PlayerPropOdds,
    "Game Odds": GameOdds,
    "Game Stats": GameStats,
    "Team Stats": TeamStats,
    "Model Artifacts": ModelArtifact,
    "Predictions": DailyPrediction,
}

table_name = st.selectbox("Select Table", list(TABLE_MAP.keys()))
model_class = TABLE_MAP[table_name]

# --- 2. Data Loading ---
# We limit to 5000 rows to prevent memory issues during exploration
limit = st.number_input(
    "Row Limit", min_value=100, max_value=50000, value=2000, step=100
)


@st.cache_data(ttl=60)  # Cache data for 1 minute
def load_data(model: type[SQLModel], row_limit: int) -> pd.DataFrame:
    with engine.connect() as conn:
        query = select(model).limit(row_limit)
        return pd.read_sql(query, conn)


df = load_data(model_class, limit)

# --- 3. Global Search Filter ---
search_query = st.text_input("Global Search", placeholder="Type to filter...")

if search_query and not df.empty:
    # Filter rows where any string column contains the search term
    string_cols = df.select_dtypes(include=["object", "string"]).columns
    if len(string_cols) > 0:
        mask = (
            df[string_cols]
            .apply(
                lambda x: x.astype(str).str.contains(
                    search_query, case=False, na=False
                )
            )
            .any(axis=1)
        )
        df = df[mask]

# --- 4. Display ---
st.caption(f"Showing {len(df)} rows")
st.dataframe(
    df,
    width='stretch',
)
