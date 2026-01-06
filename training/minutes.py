import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlmodel import Session, select

from voitto.database.database import engine
from voitto.database.models import GameStats, Unified
from voitto.engine.utils import shifted_rolling_mean

# --- Configuration ---
TARGET = "minutes"
FEATURES = [
    # Player History
    "player_L5_mins",
    "player_season_mins",
    "player_L5_fouls",
    "player_L10_mins",
    # Context
    "team_rest_days",
    "is_home",  # Now correctly inferred from DB join
]


def load_unified_data() -> pd.DataFrame:
    """
    Fetches Unified data joined with GameStats to determine home/away context.
    """
    print("Fetching data from Unified table (with GameStats join)...")
    with Session(engine) as session:
        # Join Unified -> GameStats to get the official home_team for the game
        statement = (
            select(Unified, GameStats.home_team)
            .join(GameStats, Unified.stats_game_id == GameStats.id) # type: ignore
            .order_by(Unified.game_date) # type: ignore
        )
        results = session.exec(statement).all()

        # Unpack tuple (Unified_obj, home_team_str)
        data = []
        for row, home_team in results:
            row_dict = row.model_dump()
            row_dict["game_home_team"] = home_team  # Store for comparison
            data.append(row_dict)

        df = pd.DataFrame(data)

    print(f"Loaded {len(df)} rows.")
    return df


def feature_engineering_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates rolling features specifically for predicting playing time.
    """
    df = df.sort_values("game_date")

    # 1. Infer Home/Away (Robust Check)
    # Compare the player's team to the game's home team
    df["is_home"] = (df["player_team"] == df["game_home_team"]).astype(int)

    # 2. Player-Level Rolling Stats
    player_grouped = df.groupby("player_name")

    # Recent Form (L5)
    df["player_L5_mins"] = player_grouped["minutes"].transform(
        shifted_rolling_mean, shift=1, window=5
    )
    df["player_L5_fouls"] = player_grouped["fouls"].transform(
        shifted_rolling_mean, shift=1, window=5
    )

    # Medium Term Form (L10)
    df["player_L10_mins"] = player_grouped["minutes"].transform(
        shifted_rolling_mean, shift=1, window=10
    )

    # Season Baseline (Expanding Mean)
    df["player_season_mins"] = player_grouped["minutes"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # 3. Team/Context Handling
    df["team_rest_days"] = df["team_rest_days"].fillna(1)

    return df


def train_and_validate() -> None:
    # 1. Prep
    df = load_unified_data()
    if df.empty:
        print("No data found in Unified table.")
        return

    df_eng = feature_engineering_minutes(df)
    
    # Drop rows with missing history (start of season / new players)
    df_clean = df_eng.dropna(subset=[*FEATURES, TARGET]).copy()
    df_clean = df_clean.sort_values("game_date")
    
    print(f"Total clean samples: {len(df_clean)}")

    # 2. Split (Time-Series: Train on past, Test on future)
    split_idx = int(len(df_clean) * 0.8)
    train_df = df_clean.iloc[:split_idx]
    valid_df = df_clean.iloc[split_idx:]
    
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_valid = valid_df[FEATURES]
    y_valid = valid_df[TARGET]
    
    print(
        f"Training on {len(X_train)} samples. Validating on {len(X_valid)}"
        f" samples (from {valid_df['game_date'].min().date()})."
    )

    # 3. Train
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        early_stopping_rounds=50,
        objective="reg:squarederror",
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=100
    )

    # 4. Evaluate
    preds = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, preds)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    
    print("\n--- Validation Results ---")
    print(f"MAE:  {mae:.2f} minutes")
    print(f"RMSE: {rmse:.2f} minutes")
    
    print("\n--- Feature Importance ---")
    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print(importance)

if __name__ == "__main__":
    train_and_validate()