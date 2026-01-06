import matplotlib.pyplot as plt
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
RETRAIN_DAYS = 7  # Retrain model every week
TEST_START_DATE = "2025-10-22"  # Start of the new season in your data

FEATURES = [
    "player_L1_mins",
    "player_L5_mins",
    "player_L10_mins",
    "player_season_mins",
    "player_L10_std",
    "player_mins_trend",
    "player_L5_ppm",        
    "player_L5_foul_rate",  
    "game_imbalance",
    "team_rest_days",
    "is_home",
    "opp_L5_def_rating",
    "days_since_prev_game",
    "is_back_to_back"
]


def load_unified_data() -> pd.DataFrame:
    print("Fetching data...")
    with Session(engine) as session:
        statement = (
            select(Unified, GameStats.home_team)
            .join(GameStats, Unified.stats_game_id == GameStats.id) # type: ignore
            .order_by(Unified.game_date) # type: ignore
        )
        results = session.exec(statement).all()

        data = []
        for row, home_team in results:
            row_dict = row.model_dump()
            row_dict["game_home_team"] = home_team
            data.append(row_dict)

    return pd.DataFrame(data)


def feature_engineering_minutes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("game_date")

    # --- Pre-computation ---
    safe_mins = df["minutes"].replace(0, 1)
    df["raw_ppm"] = df["points"] / safe_mins
    df["raw_foul_rate"] = df["fouls"] / safe_mins

    # --- Context ---
    df["is_home"] = (df["player_team"] == df["game_home_team"]).astype(int)
    df["team_rest_days"] = df["team_rest_days"].fillna(1)

    # --- Win Rate / Imbalance ---
    df["win"] = df["wl"].map({"W": 1, "L": 0})
    team_games = (
        df[["game_date", "player_team", "win"]]
        .drop_duplicates(subset=["game_date", "player_team"])
        .sort_values("game_date")
    )
    team_grouped = team_games.groupby("player_team")
    team_games["team_L10_win_rate"] = team_grouped["win"].transform(
        shifted_rolling_mean, shift=1, window=10
    )

    df = pd.merge(
        df,
        team_games[["game_date", "player_team", "team_L10_win_rate"]],
        on=["game_date", "player_team"],
        how="left",
    )

    opp_stats = team_games[
        ["game_date", "player_team", "team_L10_win_rate"]
    ].rename(
        columns={
            "player_team": "opponent_team",
            "team_L10_win_rate": "opp_L10_win_rate",
        }
    )
    df = pd.merge(df, opp_stats, on=["game_date", "opponent_team"], how="left")
    df["game_imbalance"] = df["team_L10_win_rate"] - df["opp_L10_win_rate"]

    # --- Rolling Stats ---
    player_grouped = df.groupby("player_name")

    df["player_L1_mins"] = player_grouped["minutes"].transform(
        shifted_rolling_mean, shift=1, window=1
    )
    df["player_L5_mins"] = player_grouped["minutes"].transform(
        shifted_rolling_mean, shift=1, window=5
    )
    df["player_L10_mins"] = player_grouped["minutes"].transform(
        shifted_rolling_mean, shift=1, window=10
    )
    df["player_season_mins"] = player_grouped["minutes"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["player_L10_std"] = player_grouped["minutes"].transform(
        lambda x: x.shift(1).rolling(10).std()
    )
    df["player_mins_trend"] = df["player_L5_mins"] - df["player_season_mins"]

    df["player_L5_ppm"] = player_grouped["raw_ppm"].transform(
        shifted_rolling_mean, shift=1, window=5
    )
    df["player_L5_foul_rate"] = player_grouped["raw_foul_rate"].transform(
        shifted_rolling_mean, shift=1, window=5
    )

    opp_grouped = df.sort_values("game_date").groupby("opponent_team")
    df["opp_L5_def_rating"] = opp_grouped["opp_def_rating"].transform(
        shifted_rolling_mean, shift=1, window=5
    )

    player_grouped = df.groupby("player_name")

    # Calculate days since the player's LAST game
    df["prev_game_date"] = player_grouped["game_date"].shift(1)

    # Convert timedelta to float days
    df["days_since_prev_game"] = (df["game_date"] - df["prev_game_date"]).dt.days

    # Handle first game of season (NaN) -> Default to "fresh" (e.g., 3+ days)
    df["days_since_prev_game"] = df["days_since_prev_game"].fillna(5)

    # Binary Flag: Is this a Back-to-Back? (Rest = 1 day)
    df["is_back_to_back"] = (df["days_since_prev_game"] == 1).astype(int)

    return df

def run_walk_forward():
    # 1. Load & Engineer
    df_raw = load_unified_data()
    if df_raw.empty: return
    
    df_eng = feature_engineering_minutes(df_raw)
    
    # 2. Setup Walk-Forward
    df_clean = df_eng.dropna(subset=FEATURES + [TARGET]).copy()
    df_clean["game_date"] = pd.to_datetime(df_clean["game_date"])
    df_clean = df_clean.sort_values("game_date")
    
    test_start = pd.Timestamp(TEST_START_DATE).tz_localize(df_clean["game_date"].dt.tz)
    
    train_mask = df_clean["game_date"] < test_start
    test_mask = df_clean["game_date"] >= test_start
    
    current_train = df_clean[train_mask].copy()
    test_data = df_clean[test_mask].copy()
    
    print(f"Initial Training Set: {len(current_train)} games")
    print(f"Test Set to Walk Forward: {len(test_data)} games starting {test_start.date()}")
    
    model = None
    last_train_date = None
    all_predictions = []
    
    # 3. Loop by Date
    unique_dates = sorted(test_data["game_date"].unique())
    
    for i, date in enumerate(unique_dates):
        should_retrain = (model is None) or \
                         (last_train_date is None) or \
                         ((date - last_train_date).days >= RETRAIN_DAYS)
        
        if should_retrain:
            print(f"[{date.date()}] Retraining model on {len(current_train)} rows...")
            X_train = current_train[FEATURES]
            y_train = current_train[TARGET]
            
            model = xgb.XGBRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=5,
                objective="reg:squarederror", n_jobs=-1
            )
            model.fit(X_train, y_train)
            last_train_date = date
            
        daily_games = test_data[test_data["game_date"] == date].copy()
        if not daily_games.empty:
            X_test = daily_games[FEATURES]
            preds = model.predict(X_test)
            daily_games["pred_minutes"] = preds
            all_predictions.append(daily_games)
            
        current_train = pd.concat([current_train, daily_games])
        
    # 4. Final Evaluation & Plotting
    if not all_predictions:
        print("No predictions made.")
        return

    results = pd.concat(all_predictions)
    mae = mean_absolute_error(results[TARGET], results["pred_minutes"])
    rmse = np.sqrt(mean_squared_error(results[TARGET], results["pred_minutes"]))
    
    print("\n--- Walk-Forward Results ---")
    print(f"MAE:  {mae:.2f} minutes")
    print(f"RMSE: {rmse:.2f} minutes")
    
    # --- PLOTTING LOGIC ---
    print("\nGenerating scatter plot...")
    plt.figure(figsize=(10, 10))
    
    # Scatter plot
    plt.scatter(results[TARGET], results["pred_minutes"], alpha=0.1, s=10, c='blue')
    
    # Identity line (Perfect prediction)
    lims = [
        np.min([results[TARGET].min(), results["pred_minutes"].min()]),
        np.max([results[TARGET].max(), results["pred_minutes"].max()]),
    ]
    plt.plot(lims, lims, 'r-', alpha=0.75, zorder=0, label="Perfect Prediction")
    
    plt.xlabel("Actual Minutes")
    plt.ylabel("Predicted Minutes")
    plt.title(f"Walk-Forward Validation: Actual vs Predicted\nMAE: {mae:.2f} | RMSE: {rmse:.2f}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    save_path = "minutes_pred_vs_true.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    run_walk_forward()
