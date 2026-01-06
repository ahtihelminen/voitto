import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import shap
import xgboost as xgb
from sqlmodel import Session, select

from voitto.database.database import engine
from voitto.database.models import GameStats, Unified
from voitto.features.utils import shifted_rolling_mean

# --- Configuration ---
TARGET = "minutes"
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
    "is_back_to_back",
]


def load_unified_data() -> pd.DataFrame:
    print("Fetching data...")
    with Session(engine) as session:
        statement = (
            select(Unified, GameStats.home_team)
            .join(GameStats, Unified.stats_game_id == GameStats.id)
            .order_by(Unified.game_date)
        )
        results = session.exec(statement).all()
        data = [
            dict(row.model_dump(), game_home_team=ht) for row, ht in results
        ]
    return pd.DataFrame(data)


def feature_engineering_minutes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("game_date")

    # --- Context & Pre-computation ---
    df["is_home"] = (df["player_team"] == df["game_home_team"]).astype(int)
    df["team_rest_days"] = df["team_rest_days"].fillna(1)

    safe_mins = df["minutes"].replace(0, 1)
    df["raw_ppm"] = df["points"] / safe_mins
    df["raw_foul_rate"] = df["fouls"] / safe_mins

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

    # --- Rust / Fatigue ---
    df["prev_game_date"] = player_grouped["game_date"].shift(1)
    df["days_since_prev_game"] = (
        df["game_date"] - df["prev_game_date"]
    ).dt.days.fillna(5)
    df["is_back_to_back"] = (df["days_since_prev_game"] == 1).astype(int)

    return df


def fix_booster_base_score(booster: xgb.Booster):
    # 1. Extract the full configuration as a JSON string
    config_str = booster.save_config()
    config = json.loads(config_str)

    # 2. Navigate to the base_score parameter
    # It's usually located at: config['learner']['learner_model_param']['base_score']
    try:
        current_base_score = config["learner"]["learner_model_param"][
            "base_score"
        ]

        # 3. Check if it's the problematic list format e.g., ["2.95E1"]
        if isinstance(current_base_score, list) and len(current_base_score) > 0:
            # Extract the first element and force it to be a simple string
            scalar_score = current_base_score[0]
            print(
                f"Patching base_score: {current_base_score} -> '{scalar_score}'"
            )

            config["learner"]["learner_model_param"]["base_score"] = (
                scalar_score
            )

            # 4. Reload the patched configuration back into the booster
            booster.load_config(json.dumps(config))
            print("Booster configuration patched successfully.")

    except KeyError as e:
        print(f"Could not locate base_score in config: {e}")


def run_analysis():
    # 1. Prep Data
    df_raw = load_unified_data()
    df_eng = feature_engineering_minutes(df_raw)
    df_clean = df_eng.dropna(subset=FEATURES + [TARGET]).copy()

    # Train on the last 50%
    split_idx = int(len(df_clean) * 0.5)
    train_df = df_clean.iloc[split_idx:]
    X = train_df[FEATURES]
    y = train_df[TARGET]

    print(f"Training Analysis Model on {len(X)} recent samples...")
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05, n_jobs=-1
    )
    model.fit(X, y)

    # 2. XGBoost Native Importance
    print("\n--- XGBoost Feature Importance (Gain) ---")
    importance = pd.DataFrame(
        {"Feature": FEATURES, "Gain": model.feature_importances_}
    ).sort_values("Gain", ascending=False)
    print(importance.head(10))

    print("\nSanitizing model for SHAP compatibility...")
    temp_json = "temp_xgb_shap.json"

    model.save_model(temp_json)

    with open(temp_json) as f:
        config = json.load(f)

    try:
        # Navigate to params
        learner_params = config.get("learner", {}).get(
            "learner_model_param", {}
        )
        base_score = learner_params.get("base_score")

        # 1. Handle String format: "[2.95E1]" -> "2.95E1"
        if (
            isinstance(base_score, str)
            and base_score.startswith("[")
            and base_score.endswith("]")
        ):
            print(f"Patching String base_score: {base_score} -> ", end="")
            # Remove brackets
            clean_val = base_score.strip("[]").strip("'").strip('"')
            # Handle quoted inner lists if any, e.g. "['2.95']"
            if clean_val.startswith("'") or clean_val.startswith('"'):
                clean_val = clean_val.strip("'").strip('"')

            learner_params["base_score"] = clean_val
            print(f"'{clean_val}'")

            with open(temp_json, "w") as f:
                json.dump(config, f)

        # 2. Handle List format: ["2.95E1"] -> "2.95E1"
        elif isinstance(base_score, list) and len(base_score) > 0:
            print(f"Patching List base_score: {base_score} -> ", end="")
            learner_params["base_score"] = str(base_score[0])
            print(f"'{learner_params['base_score']}'")

            with open(temp_json, "w") as f:
                json.dump(config, f)

        else:
            print(
                f"base_score format OK or not found: {type(base_score)} {base_score}"
            )

    except Exception as e:
        print(f"Warning: JSON patching error: {e}")
    # C. Load the clean Booster from the patched file
    booster = xgb.Booster()
    booster.load_model(temp_json)
    booster.feature_names = FEATURES  # Restore feature names

    # Cleanup
    if os.path.exists(temp_json):
        os.remove(temp_json)
    # ----------------------------------------

    # 3. SHAP Analysis
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(booster)
    X_sample = X.sample(2000, random_state=42)
    shap_values = explainer(X_sample)

    # Plot 1: Summary
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Summary: What drives Minutes?")
    plt.tight_layout()
    plt.savefig("shap_summary_beeswarm.png")
    print("Saved shap_summary_beeswarm.png")

    # Plot 2: Rust
    plt.figure()
    shap.dependence_plot(
        "days_since_prev_game", shap_values.values, X_sample, show=False
    )
    plt.title("Impact of 'Days Rest' on Minutes Prediction")
    plt.tight_layout()
    plt.savefig("shap_dependence_rust.png")
    print("Saved shap_dependence_rust.png")

    # Plot 3: Foul Trouble
    plt.figure()
    shap.dependence_plot(
        "player_L5_foul_rate", shap_values.values, X_sample, show=False
    )
    plt.title("Impact of 'Foul Rate' on Minutes Prediction")
    plt.tight_layout()
    plt.savefig("shap_dependence_fouls.png")
    print("Saved shap_dependence_fouls.png")


if __name__ == "__main__":
    run_analysis()
