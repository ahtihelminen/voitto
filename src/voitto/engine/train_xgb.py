from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm

from voitto.features.registry import get_feature_processor
from voitto.features.utils import shifted_rolling_mean


def train_xgboost_model(
    training_data: pd.DataFrame, config: dict, save_dir: str = "saved_models"
) -> str:
    target = config.get("target")
    features = config.get("features", [])

    feature_processor = get_feature_processor(features)

    modified_data = feature_processor(training_data.copy())

    X = modified_data[features]
    y = modified_data[target]

    if X.empty:
        msg = (
            "Not enough history to generate rolling "
            f"features for target '{target}'."
        )
        raise ValueError(
            msg
        )

    # 3. Fit
    params = {
        "objective": "reg:squarederror",
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": config.get("learning_rate", 0.05),
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X, y)

    # 4. Save
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    exp_name = config.get("experiment_name", "xgboost_model")
    save_path = f"{save_dir}/{exp_name}.json"
    model.save_model(save_path)

    return save_path


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates historical stats for Players AND Teams.
    """
    df = df.sort_values("game_date")

    # --- A. Player Rolling Stats ---
    player_grouped = df.groupby("player_name")

    # Standard Rolling
    df["player_L5_pts"] = player_grouped["points"].transform(
        shifted_rolling_mean, shift=1, window=5
    )
    df["player_L5_mins"] = player_grouped["minutes"].transform(
        shifted_rolling_mean, shift=1, window=5
    )
    df["player_L5_fga"] = player_grouped["fga"].transform(
        shifted_rolling_mean, shift=1, window=5
    )
    df["player_season_pts"] = player_grouped["points"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # --- B. Team Rolling Stats ---
    team_games = (
        df[["game_date", "player_team", "team_pace", "team_efg_pct"]]
        .drop_duplicates()
        .sort_values("game_date")
    )
    team_grouped = team_games.groupby("player_team")

    team_games["team_L5_pace"] = team_grouped["team_pace"].transform(
        shifted_rolling_mean, shift=1, window=5
    )

    df = pd.merge(
        df,
        team_games[["game_date", "player_team", "team_L5_pace"]],
        on=["game_date", "player_team"],
        how="left",
    )

    # --- C. Opponent Stats ---
    opp_games = (
        df[["game_date", "opponent_team", "opp_def_rating", "opp_pace"]]
        .drop_duplicates()
        .sort_values("game_date")
    )
    opp_grouped = opp_games.groupby("opponent_team")

    opp_games["opp_L5_def_rating"] = opp_grouped["opp_def_rating"].transform(
        shifted_rolling_mean, shift=1, window=5
    )
    opp_games["opp_season_def_rating"] = opp_grouped[
        "opp_def_rating"
    ].transform(lambda x: x.shift(1).expanding().mean())
    opp_games["opp_L5_pace"] = opp_grouped["opp_pace"].transform(
        shifted_rolling_mean, shift=1, window=5
    )

    return pd.merge(
        df,
        opp_games[
            [
                "game_date",
                "opponent_team",
                "opp_L5_def_rating",
                "opp_season_def_rating",
                "opp_L5_pace",
            ]
        ],
        on=["game_date", "opponent_team"],
        how="left",
    )


DEFAULT_FEATURES = [
    "market_line",
    "player_L5_pts",
    "player_season_pts",
    "player_L5_mins",
    "player_L5_fga",
    "team_L5_pace",
    "team_rest_days",
    "opp_L5_def_rating",
    "opp_season_def_rating",
    "opp_L5_pace",
]

def run_xgboost_backtest(
    df_full: pd.DataFrame,
    config: dict,
    progress_callback: Callable[[float, str], None] | None = None,
) -> pd.DataFrame:
    test_start = pd.to_datetime(config["test_start_date"], utc=True)
    retrain_days = config.get("retrain_days", 7)

    target = config.get("target", "points")

    # Global FE
    df_eng = add_rolling_features(df_full.copy())
    df_eng = df_eng.sort_values("game_date")  # Strict time order

    # Split
    train_mask = df_eng["game_date"] < test_start
    test_mask = df_eng["game_date"] >= test_start

    current_train = df_eng[train_mask].copy()
    test_df = df_eng[test_mask].copy()

    test_dates = sorted(test_df["game_date"].unique())
    daily_results = []

    last_train_date = None
    model = None
    rolling_sigma = 8.0

    for i, date in enumerate(test_dates):
        if progress_callback:
            progress_callback(
                i / len(test_dates),
                f"Simulating {pd.to_datetime(date).date()}...",
            )

        should_retrain = model is None or (
            last_train_date is not None
            and (date - last_train_date).days >= retrain_days
        )

        if should_retrain:
            train_subset = current_train.dropna(subset=DEFAULT_FEATURES)
            if len(train_subset) > 100:
                X_train = train_subset[DEFAULT_FEATURES]
                y_train = train_subset[target] - train_subset["market_line"]

                model = xgb.XGBRegressor(
                    n_estimators=200, max_depth=4, learning_rate=0.05
                )
                model.fit(X_train, y_train)
                last_train_date = date

                # Update Sigma (Std Dev of residuals)
                train_preds = model.predict(X_train)
                rolling_sigma = np.std(y_train - train_preds)

        # Predict
        day_games = test_df[test_df["game_date"] == date].copy()
        valid_mask = day_games[DEFAULT_FEATURES].notna().all(axis=1)
        valid_rows = day_games[valid_mask].copy()

        if not valid_rows.empty and model:
            X_test = valid_rows[DEFAULT_FEATURES]
            pred_residuals = model.predict(X_test)

            valid_rows["pred_points"] = (
                valid_rows["market_line"] + pred_residuals
            )

            # Probability
            z_scores = pred_residuals / rolling_sigma
            valid_rows["prob_over"] = norm.cdf(z_scores)

            # Betting Logic
            valid_rows["bet_type"] = "No Bet"
            valid_rows["bet_outcome"] = 0.0

            mask_over = valid_rows["prob_over"] > 0.55
            valid_rows.loc[mask_over, "bet_type"] = "Over"
            valid_rows.loc[mask_over, "bet_outcome"] = np.where(
                valid_rows.loc[mask_over, "points"]
                > valid_rows.loc[mask_over, "market_line"],
                0.909,
                -1.0,
            )

            mask_under = valid_rows["prob_over"] < 0.45
            valid_rows.loc[mask_under, "bet_type"] = "Under"
            valid_rows.loc[mask_under, "bet_outcome"] = np.where(
                valid_rows.loc[mask_under, "points"]
                < valid_rows.loc[mask_under, "market_line"],
                0.909,
                -1.0,
            )

            valid_rows["error"] = (
                valid_rows["pred_points"] - valid_rows["points"]
            )
            daily_results.append(valid_rows)

        current_train = pd.concat([current_train, day_games])

    if not daily_results:
        return pd.DataFrame()

    return pd.concat(daily_results)

