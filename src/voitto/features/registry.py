from collections.abc import Callable

import pandas as pd

from voitto.features.utils import (
    get_mean_feature_last_n_games,
    shifted_rolling_mean,
)

# The Registry Dictionary
# Maps "feature_name" -> function(unified) -> Series/DataFrame
FEATURE_REGISTRY: dict[
    str, Callable[[pd.DataFrame], pd.Series | pd.DataFrame]
] = {}


def register_feature(name: str) -> Callable[
    [Callable[[pd.DataFrame], pd.Series]], Callable[[pd.DataFrame], pd.Series]
]:
    """Decorator to add a function to the registry."""

    def decorator(
        func: Callable[[pd.DataFrame], pd.Series],
    ) -> Callable[[pd.DataFrame], pd.Series]:
        FEATURE_REGISTRY[name] = func
        return func

    return decorator


# =============================================================================
# --- Player Rolling Stats (Points) ---
# =============================================================================


@register_feature("player_L5_pts")
def calc_player_L5_pts(unified: pd.DataFrame) -> pd.Series:
    """Player's average points over last 5 games."""
    return get_mean_feature_last_n_games(
        unified, feature_col="points", n_games=5
    )


@register_feature("player_season_pts")
def calc_player_season_pts(unified: pd.DataFrame) -> pd.Series:
    """Player's season average points (expanding mean)."""
    return unified.groupby("player_name")["points"].transform(
        lambda x: x.shift(1).expanding().mean()
    )


# =============================================================================
# --- Player Rolling Stats (Minutes) ---
# =============================================================================


@register_feature("player_L1_mins")
def calc_player_L1_mins(unified: pd.DataFrame) -> pd.Series:
    """Player's minutes from last game."""
    return get_mean_feature_last_n_games(
        unified, feature_col="minutes", n_games=1
    )


@register_feature("player_L5_mins")
def calc_player_L5_mins(unified: pd.DataFrame) -> pd.Series:
    """Player's average minutes over last 5 games."""
    return get_mean_feature_last_n_games(
        unified, feature_col="minutes", n_games=5
    )


@register_feature("player_L10_mins")
def calc_player_L10_mins(unified: pd.DataFrame) -> pd.Series:
    """Player's average minutes over last 10 games."""
    return get_mean_feature_last_n_games(
        unified, feature_col="minutes", n_games=10
    )


@register_feature("player_season_mins")
def calc_player_season_mins(unified: pd.DataFrame) -> pd.Series:
    """Player's season average minutes (expanding mean)."""
    return unified.groupby("player_name")["minutes"].transform(
        lambda x: x.shift(1).expanding().mean()
    )


@register_feature("player_L10_std")
def calc_player_L10_std(unified: pd.DataFrame) -> pd.Series:
    """Standard deviation of player's minutes over last 10 games."""
    return unified.groupby("player_name")["minutes"].transform(
        lambda x: x.shift(1).rolling(10).std()
    )


@register_feature("player_mins_trend")
def calc_player_mins_trend(unified: pd.DataFrame) -> pd.Series:
    """Trend: difference between L5 and season average minutes."""
    # Ensure dependencies are computed
    l5 = (
        unified["player_L5_mins"]
        if "player_L5_mins" in unified.columns
        else calc_player_L5_mins(unified)
    )
    season = (
        unified["player_season_mins"]
        if "player_season_mins" in unified.columns
        else calc_player_season_mins(unified)
    )
    return l5 - season


# =============================================================================
# --- Player Rolling Stats (FGA) ---
# =============================================================================


@register_feature("player_L5_fga")
def calc_player_L5_fga(unified: pd.DataFrame) -> pd.Series:
    """Player's average field goal attempts over last 5 games."""
    return get_mean_feature_last_n_games(
        unified, feature_col="fga", n_games=5
    )


# =============================================================================
# --- Player Efficiency Stats ---
# =============================================================================


@register_feature("player_L5_ppm")
def calc_player_L5_ppm(unified: pd.DataFrame) -> pd.Series:
    """Player's points per minute over last 5 games."""
    safe_mins = unified["minutes"].replace(0, 1)
    unified_temp = unified.copy()
    unified_temp["_raw_ppm"] = unified["points"] / safe_mins
    return unified_temp.groupby("player_name")["_raw_ppm"].transform(
        shifted_rolling_mean, shift=1, window=5
    )


@register_feature("player_L5_foul_rate")
def calc_player_L5_foul_rate(unified: pd.DataFrame) -> pd.Series:
    """Player's fouls per minute over last 5 games."""
    safe_mins = unified["minutes"].replace(0, 1)
    unified_temp = unified.copy()
    unified_temp["_raw_foul_rate"] = unified["fouls"] / safe_mins
    return unified_temp.groupby("player_name")["_raw_foul_rate"].transform(
        shifted_rolling_mean, shift=1, window=5
    )


# =============================================================================
# --- Team Rolling Stats ---
# =============================================================================


@register_feature("team_L5_pace")
def calc_team_L5_pace(unified: pd.DataFrame) -> pd.Series:
    """Team's average pace over last 5 games."""
    team_games = (
        unified[["game_date", "player_team", "team_pace"]]
        .drop_duplicates()
        .sort_values("game_date")
    )
    team_grouped = team_games.groupby("player_team")
    team_games["team_L5_pace"] = team_grouped["team_pace"].transform(
        shifted_rolling_mean, shift=1, window=5
    )
    return unified.merge(
        team_games[["game_date", "player_team", "team_L5_pace"]],
        on=["game_date", "player_team"],
        how="left",
    )["team_L5_pace"]


@register_feature("team_L10_win_rate")
def calc_team_L10_win_rate(unified: pd.DataFrame) -> pd.Series:
    """Team's win rate over last 10 games."""
    unified_sorted = unified.sort_values("game_date")
    unified_sorted["_win"] = unified_sorted["wl"].map({"W": 1, "L": 0})
    team_games = (
        unified_sorted[["game_date", "player_team", "_win"]]
        .drop_duplicates(subset=["game_date", "player_team"])
        .sort_values("game_date")
    )
    team_grouped = team_games.groupby("player_team")
    team_games["team_L10_win_rate"] = team_grouped["_win"].transform(
        shifted_rolling_mean, shift=1, window=10
    )
    return unified.merge(
        team_games[["game_date", "player_team", "team_L10_win_rate"]],
        on=["game_date", "player_team"],
        how="left",
    )["team_L10_win_rate"]


# =============================================================================
# --- Opponent Rolling Stats ---
# =============================================================================


@register_feature("opp_L5_def_rating")
def calc_opp_L5_def_rating(unified: pd.DataFrame) -> pd.Series:
    """Opponent's defensive rating over last 5 games."""
    opp_games = (
        unified[["game_date", "opponent_team", "opp_def_rating"]]
        .drop_duplicates()
        .sort_values("game_date")
    )
    opp_grouped = opp_games.groupby("opponent_team")
    opp_games["opp_L5_def_rating"] = opp_grouped["opp_def_rating"].transform(
        shifted_rolling_mean, shift=1, window=5
    )
    return unified.merge(
        opp_games[["game_date", "opponent_team", "opp_L5_def_rating"]],
        on=["game_date", "opponent_team"],
        how="left",
    )["opp_L5_def_rating"]


@register_feature("opp_season_def_rating")
def calc_opp_season_def_rating(unified: pd.DataFrame) -> pd.Series:
    """Opponent's season average defensive rating (expanding mean)."""
    opp_games = (
        unified[["game_date", "opponent_team", "opp_def_rating"]]
        .drop_duplicates()
        .sort_values("game_date")
    )
    opp_grouped = opp_games.groupby("opponent_team")
    opp_games["opp_season_def_rating"] = opp_grouped[
        "opp_def_rating"
    ].transform(lambda x: x.shift(1).expanding().mean())
    return unified.merge(
        opp_games[["game_date", "opponent_team", "opp_season_def_rating"]],
        on=["game_date", "opponent_team"],
        how="left",
    )["opp_season_def_rating"]


@register_feature("opp_L5_pace")
def calc_opp_L5_pace(unified: pd.DataFrame) -> pd.Series:
    """Opponent's average pace over last 5 games."""
    opp_games = (
        unified[["game_date", "opponent_team", "opp_pace"]]
        .drop_duplicates()
        .sort_values("game_date")
    )
    opp_grouped = opp_games.groupby("opponent_team")
    opp_games["opp_L5_pace"] = opp_grouped["opp_pace"].transform(
        shifted_rolling_mean, shift=1, window=5
    )
    return unified.merge(
        opp_games[["game_date", "opponent_team", "opp_L5_pace"]],
        on=["game_date", "opponent_team"],
        how="left",
    )["opp_L5_pace"]


@register_feature("opp_L10_win_rate")
def calc_opp_L10_win_rate(unified: pd.DataFrame) -> pd.Series:
    """Opponent's win rate over last 10 games."""
    unified_sorted = unified.sort_values("game_date")
    unified_sorted["_win"] = unified_sorted["wl"].map({"W": 1, "L": 0})
    # Get team win rates first
    team_games = (
        unified_sorted[["game_date", "player_team", "_win"]]
        .drop_duplicates(subset=["game_date", "player_team"])
        .sort_values("game_date")
    )
    team_grouped = team_games.groupby("player_team")
    team_games["_team_L10_win_rate"] = team_grouped["_win"].transform(
        shifted_rolling_mean, shift=1, window=10
    )
    # Rename for opponent lookup
    opp_stats = team_games[
        ["game_date", "player_team", "_team_L10_win_rate"]
    ].rename(
        columns={
            "player_team": "opponent_team",
            "_team_L10_win_rate": "opp_L10_win_rate",
        }
    )
    return unified.merge(
        opp_stats, on=["game_date", "opponent_team"], how="left"
    )["opp_L10_win_rate"]


# =============================================================================
# --- Game Context Features ---
# =============================================================================


@register_feature("game_imbalance")
def calc_game_imbalance(unified: pd.DataFrame) -> pd.Series:
    """Difference between team and opponent win rates (last 10 games)."""
    team_wr = (
        unified["team_L10_win_rate"]
        if "team_L10_win_rate" in unified.columns
        else calc_team_L10_win_rate(unified)
    )
    opp_wr = (
        unified["opp_L10_win_rate"]
        if "opp_L10_win_rate" in unified.columns
        else calc_opp_L10_win_rate(unified)
    )
    return team_wr - opp_wr


@register_feature("is_home")
def calc_is_home(unified: pd.DataFrame) -> pd.Series:
    """Binary flag: 1 if player is on home team, 0 otherwise."""
    return (unified["player_team"] == unified["game_home_team"]).astype(int)


@register_feature("team_rest_days")
def calc_team_rest_days(unified: pd.DataFrame) -> pd.Series:
    """Days since team's last game (filled with 1 if missing)."""
    return unified["team_rest_days"].fillna(1)


@register_feature("days_since_prev_game")
def calc_days_since_prev_game(unified: pd.DataFrame) -> pd.Series:
    """Days since player's previous game."""
    unified_sorted = unified.sort_values("game_date").copy()
    prev_game_date = unified_sorted.groupby("player_name")["game_date"].shift(1)
    time_diff = unified_sorted["game_date"] - prev_game_date
    days_since = time_diff.apply(
        lambda x: x.days if pd.notna(x) else None  # type: ignore[union-attr]
    )
    # Handle first game of season -> default to "fresh" (5 days)
    return days_since.fillna(5)


@register_feature("is_back_to_back")
def calc_is_back_to_back(unified: pd.DataFrame) -> pd.Series:
    """Binary flag: 1 if player played yesterday, 0 otherwise."""
    days_since = (
        unified["days_since_prev_game"]
        if "days_since_prev_game" in unified.columns
        else calc_days_since_prev_game(unified)
    )
    return (days_since == 1).astype(int)


# =============================================================================
# --- Feature Processor ---
# =============================================================================


def get_feature_processor(
    feature_names: list[str],
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Returns a function that adds ONLY the requested features."""

    def processor(unified: pd.DataFrame) -> pd.DataFrame:
        unified_out = unified.copy()
        for name in feature_names:
            if name in FEATURE_REGISTRY:
                # Calculate and assign
                print(f"Generating {name}...")
                unified_out[name] = FEATURE_REGISTRY[name](unified_out)
            else:
                # Warn or check if it's a raw column (like 'is_home')
                if name not in unified_out.columns:
                    print(
                        f"Warning: Feature '{name}' not found in registry or "
                        "data."
                    )
        return unified_out

    return processor


# =============================================================================
# --- Default Feature Sets ---
# =============================================================================


# Default features for XGBoost points prediction (from train_xgb.py)
DEFAULT_XGB_FEATURES = [
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


# Default features for minutes prediction (from minutes.py)
DEFAULT_MINUTES_FEATURES = [
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
