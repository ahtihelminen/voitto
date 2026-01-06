import pandas as pd


def shifted_rolling_mean(
    x: pd.Series,
    shift: int,
    window: int
) -> pd.Series:
    return x.shift(shift).rolling(window).mean()

def get_mean_feature_last_n_games(
    unified: pd.DataFrame,
    feature_col: str,
    n_games: int
) -> pd.Series:
    """
    Calculates per-player the shifted rolling mean for a given feature over
    the last n games.
    """
    return unified.groupby("player_name")[feature_col].transform(
        shifted_rolling_mean, shift=1, window=n_games
    )