import pandas as pd


def shifted_rolling_mean(
    x: pd.Series,
    shift: int,
    window: int
) -> pd.Series:
    return x.shift(shift).rolling(window).mean()
