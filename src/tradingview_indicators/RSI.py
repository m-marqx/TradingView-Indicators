import numpy as np
import pandas as pd
from .moving_average import rma


def RSI(source: pd.Series, periods: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) for a given time series
    data.

    Parameters:
    -----------
    source : pd.Series
        The input time series data for which to calculate RSI.
    periods : int, optional
        The number of periods to use for RSI calculation.
        (default: 14)

    Returns:
    --------
    pd.Series
        The calculated RSI values for the input data.
    """
    upward_diff = pd.Series(np.maximum(source - source.shift(1), 0.0)).dropna()

    downward_diff = (
        pd.Series(np.maximum(source.shift(1) - source, 0.0)).dropna()
    )

    relative_strength = rma(upward_diff, periods) / rma(downward_diff, periods)

    rsi = 100 - (100 / (1 + relative_strength))
    return rsi.rename("RSI")
