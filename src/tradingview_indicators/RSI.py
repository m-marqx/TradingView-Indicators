from typing import Literal

import numpy as np
import pandas as pd

from .errors_exceptions import InvalidArgumentError
from .moving_average import sma, ema, sema, rma


def RSI(
    source: pd.Series,
    periods: int = 14,
    ma_method: Literal["sma", "ema", "dema", "tema", "rma"] = "rma",
) -> pd.Series:
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

    match ma_method:
        case "sma":
            relative_strength = sma(upward_diff, periods) / sma(downward_diff, periods)
        case "ema":
            relative_strength = ema(upward_diff, periods) / ema(downward_diff, periods)
        case "dema":
            relative_strength = sema(upward_diff, periods, 2) / sema(downward_diff, periods, 2)
        case "tema":
            relative_strength = sema(upward_diff, periods, 3) / sema(downward_diff, periods, 3)
        case "rma":
            relative_strength = rma(upward_diff, periods) / rma(downward_diff, periods)
        case _:
            raise InvalidArgumentError(
                "ma_method must be 'sma', 'ema', 'dema', 'tema', or 'rma',"
                f" got '{ma_method}'."
            )

    rsi = 100 - (100 / (1 + relative_strength))
    return rsi.rename("RSI")
