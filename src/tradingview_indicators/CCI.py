from typing import Literal

import pandas as pd
import numpy as np
from .errors_exceptions import InvalidArgumentError
from .moving_average import ema, sema, rma


def CCI(
    source: pd.Series,
    length: int = 20,
    constant: float = 0.015,
    method: Literal['sma', 'ema', 'dema', 'tema', 'rma'] = 'sma',
) -> pd.DataFrame:
    """
    Calculate Commodity Channel Index (CCI)  of the input time series
    data.

    Parameters:
    -----------
    source : pd.Series
        The input time series data.
    length : int, optional
        The number of periods to include in the CCI calculation
        (default: 20)
    constant : float, optional
        The constant factor for CCI calculation.
        (default: 0.015)
    method : str, optional
        The method to use for the moving average calculation.
        (default: "sma")
    Returns:
    --------
    pd.DataFrame
        The calculated CCI data as a DataFrame.

    Raises:
    -------
    InvalidArgumentError
        If the method is not 'sma', 'ema', 'sema', or 'rma'.
    """
    source_arr = np.array(source)

    match method:
        case "sma":
            ma = np.convolve(
                source_arr,
                np.ones(length) / length,
                mode="valid"
            )
        case "ema":
            ma = ema(source, length)
        case "dema":
            ma = sema(source, length, 2)
        case "tema":
            ma = sema(source, length, 3)
        case  "rma":
            ma = rma(source, length)
        case _:
            raise InvalidArgumentError(
            "method must be 'sma', 'ema', 'sema', or 'rma',"
                f" got '{method}'."
            )

    window = np.lib.stride_tricks.sliding_window_view(
        source_arr,
        length,
    )

    mean_window = np.mean(window, axis=1)
    abs_diff = np.abs(window - mean_window[:, np.newaxis])
    mad = np.mean(abs_diff, axis=1)

    df = pd.DataFrame()
    df["source"] = source[length - 1 :]
    df["mad"] = mad
    df["ma"] = ma
    df["CCI"] = (df["source"] - df["ma"])  / (constant * df["mad"])

    return df
