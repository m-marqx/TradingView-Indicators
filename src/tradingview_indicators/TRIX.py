from typing import Literal
import pandas as pd
import numpy as np

from .moving_average import sma, ema, sema, rma
from .errors_exceptions import InvalidArgumentError


def TRIX(
    source: pd.Series,
    length: int = 18,
    signal_length: int = 1,
    ma_method: Literal["sma", "ema", "dema", "tema", "rma"] = "ema",
) -> pd.DataFrame:
    """
    Calculate the Triple Exponential Moving Average (TRIX) momentum
    oscillator indicator.

    Parameters:
    -----------
    source : pd.Series
        The input time series data for calculating TRIX.
    length : int
        The number of periods for the TRIX moving average.
        (default: 18)
    signal_length : int
        The number of periods for the signal line moving average.
        (default: 1)
    ma_method : Literal["sma", "ema", "dema", "tema", "rma"], optional
        The method to use for calculating moving averages, either
        "ema" for Exponential Moving Average or "sma" for Simple Moving
        Average.
        (default: "ema")

    Raises:
    -------
    ValueError
        If an invalid method is provided.
    """
    if isinstance(source, pd.DataFrame):
        raise TypeError("source can't be a DataFrame")

    trix_source = np.log(source)

    match ma_method:
        case "sma":
            trix = (
                sma(sma(sma(trix_source, length), length), length)
                .diff(signal_length)
            ) * 10000
        case "ema":
            trix = (
                ema(ema(ema(trix_source, length), length), length)
                .diff(signal_length)
            ) * 10000
        case "dema":
            trix = (
                sema(sema(sema(trix_source, length, 2), length, 2), length, 2)
                .diff(signal_length)
            ) * 10000
        case "tema":
            trix = (
                sema(sema(sema(trix_source, length, 3), length, 3), length, 3)
                .diff(signal_length)
            ) * 10000
        case "rma":
            trix = (
                rma(rma(rma(trix_source, length), length), length)
                .diff(signal_length)
            ) * 10000
        case _:
            raise InvalidArgumentError(
                f"ma_method must be 'sma', 'ema', 'dema', 'tema', or 'rma',"
                f" got '{ma_method}'."
            )

    return trix.rename("TRIX")
