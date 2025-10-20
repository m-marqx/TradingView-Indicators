from typing import Literal
import pandas as pd

from .moving_average import sma, ema, sema, rma
from .tsi import tsi

from .errors_exceptions import InvalidArgumentError


def SMIO(
    source,
    long_length: int = 20,
    short_length: int = 5,
    signal_length: int = 5,
    ma_method: Literal["sma", "ema", "dema", "tema", "rma"] = "ema",
):
    """
    Calculate the SMI Ergodic Oscillator (SMIO) indicator.

    Parameters:
    -----------
    source : pd.Series
        The input time series data for calculating SMIO.
    long_length : int
        The number of periods for the long-term moving average.
        (default: 20)
    short_length : int
        The number of periods for the short-term moving average.
        (default: 5)
    signal_length : int
        The number of periods for the signal line moving average.
        (default: 5)
    ma_method : Literal["sma", "ema", "dema", "tema", "rma"], optional
        The method to use for calculating moving averages, either
        "ema" for Exponential Moving Average or "sma" for Simple Moving
        Average.
        (default: "ema")

    Raises:
    -------
    ValueError
        If an invalid method is provided.

    Returns:
    --------
    pd.Series
        The Stochastic Momentum Index (SMIO) values.
    """
    if isinstance(source, pd.DataFrame):
        raise TypeError("source can't be a DataFrame")

    erg = tsi(source, short_length, long_length)

    match ma_method:
        case "sma":
            sig = sma(erg, signal_length)
        case "ema":
            sig = ema(erg, signal_length)
        case "dema":
            sig = sema(erg, signal_length, 2)
        case "tema":
            sig = sema(erg, signal_length, 3)
        case "rma":
            sig = rma(erg, signal_length)
        case _:
            raise InvalidArgumentError(
                f"ma_method must be 'sma', 'ema', 'dema', 'tema', or 'rma',"
                f" got '{ma_method}'."
            )

    smio = (erg - sig).rename("SMIO")
    return smio
