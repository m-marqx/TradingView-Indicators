from typing import Literal
import pandas as pd

from .moving_average import sma, ema, sema, rma

from .errors_exceptions import InvalidArgumentError


def tsi(
    source: pd.Series,
    short_length: int = 13,
    long_length: int = 25,
    ma_method: Literal["sma", "ema", "dema", "tema", "rma"] = "ema",
) -> pd.DataFrame:
    """
    Calculate the True Strength Index (TSI) momentum oscillator indicator.

    Parameters:
    -----------
    source : pd.Series
        The input time series data for calculating TSI.
    short_length : int
        The number of periods for the short-term moving average.
        (default: 13)
    long_length : int
        The number of periods for the long-term moving average.
        (default: 25)
    signal_length : int
        The number of periods for the signal line moving average.
        (default: 13)
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

    PC = source.diff().iloc[1:]
    absolute_PC = PC.abs()

    match ma_method:
        case "sma":
            short_smoothed = sma(PC, short_length)
            long_smoothed = sma(short_smoothed, long_length)
            absolute_short_smoothed = sma(absolute_PC, short_length)
            absolute_long_smoothed = sma(absolute_short_smoothed, long_length)
        case "ema":
            short_smoothed = ema(PC, short_length)
            long_smoothed = ema(short_smoothed, long_length)
            absolute_short_smoothed = ema(absolute_PC, short_length)
            absolute_long_smoothed = ema(absolute_short_smoothed, long_length)
        case "dema":
            short_smoothed = sema(PC, short_length, 2)
            long_smoothed = sema(short_smoothed, long_length, 2)
            absolute_short_smoothed = sema(absolute_PC, short_length, 2)
            absolute_long_smoothed = sema(
                absolute_short_smoothed, long_length, 2
            )
        case "tema":
            short_smoothed = sema(PC, short_length, 3)
            long_smoothed = sema(short_smoothed, long_length, 3)
            absolute_short_smoothed = sema(absolute_PC, short_length, 3)
            absolute_long_smoothed = sema(
                absolute_short_smoothed, long_length, 3
            )
        case "rma":
            short_smoothed = rma(PC, short_length)
            long_smoothed = rma(short_smoothed, long_length)
            absolute_short_smoothed = rma(absolute_PC, short_length)
            absolute_long_smoothed = rma(absolute_short_smoothed, long_length)
        case _:
            raise InvalidArgumentError(
                f"ma_method must be 'sma', 'ema', 'dema', 'tema', or 'rma',"
                f" got '{ma_method}'."
            )

    TSI = (long_smoothed / absolute_long_smoothed).rename("TSI")

    return TSI
