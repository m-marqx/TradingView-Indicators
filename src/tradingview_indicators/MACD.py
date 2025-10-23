from typing import Literal
import pandas as pd

from .errors_exceptions import InvalidArgumentError
from .moving_average import sma, ema, sema, rma
from .utils import DynamicTimeWarping

def MACD(
    source: pd.Series,
    fast_length: int,
    slow_length: int,
    signal_length: int,
    diff_method: Literal["absolute", "ratio", "dtw"] = "absolute",
    ma_method: Literal["sma", "ema", "dema", "tema", "rma"] = "ema",
    signal_method: Literal["sma", "ema", "dema", "tema", "rma"] = "ema",
) -> pd.DataFrame:
    """
    Calculate the Moving Average Convergence Divergence (MACD)
    indicator.

    Parameters:
    -----------
    source : pd.Series
        The input time series data for calculating MACD.
    fast_length : int
        The number of periods for the fast moving average.
    slow_length : int
        The number of periods for the slow moving average.
    signal_length : int
        The number of periods for the signal line moving average.
    method : Literal["ema", "sma"], optional
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

    match ma_method:
        case "sma":
            fast_ma = sma(source, fast_length)
            slow_ma = sma(source, slow_length)
        case "ema":
            fast_ma = ema(source, fast_length)
            slow_ma = ema(source, slow_length)
        case "dema":
            fast_ma = sema(source, fast_length, 2)
            slow_ma = sema(source, slow_length, 2)
        case "tema":
            fast_ma = sema(source, fast_length, 3)
            slow_ma = sema(source, slow_length, 3)
        case "rma":
            fast_ma = rma(source, fast_length)
            slow_ma = rma(source, slow_length)
        case _:
            raise InvalidArgumentError(f"ma_method must be 'sma', 'ema', 'dema',"
                f" 'tema', or 'rma', got '{ma_method}'.")

    match diff_method:
        case "absolute":
            macd = (fast_ma - slow_ma).dropna()
        case "ratio":
            macd = (fast_ma / slow_ma).dropna()
        case "dtw":
            macd = (
                DynamicTimeWarping(fast_ma, slow_ma)
                .calculate_dtw_distance("absolute", True)
            )
        case _:
            raise InvalidArgumentError(
                "diff_method must be 'absolute', 'ratio', or 'dtw',"
                f" got '{diff_method}'."
            )

    match signal_method:
        case "sma":
            macd_signal = sma(macd, signal_length)
        case "ema":
            macd_signal = ema(macd, signal_length)
        case "dema":
            macd_signal = sema(macd, signal_length, 2)
        case "tema":
            macd_signal = sema(macd, signal_length, 3)
        case "rma":
            macd_signal = rma(macd, signal_length)
        case _:
            raise InvalidArgumentError(
                f"signal_method must be 'sma', 'ema', 'dema', 'tema', or 'rma',"
                f" got '{signal_method}'."
            )

    match diff_method:
        case "absolute":
            histogram = macd - macd_signal
        case "ratio":
            histogram = macd / macd_signal
        case "dtw":
            histogram = (
                DynamicTimeWarping(macd, macd_signal)
                .calculate_dtw_distance("absolute", True)
            )

    macd_df = pd.concat(
        [
            macd.rename('macd'),
            macd_signal.rename("signal"),
            histogram.rename('histogram')
        ], axis=1
    )

    return macd_df
