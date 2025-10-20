from typing import Literal
import pandas as pd

from .errors_exceptions import InvalidArgumentError
from .moving_average import sma, ema, sema, rma
from .utils import DynamicTimeWarping

def didi_index(
        source: pd.Series,
        short_length: int,
        mid_length: int,
        long_length: int,
        ma_method: Literal["sma", "ema", "dema", "tema", "rma"] = "ema",
        method: Literal["absolute", "ratio"] = "absolute",
        use_dtw: bool = False,
    ) -> pd.Series:
    """
    Calculate the Didi Index for the given time series data, using the
    specified moving average method and distance calculation method.

    Parameters:
    -----------
    source : pd.Series
        The time series data to calculate the Didi Index for.
    short_length : int
        The number of periods to include in the short moving average
        calculation.
    mid_length : int
        The number of periods to include in the mid moving average
        calculation.
    long_length : int
        The number of periods to include in the long moving average
        calculation.
    ma_method : Literal["sma", "ema", "dema", "tema", "rma"], optional
        The method to use for the moving average calculation.
        (default: "ema")
    method : Literal["absolute", "ratio"], optional
        The method to use for the distances calculation.
        (default: "absolute")
    """
    match ma_method:
        case "sma":
            short_ma = sma(source, short_length)
            mid_ma = sma(source, mid_length)
            long_ma = sma(source, long_length)

        case "ema":
            short_ma = ema(source, short_length)
            mid_ma = ema(source, mid_length)
            long_ma = ema(source, long_length)

        case "dema":
            short_ma = sema(source, short_length, 2)
            mid_ma = sema(source, mid_length, 2)
            long_ma = sema(source, long_length, 2)

        case "tema":
            short_ma = sema(source, short_length, 3)
            mid_ma = sema(source, mid_length, 3)
            long_ma = sema(source, long_length, 3)

        case "rma":
            short_ma = rma(source, short_length)
            mid_ma = rma(source, mid_length)
            long_ma = rma(source, long_length)

        case _:
            raise InvalidArgumentError(
                "ma_method must be 'sma', 'ema', 'dema', 'tema', or 'rma'."
                f" got '{ma_method}'."
            )

    if use_dtw:
        short_didi = (
            DynamicTimeWarping(short_ma, mid_ma)
            .calculate_dtw_distance(method, True)
        )

        long_didi = (
            DynamicTimeWarping(long_ma, mid_ma)
            .calculate_dtw_distance(method, True)
        )

    elif method == "absolute":
        short_didi = short_ma - mid_ma
        long_didi = long_ma - mid_ma

    elif method == "ratio":
        short_didi = short_ma / mid_ma
        long_didi = long_ma / mid_ma
    else:
        raise InvalidArgumentError(
            "method must be 'absolute' or 'ratio'." f" got '{method}'."
        )

    return long_didi - short_didi
