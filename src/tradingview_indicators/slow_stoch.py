from typing import Literal
import pandas as pd

from .errors_exceptions import InvalidArgumentError
from .stoch import stoch
from .moving_average import sma, ema, sema, rma


def slow_stoch(
    source: pd.Series,
    high: pd.Series,
    low: pd.Series,
    k_length: int = 14,
    k_smoothing: int = 1,
    d_smoothing: int = 3,
    smoothing_method: Literal["sma", "ema", "dema", "tema", "rma"] = "sma",
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate the Slow Stochastic Oscillator values using the
    specified parameters.

    Parameters:
    -----------
    k_length : int, optional
        The length of the %K line period.
        (default: 14)
    k_smoothing : int, optional
        The smoothing period for the %K line.
        (default: 1)
    d_smoothing : int, optional
        The smoothing period for the %D line.
        (default: 3)
    smoothing_method : Literal["sma", "ema"], optional
        The smoothing method to use for calculations, either 'sma'
        (Simple Moving Average) or 'ema'
        (Exponential Moving Average).
        (default: 'sma')

    Returns:
    --------
    tuple[pd.Series, pd.Series]
        A tuple containing the %K line and %D line values.
    """
    stoch_source = stoch(source, high, low, k_length).dropna()

    match smoothing_method:
        case "sma":
            k = sma(stoch_source, k_smoothing)
            d = sma(k, d_smoothing)
        case "ema":
            k = ema(stoch_source, k_smoothing)
            d = ema(k, d_smoothing)
        case "dema":
            k = sema(stoch_source, k_smoothing, 2)
            d = sema(k, d_smoothing, 2)
        case "tema":
            k = sema(stoch_source, k_smoothing, 3)
            d = sema(k, d_smoothing, 3)
        case "rma":
            k = rma(stoch_source, k_smoothing)
            d = rma(k, d_smoothing)
        case _:
            raise InvalidArgumentError(
                f"smoothing_method must be 'sma', 'ema', 'dema', 'tema',"
                f" or 'rma', got '{smoothing_method}'."
            )

    return k.rename("%K"), d.rename("%D")
