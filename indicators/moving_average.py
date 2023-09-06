from typing import Literal
import pandas as pd
import numpy as np

class MovingAverage:
    """
    A class for calculating Simple Moving Average (SMA)
    and Exponential Moving Average (EMA) of time series data.

    Attributes:
    -----------
    None

    Methods:
    --------
    sma(source: pd.Series, length: int) -> pd.Series:
        Calculate the Simple Moving Average (SMA)
        of the input time series data.

    ema(source: pd.Series, length: int) -> pd.Series:
        Calculate the Exponential Moving Average (EMA)
        of the input time series data.

    sema(source: pd.Series, length: int, smooth: int) -> pd.Series:
        Calculate the Smoothed Exponential Moving Average (SEMA)
        of the input time series data.
    """

