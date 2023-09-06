import pandas as pd
import numpy as np
from .moving_average import MovingAverage

ma = MovingAverage()


class CCI:
    """
    A class for calculating Commodity Channel Index (CCI)
    of time series data.

    Attributes:
    -----------
    source_arr : numpy.ndarray
        The input time series data as a NumPy array.
    source : pd.Series
        The input time series data as a DataFrame.
    length : int
        The number of periods to include in the CCI calculation.

    Methods:
    --------
    CCI_precise(smooth_column: str = "sma", constant: float = 0.015) -> CCI:
        Calculate CCI using a precise method.

    set_sma() -> CCI:
        Set the Simple Moving Average (SMA) for the CCI calculation.

    set_ema() -> CCI:
        Set the Exponential Moving Average (EMA) for the CCI calculation.

    CCI(constant: float = 0.015) -> pd.DataFrame:
        Calculate CCI using the specified method.
    """

    def __init__(self, source: pd.Series, length: int = 20):
        """
        Initialize the CCI object.

        Parameters:
        -----------
        source : pd.Series
            The input time series data.
        length : int, optional
            The number of periods to include in the CCI calculation, by default 20.
        """
        self.source_arr = np.array(source)
        self.source = source
        self.length = length

