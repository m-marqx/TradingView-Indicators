from typing import Literal
import pandas as pd
from .moving_average import MovingAverage

ma = MovingAverage()


class MACD:
    """
    Calculate the Moving Average Convergence Divergence (MACD) indicator.

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
        "ema" for Exponential Moving Average or "sma" for Simple
        Moving Average.
        (default: "ema")

    Raises:
    -------
    ValueError
        If an invalid method is provided.

    Attributes:
    -----------
    source : pd.Series
        The input time series data for calculating MACD.
    fast_length : int
        The number of periods for the fast moving average.
    slow_length : int
        The number of periods for the slow moving average.
    signal_length : int
        The number of periods for the signal line moving average.
    fast_ma : pd.Series
        The fast moving average values.
    slow_ma : pd.Series
        The slow moving average values.

    Methods:
    --------
    get_histogram(self) -> pd.DataFrame:
        Calculate the MACD histogram.

    """
    def __init__(
        self,
        source: pd.Series,
        fast_length: int,
        slow_length: int,
        signal_length: int,
        method: Literal["ema", "sma"] = "ema",
    ) -> None:
        """
        Initialize the Moving Average Convergence Divergence (MACD)
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
            "ema" for Exponential Moving Average or "sma" for Simple
            Moving Average.
            (default: "ema")

        Raises:
        -------
        ValueError
            If an invalid method is provided.
        """
        self.source = source
        self.fast_length = fast_length
        self.slow_length = slow_length
        self.signal_length = signal_length
        if method == "sma":
            self.__set_sma()
        if method == "ema":
            self.__set_ema()
        else:
            raise ValueError(f"'{method}' is not a valid method.")

    def __set_ema(self) -> None:
        self.fast_ma = ma.ema(self.source, self.fast_length)
        self.slow_ma = ma.ema(self.source, self.slow_length)

    def __set_sma(self) -> None:
        self.fast_ma = ma.sma(self.source, self.fast_length)
        self.slow_ma = ma.sma(self.source, self.slow_length)

    @property
    def get_histogram(self) -> pd.DataFrame:
        """
        Calculate the MACD histogram.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the MACD histogram values.
        """
        macd = (self.fast_ma - self.slow_ma).dropna()
        macd_signal = ma.ema(macd, self.signal_length)
        histogram = macd - macd_signal
        return histogram
