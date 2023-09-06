from typing import Literal
import pandas as pd
from .moving_average import MovingAverage

ma = MovingAverage()


class SlowStochastic:
    """
    Calculate Slow Stochastic Oscillator values from a given DataFrame.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        The DataFrame containing historical price data.
    source : str
        The name of the column in the DataFrame representing the source
        values.
    high : str, optional
        The name of the column in the DataFrame representing high
        prices. If not provided, the class will look for columns named
        "High" or "high".
    low : str, optional
        The name of the column in the DataFrame representing low prices.
        If not provided, the class will look for columns named "Low" or
        "low".
    length : int, optional
        The lookback period for calculating the Stochastic Oscillator.
        (default: 14)

    Attributes:
    -----------
    source : pd.Series
        The source values from the DataFrame.
    high : pd.Series
        The high prices from the DataFrame.
    low : pd.Series
        The low prices from the DataFrame.
    length : int
        The lookback period for calculating the Stochastic Oscillator.

    Methods:
    --------
    stoch(length: int) -> pd.Series:
        Calculate the Fast Stochastic Oscillator values for the given
        period length.

    slow_stoch(
        k_length: int = 14,
        k_smoothing: int = 1,
        d_smoothing: int = 3,
        smoothing_method: Literal["sma", "ema"] = "sma"
    ) -> tuple[pd.Series, pd.Series]:
        Calculate the Slow Stochastic Oscillator (SLOWSTOCH) values.

    Raises:
    -------
    ValueError
        If an invalid `smoothing_method` is provided.

    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        source: str,
        high: str = None,
        low: str = None,
        length: int = 14
    ) -> None:
        """
        Initialize the SlowStochastic object with the given data and
        parameters.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The DataFrame containing the source, high, and low data.
        source : str
            The column name in the DataFrame representing the source
            data.
        high : str, optional
            The column name in the DataFrame representing the high
            data. If not provided,
            it will be inferred from common column names.
        low : str, optional
            The column name in the DataFrame representing the low
            data. If not provided,
            it will be inferred from common column names.
        length : int, optional
            The length of the stochastic period. Default is 14.
        """
        self.source = dataframe[source]

        if high is None:
            if "High" in dataframe.columns:
                self.high = dataframe["High"]
            elif "high" in dataframe.columns:
                self.high = dataframe["high"]
        else:
            self.high = dataframe[high]

        if low is None:
            if "Low" in dataframe.columns:
                self.low = dataframe["Low"]
            elif "low" in dataframe.columns:
                self.low = dataframe["low"]
        else:
            self.low = dataframe[low]

        self.length = length

    def stoch(self, length) -> pd.Series:
        """
        Calculate the Fast Stochastic Oscillator values for the given
        period length.

        Parameters:
        -----------
        length : int
            The length of the stochastic period.

        Returns:
        --------
        pd.Series
            The Fast Stochastic Oscillator values.
        """
        lowest_low = self.low.rolling(length).min()
        hightest_high = self.high.rolling(length).max()
        stochastic = (
            100
            * (self.source - lowest_low)
            / (hightest_high - lowest_low)
        )
        return stochastic

    def slow_stoch(
        self,
        k_length: int = 14,
        k_smoothing: int = 1,
        d_smoothing: int = 3,
        smoothing_method: Literal["sma", "ema"] = "sma",
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
        if smoothing_method == "sma":
            k = ma.sma(self.stoch(k_length), k_smoothing)
            d = ma.sma(k, d_smoothing)
        elif smoothing_method == "ema":
            k = ma.ema(self.stoch(k_length), k_smoothing)
            d = ma.ema(k, d_smoothing)
        else:
            raise ValueError("Method must be either 'sma' or 'ema'.")
        return k, d
