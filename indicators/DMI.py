import pandas as pd
import numpy as np
from .moving_average import MovingAverage

ma = MovingAverage()


class DMI:
    """
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

    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        source: str = None,
        high: str = None,
        low: str = None,
    ) -> None:
        """
        Initialize the DMI object with the given data and
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
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("dataframe param must be a DataFrame")

        if not isinstance(source, str):
            raise ValueError("source param must be a string")

        if source is None:
            if "Close" in dataframe.columns:
                self.source = dataframe["Close"]
            elif "close" in dataframe.columns:
                self.source = dataframe["close"]
        else:
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

    def true_range(self) -> pd.Series:
        """
        Calculate the True Range (TR) values for the given data.

        True Range is the greatest of the following three values:
        1. The current high minus the current low.
        2. The absolute value of the current high minus the previous close.
        3. The absolute value of the current low minus the previous close.

        Returns:
        --------
        pd.Series
            The True Range (TR) values.
        """
        true_range = np.maximum(
            self.high - self.low,
            abs(self.high - self.source.shift()),
            abs(self.low - self.source.shift())
        )
        return true_range


