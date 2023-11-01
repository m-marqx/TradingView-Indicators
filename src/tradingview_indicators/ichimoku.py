import pandas as pd

class Ichimoku:
    """
    Attributes:
    -----------
    source : pd.Series
        The source values from the DataFrame.
    high : pd.Series
        The high prices from the DataFrame.
    low : pd.Series
        The low prices from the DataFrame.
    close : pd.Series
        The close prices from the DataFrame.

    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        high: str = None,
        low: str = None,
        close: str = None,
    ) -> None:
        """
        Initialize the DMI object with the given data and
        parameters.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The DataFrame containing the source, high, and low data.
        high : str, optional
            The column name in the DataFrame representing the high
            data. If not provided, it will be inferred from common
            column names.
        low : str, optional
            The column name in the DataFrame representing the low
            data. If not provided, it will be inferred from common
            column names.
        close: str, optional
            The column name in the DataFrame representing the close
            data. If not provided, it will be inferred from common
            column names.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("dataframe param must be a DataFrame")

        ohlc_columns_lowercase = ['open', 'high', 'low', 'close']
        ohlc_columns_uppercase = ['Open', 'High', 'Low', 'Close']

        is_ohlc_columns_lowercase = all(
            column in dataframe.columns
            for column in ohlc_columns_lowercase
        )

        is_ohlc_columns_uppercase = all(
            column in dataframe.columns
            for column in ohlc_columns_uppercase
        )

        if not is_ohlc_columns_lowercase and not is_ohlc_columns_uppercase:
            raise ValueError("OHLC columns not found in dataframe")

        if high is None:
            if "High" in dataframe.columns:
                self.high = dataframe["High"]
            else:
                self.high = dataframe["high"]
        else:
            self.high = dataframe[high]

        if low is None:
            if "Low" in dataframe.columns:
                self.low = dataframe["Low"]
            else:
                self.low = dataframe["low"]
        else:
            self.low = dataframe[low]

        if close is None:
            if "Close" in dataframe.columns:
                self.close = dataframe["Close"]
            else:
                self.close = dataframe["close"]
        else:
            self.close = dataframe[close]

    def ichimoku_clouds(
        self,
        conversion_periods: int,
        base_periods: int,
        lagging_span_2_periods: int,
        displacement: int,
    ) -> pd.DataFrame:
        """
        Calculate the components of the Ichimoku Cloud indicator.

        This method computes the Ichimoku Cloud indicator components,
        which include the Conversion Line, Base Line, Leading Span A,
        Leading Span B, Lagging Span, and other intermediate lines.

        Parameters:
        -----------
        conversion_periods : int
            The number of periods to calculate the Conversion Line.
        base_periods : int
            The number of periods to calculate the Base Line.
        lagging_span_2_periods : int
            The number of periods to calculate Lagging Span 2.
        displacement : int
            The displacement of the indicator lines into the future.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the computed Ichimoku Cloud
            components, including 'conversion_line', 'base_line',
            'lagging_span', 'lead_line1', 'lead_line2',
            'leading_span_a', and 'leading_span_b'.

        """
        conversion_line = (
            self._donchian(conversion_periods)
            .rename("conversion_line")
        )

        base_line = (
            self._donchian(base_periods)
            .rename("base_line")
        )

        lead_line1 = (conversion_line + base_line) / 2

        lead_line2 = (
            self._donchian(lagging_span_2_periods)
            .rename("kumo_cloud_lower_line")
        )

        leading_span_a = lead_line1.shift(displacement - 1)
        leading_span_b = lead_line2.shift(displacement - 1)

        lagging_span = (
            self.close
            .shift(-displacement + 1)
            .rename('lagging_span')
        )

        ichimoku_clouds = pd.DataFrame()

        ichimoku_clouds["conversion_line"] = conversion_line
        ichimoku_clouds["base_line"] = base_line
        ichimoku_clouds["lagging_span"] = lagging_span

        ichimoku_clouds["lead_line1"] = lead_line1
        ichimoku_clouds["lead_line2"] = lead_line2

        ichimoku_clouds["leading_span_a"] = leading_span_a
        ichimoku_clouds["leading_span_b"] = leading_span_b
        return ichimoku_clouds

    def _donchian(self, length) -> pd.Series:
        """
        Calculate the Donchian line.

        This method calculates the Donchian line based on the given
        'length' for rolling highest high and lowest low values.

        Parameters:
        -----------
        length : int
            The number of periods for Donchian line calculation.

        Returns:
        --------
        pd.Series
            A Series representing the Donchian line.
        """
        max_rolling = self.high.rolling(length).max()
        min_rolling = self.low.rolling(length).min()
        return (max_rolling + min_rolling) / 2
