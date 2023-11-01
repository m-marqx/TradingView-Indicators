import pandas as pd
from .utils import OHLC_finder

def Ichimoku(
    dataframe: pd.DataFrame,
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
    dataframe : pd.DataFrame
        The DataFrame containing the high, low and close prices.
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
    _, high, low, close = OHLC_finder(
        dataframe,
    )

    def _donchian(length) -> pd.Series:
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
        max_rolling = high.rolling(length).max()
        min_rolling = low.rolling(length).min()
        return (max_rolling + min_rolling) / 2

    conversion_line = (
        _donchian(conversion_periods)
        .rename("conversion_line")
    )

    base_line = (
        _donchian(base_periods)
        .rename("base_line")
    )

    lead_line1 = (conversion_line + base_line) / 2

    lead_line2 = (
        _donchian(lagging_span_2_periods)
        .rename("kumo_cloud_lower_line")
    )

    leading_span_a = lead_line1.shift(displacement - 1)
    leading_span_b = lead_line2.shift(displacement - 1)

    lagging_span = (
        close
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
