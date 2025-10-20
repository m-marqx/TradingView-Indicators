import pandas as pd

def stoch(source: pd.Series, high: pd.Series, low: pd.Series, length: int) -> pd.Series:
    """
    Calculate the Fast Stochastic Oscillator values for the given
    period length.

    Parameters:
    -----------
    source : pd.Series
        The input time series data for calculating the Stochastic
        Oscillator.
    high : pd.Series
        The high prices for the given time series data.
    low : pd.Series
        The low prices for the given time series data.
    length : int
        The length of the stochastic period.

    Returns:
    --------
    pd.Series
        The Fast Stochastic Oscillator values.
    """
    lowest_low = low.rolling(length).min()
    highest_high = high.rolling(length).max()
    stochastic = (
        100
        * (source - lowest_low)
        / (highest_high - lowest_low)
    )
    return stochastic.rename("stoch")
