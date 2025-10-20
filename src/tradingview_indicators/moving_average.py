from typing import Literal
import pandas as pd
import numpy as np

from .errors_exceptions import InvalidArgumentError

def sma(source: pd.Series, length: int) -> pd.Series:
    """
    Calculate the Simple Moving Average (SMA)
    of the input time series data.

    Parameters:
    -----------
    source : pd.Series
        The time series data to calculate the SMA for.
    length : int
        The number of periods to include in the SMA calculation.

    Returns:
    --------
    pd.Series
        The calculated SMA time series data.
    """
    sma_series = source.rolling(length).mean()
    return sma_series.dropna(axis=0)

def ema(source: pd.Series, length: int) -> pd.Series:
    """
    Calculate the Exponential Moving Average (EMA)
    of the input time series data.

    Parameters:
    -----------
    source : pandas.Series
        The time series data to calculate the EMA for.
    length : int
        The number of periods to include in the EMA calculation.

    Returns:
    --------
    pandas.Series
        The calculated EMA time series data.
    """
    sma_series = source.rolling(window=length, min_periods=length).mean()[:length]
    rest = source[length:]
    return (
        pd.concat([sma_series, rest])
        .ewm(span=length, adjust=False)
        .mean()
        .dropna(axis=0)
    )

def sema(source: pd.Series, length: int, smooth: int) -> pd.Series:
    """
    Calculate the Smoothed Exponential Moving Average (SEMA)
    of the input time series data.

    Parameters:
    -----------
    source : pandas.Series
        The time series data to calculate the SEMA for.
    length : int
        The number of periods to include in the SEMA calculation.
    smooth : int
        The smooth of EMAs to calculate.

    Returns:
    --------
    pandas.Series
        The calculeted SEMA time series data.
    """

    emas_dict = {}
    emas_dict["source_1"] = ema(source, length)
    for value in range(2, smooth + 1):
        emas_dict[f"source_{value}"] = ema(
            emas_dict[f"source_{value-1}"],
            length,
        )
    emas_df = pd.DataFrame(emas_dict)
    emas_df["sema"] = (
        emas_df[emas_df.columns[:-1]].diff(axis=1).sum(axis=1) * - 1
        * smooth
        + emas_df[emas_df.columns[-1]]
    )
    sema_series = emas_df["sema"]
    return sema_series.dropna(axis=0)

def _rma_pandas(
    source: pd.Series,
    length: int,
    **kwargs
) -> pd.Series:
    """
    Calculate the Relative Moving Average (RMA) of the input time series
    data.

    Parameters:
    -----------
    source : pandas.Series
        The time series data to calculate the RMA for.
    length : int
        The number of periods to include in the RMA calculation.
    **kwargs : additional keyword arguments
        Additional keyword arguments to pass to the pandas EWM (Exponential
        Weighted Moving Average) function.

    Returns:
    --------
    pandas.Series
        The calculated RMA time series data.

    Note:
    -----
    The first values are different from the TradingView RMA.
    """
    sma_series = (
        source
        .rolling(window=length, min_periods=length)
        .mean()[:length]
    )

    rest = source[length:]

    return (
        pd.concat([sma_series, rest])
        .ewm(alpha=1 / length, **kwargs)
        .mean()
    ).rename("RMA")

def _rma_python(
    source: pd.Series,
    length: int
) -> pd.Series:
    """
    Calculate the Relative Moving Average (RMA) of the input time series
    data using pure python.

    Parameters:
    -----------
    source : pandas.Series
        The time series data to calculate the RMA for.
    length : int
        The number of periods to include in the RMA calculation.

    Returns:
    --------
    pd.Series
        The calculated RMA time series data.

    Note:
    -----
    The pure python version is the only one with precision in the
    initial RMA values. However, with the simple RMA version,
    both pandas and python versions will yield the same precision
    in initial values.
    """
    alpha = 1 / length
    source_pd = _rma_pandas(source, length)[:length]
    source_values = source[length:].to_numpy().tolist()

    rma_series = float(source_pd.dropna().iloc[0])
    rma_list = [rma_series]

    for source_value in source_values:
        rma_series = alpha * source_value + (1 - alpha) * rma_series
        rma_list.append(rma_series)

    rma_series = pd.Series(
        rma_list,
        name="RMA",
        index=source[length - 1:].index
    )

    return rma_series

def rma(
    source: pd.Series,
    length: int,
    method: Literal["numpy", "pandas"] = "numpy"
) -> np.ndarray | pd.Series:
    """
    Calculate the Relative Moving Average (RMA) of the input time series
    data.

    Parameters:
    -----------
    source : pandas.Series
        The time series data to calculate the RMA for.
    length : int
        The number of periods to include in the RMA calculation.
    method : {"numpy", "pandas"}, optional
        The method to use for calculating the RMA, by default "numpy".

    Returns:
    --------
    np.ndarray or pandas.Series
        The calculated RMA time series data.
    """
    match method:
        case "numpy":
            return _rma_python(source, length)
        case "pandas":
            return _rma_pandas(source, length)
        case _:
            raise InvalidArgumentError(
                "method must be 'numpy' or 'pandas',"
                f" got '{method}'."
            )
