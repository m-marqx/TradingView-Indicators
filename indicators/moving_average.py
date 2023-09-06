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

    def sma(self, source: pd.Series, length: int) -> pd.Series:
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
        sma = source.rolling(length).mean()
        return sma.dropna(axis=0)

    def ema(self, source: pd.Series, length: int) -> pd.Series:
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
        sma = source.rolling(window=length, min_periods=length).mean()[:length]
        rest = source[length:]
        return (
            pd.concat([sma, rest])
            .ewm(span=length, adjust=False)
            .mean()
            .dropna(axis=0)
        )

    def sema(self, source: pd.Series, length: int, smooth: int) -> pd.Series:
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
        emas_dict["source_1"] = self.ema(source, length)
        for value in range(2, smooth + 1):
            emas_dict[f"source_{value}"] = self.ema(
                emas_dict[f"source_{value-1}"],
                length,
            )
        emas_df = pd.DataFrame(emas_dict)
        emas_df["sema"] = (
            emas_df[emas_df.columns[:-1]].diff(axis=1).sum(axis=1) * - 1
            * smooth
            + emas_df[emas_df.columns[-1]]
        )
        sema = emas_df["sema"]
        return sema.dropna(axis=0)

    def _rma_pandas(
        self,
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
        sma = source.rolling(window=length, min_periods=length).mean()[:length]
        rest = source[length:]
        return (
            pd.concat([sma, rest])
            .ewm(alpha=1 / length, **kwargs)
            .mean()
        ).rename("RMA")

