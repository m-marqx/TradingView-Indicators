from typing import Literal
import pandas as pd
from .moving_average import sma, ema, sema, rma
from .utils import DynamicTimeWarping

class DidiIndex:
    """
    The DidiIndex class calculates the Didi Index, which is a technical
    indicator used in financial analysis. It calculates the absolute and
    ratio values of the Didi Index, as well as the Dynamic Time Warping
    (DTW) of the absolute value.

    Parameters:
    -----------
    source : pd.Series
        The time series data to calculate the Didi Index for.
    short_length : int
        The number of periods to include in the short moving average
        calculation.
    mid_length : int
        The number of periods to include in the mid moving average
        calculation.
    long_length : int
        The number of periods to include in the long moving average
        calculation.
    method : Literal["sma", "ema", "dema", "tema", "rma"], optional
        The method to use for the moving average calculation.
        (default: "ema")

    Methods:
    --------
    absolute() -> pd.Series:
        Calculate the Didi Index absolute.

    ratio() -> pd.Series:
        Calculate the Didi Index ratio.

    dtw(method:str = 'absolute') -> pd.Series:
        Calculate the Dynamic Time Warping (DTW) of the Didi Index
        absolute.
    """

    def __init__(
        self,
        source: pd.Series,
        short_length: int,
        mid_length: int,
        long_length: int,
        method: Literal["sma", "ema", "dema", "tema", "rma"] = "ema",
    ) -> None:
        """
        Initialize the Didi Index.

        Parameters:
        -----------
        source : pd.Series
            The time series data to calculate the Didi Index for.
        short_length : int
            The number of periods to include in the short moving average
            calculation.
        mid_length : int
            The number of periods to include in the mid moving average
            calculation.
        long_length : int
            The number of periods to include in the long moving average
            calculation.
        method : Literal["sma", "ema", "sema", "rma"]
            The method to use for the moving average calculation.
            (default: "ema")
        """
        self.source = source
        self.short_length = short_length
        self.mid_length = mid_length
        self.long_length = long_length
        self.method = method

        match method:
            case "sma":
                self.short_ma = sma(source, short_length)
                self.mid_ma = sma(source, mid_length)
                self.long_ma = sma(source, long_length)

            case "ema":
                self.short_ma = ema(source, short_length)
                self.mid_ma = ema(source, mid_length)
                self.long_ma = ema(source, long_length)

            case "dema":
                self.short_ma = sema(source, short_length, 2)
                self.mid_ma = sema(source, mid_length, 2)
                self.long_ma = sema(source, long_length, 2)

            case "tema":
                self.short_ma = sema(source, short_length, 3)
                self.mid_ma = sema(source, mid_length, 3)
                self.long_ma = sema(source, long_length, 3)

            case "rma":
                self.short_ma = rma(source, short_length)
                self.mid_ma = rma(source, mid_length)
                self.long_ma = rma(source, long_length)

            case _:
                raise ValueError(
                    "Invalid method provided."
                    "Use 'sma', 'ema', 'dema', 'tema' or 'rma'."
                )

    def absolute(self) -> pd.Series:
        """
        Calculate the Didi Index absolute.

        Returns:
        --------
        pd.Series
            The calculated Didi Index absolute.
        """
        short_didi = self.short_ma - self.mid_ma
        long_didi = self.long_ma - self.mid_ma
        return long_didi - short_didi

    def ratio(self) -> pd.Series:
        """
        Calculate the Didi Index ratio.

        Returns:
        --------
        pd.Series
            The calculated Didi Index ratio.
        """
        short_didi = self.short_ma / self.mid_ma
        long_didi = self.long_ma / self.mid_ma
        return long_didi - short_didi

    def dtw(self, method:str = 'absolute') -> pd.Series:
        """
        Calculate the Dynamic Time Warping (DTW) of the Didi Index
        absolute.

        Returns:
        --------
        pd.Series
            The calculated DTW of the Didi Index absolute.
        """
        short_didi = (
            DynamicTimeWarping(self.short_ma, self.mid_ma)
            .calculate_dtw_distance(method, True)
        )

        long_didi = (
            DynamicTimeWarping(self.long_ma, self.mid_ma)
            .calculate_dtw_distance(method, True)
        )

        return long_didi - short_didi
