from typing import Literal
import pandas as pd
from .moving_average import sma, ema, sema, rma
from .utils import DynamicTimeWarping

class DidiIndex:
    def __init__(
        self,
        source: pd.Series,
        short_length: int,
        mid_length: int,
        long_length: int,
        method: Literal["sma", "ema", "sema", "rma"] = "ema",
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

        if method == "sma":
            self.ma = sma
        elif method == "ema":
            self.ma = ema
        elif method == "sema":
            self.ma = sema
        elif method == "rma":
            self.ma = rma
        else:
            raise ValueError(
                "Invalid method provided."
                "Use 'sma', 'ema', 'sema', or 'rma'."
            )

        self.short_ma = self.ma(source, short_length)
        self.mid_ma = self.ma(source, mid_length)
        self.long_ma = self.ma(source, long_length)

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
