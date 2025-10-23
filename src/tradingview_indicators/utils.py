from typing import Literal
import pandas as pd
import numpy as np
import fastdtw
from .errors_exceptions import InvalidArgumentError


class DynamicTimeWarping:
    """Class for computing Dynamic Time Warping (DTW).

    This class provides methods to calculate the DTW distance and ratio
    between two input sequences.

    Parameters
    ----------
    input_x : numpy.ndarray or pandas.Series
        The first input sequence.
    input_y : numpy.ndarray or pandas.Series
        The second input sequence.

    Attributes
    ----------
    input_x : numpy.ndarray or pandas.Series
        The first input sequence.
    input_y : numpy.ndarray or pandas.Series
        The second input sequence.

    Methods
    -------
    __init__(self, input_x, input_y)
        Initialize the DynamicTimeWarping class with the input sequences.

    dtw_df(self)
        Get the DTW dataframe between the input sequences.

    calculate_dtw_distance(self, method="ratio", align_sequences=False)
        Calculate the DTW distance between the input sequences.
    """

    def __init__(
        self,
        input_x: np.ndarray | pd.Series,
        input_y: np.ndarray | pd.Series,
    ):
        """
        Initialize the DynamicTimeWarping class with the input
        sequences.

        Parameters
        ----------
        input_x : numpy.ndarray or pandas.Series
            The first input sequence.
        input_y : numpy.ndarray or pandas.Series
            The second input sequence.

        """
        self.input_x = input_x
        self.input_y = input_y
        _, self.path = fastdtw.fastdtw(input_x, input_y)
        self.dtw = pd.DataFrame(self.path)

        self.column_x = (
            input_x.name if isinstance(input_x, pd.Series)
            else "input_x"
        )

        self.column_y = (
            input_y.name if isinstance(input_y, pd.Series)
            else "input_y"
        )

    @property
    def dtw_df(self) -> pd.DataFrame:
        """
        Get the DTW dataframe between the input sequences.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the DTW dataframe between the input
            sequences.
            The DataFrame has the following columns:
            - <column_x>_path: The path of the input_x sequence in
            the DTW alignment.
            - <column_y>_path: The path of the input_y sequence in
            the DTW alignment.
            - <column_x>: The values of the input_x sequence.
            - <column_y>: The values of the input_y sequence.
        """
        dtw_df1 = (
            self.input_x.reset_index(drop=True)
            .reindex(self.dtw[0])
            .rename_axis([self.column_x + "_path"])
            .reset_index()
        )

        dtw_df2 = (
            self.input_y.reset_index(drop=True)
            .reindex(self.dtw[1])
            .rename_axis([self.column_y + "_path"])
            .reset_index()
        )

        dtw_concat = pd.concat([dtw_df1, dtw_df2], axis=1)
        ordered_columns = [
            self.column_x + "_path",
            self.column_y + "_path",
            self.column_x,
            self.column_y,
        ]
        dtw_concat = dtw_concat[ordered_columns]
        return dtw_concat

    def calculate_dtw_distance(
        self,
        method: Literal["ratio", "absolute"] = "ratio",
        align_sequences: bool = False
    ) -> pd.Series:
        """
        Calculate the DTW distance between the input sequences.

        Parameters
        ----------
        method : str, optional
            The method to calculate the DTW distance.
            The options are:
            - "ratio": The ratio between the input_x and input_y
            sequences.
            - "absolute": The absolute difference between the input_x
            and input_y sequences.
            (default: "ratio")

        align_sequences : bool, optional
            Whether to align the input sequences based on their lengths.
            If True, the longer sequence will be aligned to match
            the length of the shorter sequence.
            If False, the original sequences will stay intact.
            (default: False)

        Returns
        -------
        pd.Series
            A Series containing the DTW distance between the input
            sequences.
        """
        if align_sequences:
            x_series, y_series = self.align_dtw_distance()
        else:
            x_series = self.dtw_df[self.column_x]
            y_series = self.dtw_df[self.column_y]

        match method:
            case "ratio":
                return x_series / y_series
            case "absolute":
                return x_series - y_series
            case _:
                raise InvalidArgumentError(
                    "method must be 'ratio' or 'absolute'."
                    f" got '{method}'."
                )

    def align_dtw_distance(self):
        """
        Aligns two time series using Dynamic Time Warping (DTW)
        algorithm and returns the aligned series.

        Returns:
            x_series (pandas.Series): Aligned x series.
            y_series (pandas.Series): Aligned y series.
        """
        x_source = self.input_x.copy().rename('x')
        y_source = self.input_y.copy().rename('y')

        if len(self.input_x) > len(self.input_y):
            x_source = x_source.reindex(self.input_y.index)
        elif len(self.input_x) < len(self.input_y):
            y_source = y_source.reindex(self.input_x.index)

        dtw_df = DynamicTimeWarping(x_source, y_source).dtw_df

        x_name = "x"
        y_name = "y"

        x_series = dtw_df[x_name].reindex(
            dtw_df[x_name + '_path'].drop_duplicates()
        )

        y_series = dtw_df[y_name].reindex(
            dtw_df[y_name + '_path'].drop_duplicates()
        )

        x_series.index = x_source.dropna().index
        y_series.index = y_source.dropna().index
        return x_series, y_series


def OHLC_finder(
    dataframe: pd.DataFrame,
    Open: str = None,
    High: str = None,
    Low: str = None,
    Close: str = None,
) -> None:
    """
    Initialize the DMI object with the given data and
    parameters.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        The DataFrame containing the source, high, and low data.
    Open : str
        The column name in the DataFrame representing the open
        data.
    High : str, optional
        The column name in the DataFrame representing the high
        data. If not provided,
        it will be inferred from common column names.
    Low : str, optional
        The column name in the DataFrame representing the low
        data. If not provided, it will be inferred from common
        column names.
    Close : str, optional
        The column name in the DataFrame representing the close
        data. If not provided, it will be inferred from common
        column names.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("dataframe param must be a DataFrame")

    ohlc_columns_lowercase = ["open", "high", "low", "close"]
    ohlc_columns_uppercase = ["Open", "High", "Low", "Close"]
    source_columns = [Open, High, Low, Close]

    is_na_source = all(
        source is None
        for source in source_columns
    )

    is_ohlc_columns_lowercase = all(
        column in dataframe.columns
        for column in ohlc_columns_lowercase
    )

    is_ohlc_columns_uppercase = all(
        column in dataframe.columns
        for column in ohlc_columns_uppercase
    )

    columns_not_found = (
        not is_ohlc_columns_lowercase
        and not is_ohlc_columns_uppercase
    )

    if is_na_source and columns_not_found:
        raise ValueError("OHLC columns not found in dataframe")

    if Open is None:
        if "Open" in dataframe.columns:
            open_price = dataframe["Open"]
        elif "open" in dataframe.columns:
            open_price = dataframe["open"]
    else:
        open_price = dataframe[Open]

    if High is None:
        if "High" in dataframe.columns:
            high_price = dataframe["High"]
        elif "high" in dataframe.columns:
            high_price = dataframe["high"]
    else:
        high_price = dataframe[High]

    if Low is None:
        if "Low" in dataframe.columns:
            low_price = dataframe["Low"]
        elif "low" in dataframe.columns:
            low_price = dataframe["low"]
    else:
        low_price = dataframe[Low]

    if Close is None:
        if "Close" in dataframe.columns:
            close_price = dataframe["Close"]
        elif "close" in dataframe.columns:
            close_price = dataframe["close"]
    else:
        close_price = dataframe[Close]

    return (open_price, high_price, low_price, close_price)
