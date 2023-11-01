import pandas as pd

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

    if Open is None:
        if "Open" in dataframe.columns:
            open_price = dataframe["Open"]
        elif "open" in dataframe.columns:
            open_price = dataframe["open"]
        else:
            open_price = None
    else:
        open_price = dataframe[Open]

    if High is None:
        if "High" in dataframe.columns:
            high_price = dataframe["High"]
        elif "high" in dataframe.columns:
            high_price = dataframe["high"]
        else:
            high_price = None
    else:
        high_price = dataframe[High]

    if Low is None:
        if "Low" in dataframe.columns:
            low_price = dataframe["Low"]
        elif "low" in dataframe.columns:
            low_price = dataframe["low"]
        else:
            low_price = None
    else:
        low_price = dataframe[Low]

    if Close is None:
        if "Close" in dataframe.columns:
            close_price = dataframe["Close"]
        elif "close" in dataframe.columns:
            close_price = dataframe["close"]
        else:
            close_price = None
    else:
        close_price = dataframe[Close]

    return (open_price, high_price, low_price, close_price)
