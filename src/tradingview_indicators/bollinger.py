from typing import Literal
import pandas as pd
import numpy as np

from .errors_exceptions import InvalidArgumentError
from .moving_average import sma, ema, sema, rma
from .utils import DynamicTimeWarping


def bollinger_bands(
    source: pd.Series,
    length: int,
    mult: float,
    ma_method: Literal["sma", "ema", "dema", "tema", "rma"] = "ema",
) -> pd.DataFrame:
    match ma_method:
        case "sma":
            basis = sma(source, length)
        case "ema":
            basis = ema(source, length)
        case "dema":
            basis = sema(source, length, 2)
        case "tema":
            basis = sema(source, length, 3)
        case "rma":
            basis = rma(source, length)
        case _:
            raise InvalidArgumentError(
                "ma_method must be 'sma', 'ema', 'dema', 'tema', or 'rma',"
                f" got '{ma_method}'."
            )

    deviation = mult * source.rolling(window=length).std()

    return pd.DataFrame(
        {
            "basis": basis,
            "upper": basis + deviation,
            "lower": basis - deviation,
        }
    )


def bollinger_trends(
    source: pd.Series,
    short_length: int = 20,
    long_length: int = 50,
    mult: float = 2,
    ma_method: Literal["sma", "ema", "dema", "tema", "rma"] = "sma",
    stdev_method: Literal["absolute", "ratio", "dtw"] = "absolute",
    diff_method: Literal["normal", "absolute", "ratio", "dtw"] = "normal",
    based_on: Literal["short_length", "long_length"] = "short_length",
) -> pd.Series:
    short_bands = bollinger_bands(source, short_length, mult, ma_method)
    long_bands = bollinger_bands(source, long_length, mult, ma_method)

    short_lower = short_bands["lower"]
    short_upper = short_bands["upper"]

    long_lower = long_bands["lower"]
    long_upper = long_bands["upper"]

    match based_on:
        case "short_length":
            middle = short_bands["basis"]
        case "long_length":
            middle = long_bands["basis"]
        case _:
            raise InvalidArgumentError(
                "based_on must be 'short_length' or 'long_length'."
                f" got '{based_on}'."
            )

    short_length_index = short_length - 1

    match stdev_method:
        case "absolute":
            lower_diff = abs(short_lower - long_lower)
            upper_diff = abs(short_upper - long_upper)

        case "ratio":
            lower_diff = abs(short_lower / long_lower)
            upper_diff = abs(short_upper / long_upper)

        case "dtw":
            short_lower = short_lower.iloc[short_length_index:]
            short_upper = short_upper.iloc[short_length_index:]

            long_lower = long_lower.iloc[short_length_index:]
            long_upper = long_upper.iloc[short_length_index:]

            lower_diff = abs(
                DynamicTimeWarping(short_lower, long_lower)
                .calculate_dtw_distance("ratio", True)
            )

            upper_diff = abs(
                DynamicTimeWarping(short_upper, long_upper)
                .calculate_dtw_distance("ratio", True)
            )
        case _:
            raise InvalidArgumentError(
                "stdev_method must be 'absolute', 'ratio', or 'dtw'."
                f" got '{stdev_method}'."
            )

    match diff_method:
        case "normal":
            return (
                (lower_diff - upper_diff) / middle * 100
            ).rename("Bollinger Trend")
        case "absolute":
            return (
                (lower_diff - upper_diff) - middle * 100
            ).rename("Bollinger Trend")
        case "ratio":
            return (
                (lower_diff / upper_diff) / middle * 100
            ).rename("Bollinger Trend")
        case "dtw":
            middle = middle.iloc[short_length_index:]

            distance_diff = (
                DynamicTimeWarping(lower_diff, upper_diff)
                .calculate_dtw_distance("ratio", True)
            ) * 100

            null_series = pd.Series(
                np.full(short_length_index, np.nan),
                index=range(0, short_length_index),
            )

            return pd.concat(
                [
                    null_series, (
                        DynamicTimeWarping(distance_diff, middle)
                        .calculate_dtw_distance("ratio", True)  * 100
                    )
                ]
            ).rename("Bollinger Trend")

        case _:
            raise InvalidArgumentError(
                "diff_method must be 'normal', 'absolute', 'ratio', or 'dtw'."
                f" got '{diff_method}'."
            )