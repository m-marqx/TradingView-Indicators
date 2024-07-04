from typing import Literal
import pandas as pd
from .moving_average import sma, ema, sema, rma
from .utils import DynamicTimeWarping

def bollinger_bands(
    source:pd.Series,
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

    deviation = mult * source.rolling(window=length).std()

    return pd.DataFrame({
        "basis": basis,
        "upper": basis + deviation,
        "lower": basis - deviation
    })

