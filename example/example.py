import pandas as pd
import tradingview_indicators as ta

df = pd.read_csv("BTCUSDT_1d_spot.csv")
source = df["close"].copy()
df["RSI"] = ta.RSI(source, 14)
df["MACD"] = ta.MACD(source, 12, 26, 9).get_histogram

dmi = ta.DMI(df, "close")

df["ADX"] = dmi.adx()[0]
df["DI+"] = dmi.adx()[1]
df["DI-"] = dmi.adx()[2]
df["DI_Delta"] = dmi.di_delta()[0]
df["DI_Ratio"] = dmi.di_delta()[1]
df
