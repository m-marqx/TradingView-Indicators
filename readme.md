## PT-BR
Se você quiser ler em português basta [clicar aqui](https://github.com/m-marqx/TradingView-Indicators/blob/main/readme%20-%20pt-br.md)

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/0ab9ac33df2d4df2a0f8e52237f8a8ad)](https://app.codacy.com/gh/m-marqx/TradingView-Indicators/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

## Objective

The initial objective of this repository was to create technical analysis indicators in a way that they would be easy to maintain and highly modular. It was observed that the initial values of the indicators, both by [TA-Lib](https://github.com/TA-Lib/ta-lib-python) and pandas, had some inaccuracies and needed adjustments. With this in mind, I created this repository specifically to address this issue of inaccuracy.

## Installation

To install TradingView Indicators, you need to use the package manager [pip](https://pip.pypa.io/en/stable/):

```
pip install tradingview-indicators
```


## Example

```python
import pandas as pd
import tradingview_indicators as ta

df = pd.read_csv("BTCUSDT_1d_spot.csv")
source = df["close"].copy()
df["EMA"] = ta.ema(source, 14)
df["RSI"] = ta.RSI(source, 14)
df["MACD"] = ta.MACD(source, 12, 26, 9).get_histogram

dmi = ta.DMI(df, "close")

df["ADX"] = dmi.adx()[0]
df["DI+"] = dmi.adx()[1]
df["DI-"] = dmi.adx()[2]
df["DI_Delta"] = dmi.di_difference()[0]
df["DI_Ratio"] = dmi.di_difference()[1]
```
