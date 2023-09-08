## PT-BR
Se você quiser ler em português basta [clicar aqui](https://github.com/m-marqx/TradingView-Indicators/blob/main/readme%20-%20pt-br.md)

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
df["RSI"] = ta.RSI(source, 14)
df["MACD"] = ta.MACD(source, 12, 26, 9).get_histogram

dmi = ta.DMI(df, "close")

df["ADX"] = dmi.adx()[0]
df["DI+"] = dmi.adx()[1]
df["DI-"] = dmi.adx()[2]
df["DI_Delta"] = dmi.di_delta()[0]
df["DI_Ratio"] = dmi.di_delta()[1]
```
