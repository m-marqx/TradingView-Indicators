Se você quiser ler em português basta [clicar aqui](https://github.com/m-marqx/TradingView-Indicators/blob/main/readme%20-%20pt-br.md)

## Objective

The initial objective of this repository was to create technical analysis indicators in a way that they would be easy to maintain and highly modular. It was observed that the initial values of the indicators, both by [TA-Lib](https://github.com/TA-Lib/ta-lib-python) and pandas, had some inaccuracies and needed adjustments. With this in mind, I created this repository specifically to address this issue of inaccuracy.

## Dependencies

To use TradingView-Indicators, there is a dependency on the `pandas` and `numpy` libraries. To install them if you don't already have them, simply run `pip install -r requirements.txt`.

## Example

```python
from indicators import (
    RSI,
    MACD,
    DMI,
)

df = pd.read_csv("example/BTCUSDT_1d_spot.csv")
source = df["close"].copy()
df["RSI"] = RSI(source, 14)
df["MACD"] = MACD(source, 12, 26, 9).get_histogram
df["ADX"] = DMI(df).adx()[0]
df["DI+"] = DMI(df).adx()[1]
df["DI-"] = DMI(df).adx()[2]
df["DI_Delta"] = DMI(df).di_delta()[0]
df["DI_Ratio"] = DMI(df).di_delta()[1]
```
