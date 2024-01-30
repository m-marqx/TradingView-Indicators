[![Codacy Badge](https://app.codacy.com/project/badge/Grade/0ab9ac33df2d4df2a0f8e52237f8a8ad)](https://app.codacy.com/gh/m-marqx/TradingView-Indicators/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

## Objetivo

O objetivo inicial deste repositório foi de criar indicadores da análise técnica de forma que fossem de fácil manutenção e com um alto grau modularidade, foi observado que os valores iniciais dos indicadores tanto pelo [TA-Lib](https://github.com/TA-Lib/ta-lib-python) quanto pelo pandas havia uma certa imprecisão e precisavam de ajustes, tendo isso em vista eu criei esse repositório justamente para corrigir este problema de imprecisão.

## Instalação
para intalar o TradingView Indicators você precisará utilizar o gerenciador de pacotes [pip](https://pip.pypa.io/en/stable/):

```
pip install tradingview-indicators
```

## Exemplo

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
