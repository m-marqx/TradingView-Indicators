## Objetivo

O objetivo inicial deste repositório foi de criar indicadores da análise técnica de forma que fossem de fácil manutenção e com um alto grau modularidade, foi observado que os valores iniciais dos indicadores tanto pelo [TA-Lib](https://github.com/TA-Lib/ta-lib-python) quanto pelo pandas havia uma certa imprecisão e precisavam de ajustes, tendo isso em vista eu criei esse repositório justamente para corrigir este problema de imprecisão.

## Dependencias

Para usar o TradingView-Indicators possui uma depêndencia com as bibliotecas `pandas` e `numpy` para instalar elas caso não possua basta digitar  ```pip install -r requirements.txt```

## Exemplo

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

