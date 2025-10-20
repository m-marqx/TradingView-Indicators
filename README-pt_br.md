# TradingView Indicators

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/0ab9ac33df2d4df2a0f8e52237f8a8ad)](https://app.codacy.com/gh/m-marqx/TradingView-Indicators/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/tradingview-indicators.svg)](https://pypi.org/project/tradingview-indicators/)

> Uma biblioteca Python pronta para produção com indicadores precisos de análise técnica financeira, desenvolvida para corresponder aos padrões da plataforma TradingView.

[🇺🇸 English Version](https://github.com/m-marqx/TradingView-Indicators/blob/main/README.md)

---

## 🎯 Visão Geral

**TradingView Indicators** é uma biblioteca Python open-source que fornece implementações altamente precisas de indicadores de análise técnica utilizados nos mercados financeiros. Construída com as melhores práticas de engenharia de dados, esta biblioteca resolve problemas críticos de precisão encontrados em soluções existentes como TA-Lib e pandas-ta.

### Características Principais

- ✅ **Cálculos Precisos** - Valores que correspondem exatamente aos padrões da plataforma TradingView
- ⚡ **Performance Otimizada** - Operações vetorizadas usando NumPy e Pandas para processar mais de 1M de pontos de dados
- 🧩 **Arquitetura Modular** - Código limpo e de fácil manutenção seguindo princípios SOLID
- 🧪 **Testes Abrangentes** - Mais de 12 suítes de testes garantindo confiabilidade e precisão
- 📚 **Bem Documentado** - Documentação clara da API com exemplos funcionais
- 🔧 **Type-Safe** - Suporte completo a type hints para melhor integração com IDEs

---

## 📦 Instalação

Instale via pip:

```bash
pip install tradingview-indicators
```

### Requisitos

- Python 3.11+
- pandas >= 2.3.3
- numpy >= 2.3.4

---

## 🚀 Início Rápido

```python
import pandas as pd
import tradingview_indicators as ta

# Carregue seus dados de mercado
df = pd.read_csv("BTCUSDT_1d_spot.csv")
source = df["close"]

# Calcule indicadores
df["EMA_14"] = ta.ema(source, 14)
df["RSI_14"] = ta.RSI(source, 14)

# MACD com histograma
macd = ta.MACD(source, 12, 26, 9)
df["MACD_Histogram"] = macd.get_histogram

# Índice de Movimento Direcional
dmi = ta.DMI(df, "close")
df["ADX"] = dmi.adx()[0]
df["DI_Plus"] = dmi.adx()[1]
df["DI_Minus"] = dmi.adx()[2]

# Bandas de Bollinger
bb = ta.bollinger_bands(source, 20, 2)
df["BB_Upper"] = bb[0]
df["BB_Middle"] = bb[1]
df["BB_Lower"] = bb[2]
```

---

## 📊 Indicadores Disponíveis

| Indicador | Função | Descrição |
|-----------|----------|-------------|
| **Médias Móveis** | `sma()`, `ema()`, `rma()`, `sema()` | Médias Móveis Simples, Exponencial, Rolling e Suavizada |
| **RSI** | `RSI()` | Índice de Força Relativa com múltiplos métodos de MA |
| **MACD** | `MACD()` | Convergência e Divergência de Médias Móveis |
| **Bandas de Bollinger** | `bollinger_bands()`, `bollinger_trends()` | Bandas de volatilidade e análise de tendência |
| **Estocástico** | `stoch()`, `slow_stoch()` | Osciladores de momentum |
| **DMI/ADX** | `DMI()` | Índice de Movimento Direcional e Índice Direcional Médio |
| **CCI** | `CCI()` | Índice de Canal de Commodity |
| **Ichimoku** | `Ichimoku()` | Indicador Nuvem de Ichimoku |
| **TRIX** | `TRIX()` | Média Exponencial Tripla |
| **TSI** | `tsi()` | Índice de Força Verdadeira |
| **SMIO** | `SMIO()` | Oscilador SMI |
| **Índice Didi** | `didi_index()` | Indicador brasileiro de análise técnica |

---

## 📁 Estrutura do Repositório

```
TradingView-Indicators/
├── src/
│   └── tradingview_indicators/    # Código fonte principal do pacote
│       ├── __init__.py            # Inicialização e exportações do pacote
│       ├── moving_average.py      # Implementações de MA (SMA, EMA, RMA, SEMA)
│       ├── RSI.py                 # Índice de Força Relativa
│       ├── MACD.py                # Indicador MACD
│       ├── bollinger.py           # Bandas de Bollinger
│       ├── stoch.py               # Oscilador Estocástico
│       ├── slow_stoch.py          # Estocástico Lento
│       ├── DMI.py                 # Índice de Movimento Direcional
│       ├── CCI.py                 # Índice de Canal de Commodity
│       ├── ichimoku.py            # Nuvem de Ichimoku
│       ├── TRIX.py                # Média Exponencial Tripla
│       ├── tsi.py                 # Índice de Força Verdadeira
│       ├── SMIO.py                # Oscilador SMI
│       ├── didi_index.py          # Índice Didi
│       ├── utils.py               # Funções utilitárias
│       └── errors_exceptions.py   # Exceções customizadas
│
├── tests/                         # Suíte de testes abrangente
│   ├── test_moving_average.py
│   ├── test_RSI.py
│   ├── test_macd.py
│   ├── test_bollinger_bands.py
│   └── ...                        # Arquivos de teste adicionais
│
├── example/                       # Exemplos de uso
│   ├── example.ipynb              # Notebook Jupyter com exemplos
│   └── BTCUSDT_1d_spot.csv        # Dados de mercado de exemplo
│
├── pyproject.toml                 # Metadados do projeto e dependências
├── setup.py                       # Configuração do pacote
├── requirements.txt               # Dependências de desenvolvimento
└── README.md                      # Este arquivo
```

### Componentes Principais

#### 🔧 Código Fonte (`src/tradingview_indicators/`)
O pacote principal contém implementações modulares de indicadores:
- Cada indicador está em seu próprio arquivo para facilitar a manutenção
- Design de API consistente em todos os indicadores
- Type hints para melhor qualidade de código e suporte de IDE
- Tratamento de erros customizado para validação de dados

#### 🧪 Testes (`tests/`)
Framework de testes abrangente garantindo precisão:
- Testes unitários para cada indicador
- Validação contra saídas do TradingView
- Tratamento de casos extremos
- Testes de regressão

#### 📚 Exemplos (`example/`)
Demonstrações práticas de uso:
- Notebook Jupyter com exemplos do mundo real
- Dados de exemplo do mercado de criptomoedas
- Guias passo a passo de cálculo

---

## 🛠️ Arquitetura Técnica

### Princípios de Design

1. **Modularidade** - Cada indicador é independente e reutilizável
2. **Precisão** - Todos os cálculos validados contra os padrões do TradingView
3. **Performance** - Operações vetorizadas para processamento eficiente em larga escala
4. **Manutenibilidade** - Código limpo seguindo PEP 8 e princípios SOLID
5. **Type Safety** - Type hints abrangentes para melhor experiência do desenvolvedor

### Pipeline de Processamento de Dados

```
Dados Brutos de Mercado (DataFrame/Series)
          ↓
   Validação de Dados
          ↓
  Cálculos Vetorizados (NumPy/Pandas)
          ↓
   Transformação de Resultado
          ↓
   Saída Type-Safe (Series/DataFrame)
```

---

## 🎓 Casos de Uso

- **Trading Algorítmico** - Construa estratégias de trading confiáveis com indicadores precisos
- **Análise Financeira** - Realize análise técnica em ações, cripto e forex
- **Data Science** - Integre com pipelines de dados para pesquisa de mercado
- **Backtesting** - Teste estratégias de trading com cálculos históricos precisos
- **Educação** - Aprenda análise técnica com implementações de nível profissional

---

## 🧪 Testes

Execute a suíte de testes:

```bash
pytest tests/
```

A biblioteca inclui mais de 12 suítes de testes abrangentes cobrindo:
- Validação de precisão de indicadores
- Tratamento de casos extremos
- Validação de tipos de dados
- Benchmarks de performance

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Este projeto segue padrões profissionais de desenvolvimento:

1. Faça um fork do repositório
2. Crie uma branch de feature (`git checkout -b feature/indicador-incrivel`)
3. Escreva testes para suas mudanças
4. Garanta que todos os testes passem (`pytest`)
5. Siga as diretrizes de estilo PEP 8
6. Envie um pull request

---

## 📄 Licença

[Sua Licença Aqui]

---

## 🔗 Links

- **Pacote PyPI**: [tradingview-indicators](https://pypi.org/project/tradingview-indicators/)
- **Repositório GitHub**: [m-marqx/TradingView-Indicators](https://github.com/m-marqx/TradingView-Indicators)
- **Rastreador de Issues**: [GitHub Issues](https://github.com/m-marqx/TradingView-Indicators/issues)

---

## 📧 Contato

Para perguntas, sugestões ou oportunidades de colaboração, por favor abra uma issue no GitHub.

---

## 🙏 Agradecimentos

Este projeto foi criado para resolver problemas de precisão em bibliotecas de análise técnica existentes e fornecer uma solução pronta para produção para a comunidade de análise de dados financeiros.

**Tecnologias Utilizadas:**
- Python 3.11+
- Pandas (análise de séries temporais)
- NumPy (operações vetorizadas)
- pytest (framework de testes)

---

Feito com ❤️ para a comunidade de análise de dados financeiros
