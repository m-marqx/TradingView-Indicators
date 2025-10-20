# TradingView Indicators

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/0ab9ac33df2d4df2a0f8e52237f8a8ad)](https://app.codacy.com/gh/m-marqx/TradingView-Indicators/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/tradingview-indicators.svg)](https://pypi.org/project/tradingview-indicators/)

> A production-ready Python library for accurate financial technical analysis indicators, engineered to match TradingView platform standards.

[🇧🇷 Versão em Português](https://github.com/m-marqx/TradingView-Indicators/blob/main/readme%20-%20pt-br.md)

---

## 🎯 Overview

**TradingView Indicators** is an open-source Python library that provides highly accurate implementations of technical analysis indicators used in financial markets. Built with data engineering best practices, this library addresses critical accuracy issues found in existing solutions like TA-Lib and pandas-ta.

### Key Features

- ✅ **Accurate Calculations** - Values precisely match TradingView platform standards
- ⚡ **Optimized Performance** - Vectorized operations using NumPy and Pandas for processing 1M+ data points
- 🧩 **Modular Architecture** - Clean, maintainable code following SOLID principles
- 🧪 **Comprehensive Testing** - 12+ test suites ensuring reliability and accuracy
- 📚 **Well Documented** - Clear API documentation with working examples
- 🔧 **Type-Safe** - Full type hints support for better IDE integration

---

## 📦 Installation

Install via pip:

```bash
pip install tradingview-indicators
```

### Requirements

- Python 3.11+
- pandas >= 2.3.3
- numpy >= 2.3.4

---

## 🚀 Quick Start

```python
import pandas as pd
import tradingview_indicators as ta

# Load your market data
df = pd.read_csv("BTCUSDT_1d_spot.csv")
source = df["close"]

# Calculate indicators
df["EMA_14"] = ta.ema(source, 14)
df["RSI_14"] = ta.RSI(source, 14)

# MACD with histogram
macd = ta.MACD(source, 12, 26, 9)
df["MACD_Histogram"] = macd.get_histogram

# Directional Movement Index
dmi = ta.DMI(df, "close")
df["ADX"] = dmi.adx()[0]
df["DI_Plus"] = dmi.adx()[1]
df["DI_Minus"] = dmi.adx()[2]

# Bollinger Bands
bb = ta.bollinger_bands(source, 20, 2)
df["BB_Upper"] = bb[0]
df["BB_Middle"] = bb[1]
df["BB_Lower"] = bb[2]
```

---

## 📊 Available Indicators

| Indicator | Function | Description |
|-----------|----------|-------------|
| **Moving Averages** | `sma()`, `ema()`, `rma()`, `sema()` | Simple, Exponential, Rolling, Smoothed Moving Averages |
| **RSI** | `RSI()` | Relative Strength Index with multiple MA methods |
| **MACD** | `MACD()` | Moving Average Convergence Divergence |
| **Bollinger Bands** | `bollinger_bands()`, `bollinger_trends()` | Volatility bands and trend analysis |
| **Stochastic** | `stoch()`, `slow_stoch()` | Momentum oscillators |
| **DMI/ADX** | `DMI()` | Directional Movement Index and Average Directional Index |
| **CCI** | `CCI()` | Commodity Channel Index |
| **Ichimoku** | `Ichimoku()` | Ichimoku Cloud indicator |
| **TRIX** | `TRIX()` | Triple Exponential Average |
| **TSI** | `tsi()` | True Strength Index |
| **SMIO** | `SMIO()` | SMI Oscillator |
| **Didi Index** | `didi_index()` | Brazilian technical analysis indicator |

---

## 📁 Repository Structure

```
TradingView-Indicators/
├── src/
│   └── tradingview_indicators/    # Main package source code
│       ├── __init__.py            # Package initialization and exports
│       ├── moving_average.py      # MA implementations (SMA, EMA, RMA, SEMA)
│       ├── RSI.py                 # Relative Strength Index
│       ├── MACD.py                # MACD indicator
│       ├── bollinger.py           # Bollinger Bands
│       ├── stoch.py               # Stochastic oscillator
│       ├── slow_stoch.py          # Slow Stochastic
│       ├── DMI.py                 # Directional Movement Index
│       ├── CCI.py                 # Commodity Channel Index
│       ├── ichimoku.py            # Ichimoku Cloud
│       ├── TRIX.py                # Triple Exponential Average
│       ├── tsi.py                 # True Strength Index
│       ├── SMIO.py                # SMI Oscillator
│       ├── didi_index.py          # Didi Index
│       ├── utils.py               # Utility functions
│       └── errors_exceptions.py   # Custom exceptions
│
├── tests/                         # Comprehensive test suite
│   ├── test_moving_average.py
│   ├── test_RSI.py
│   ├── test_macd.py
│   ├── test_bollinger_bands.py
│   └── ...                        # Additional test files
│
├── example/                       # Usage examples
│   ├── example.ipynb              # Jupyter notebook with examples
│   └── BTCUSDT_1d_spot.csv        # Sample market data
│
├── pyproject.toml                 # Project metadata and dependencies
├── setup.py                       # Package setup configuration
├── requirements.txt               # Development dependencies
└── README.md                      # This file
```

### Core Components

#### 🔧 Source Code (`src/tradingview_indicators/`)
The main package contains modular indicator implementations:
- Each indicator is in its own file for maintainability
- Consistent API design across all indicators
- Type hints for better code quality and IDE support
- Custom error handling for data validation

#### 🧪 Tests (`tests/`)
Comprehensive testing framework ensuring accuracy:
- Unit tests for each indicator
- Validation against TradingView outputs
- Edge case handling
- Regression testing

#### 📚 Examples (`example/`)
Practical usage demonstrations:
- Jupyter notebook with real-world examples
- Sample cryptocurrency market data
- Step-by-step calculation guides

---

## 🛠️ Technical Architecture

### Design Principles

1. **Modularity** - Each indicator is self-contained and reusable
2. **Accuracy** - All calculations validated against TradingView standards
3. **Performance** - Vectorized operations for efficient large-scale processing
4. **Maintainability** - Clean code following PEP 8 and SOLID principles
5. **Type Safety** - Comprehensive type hints for better developer experience

### Data Processing Pipeline

```
Raw Market Data (DataFrame/Series)
          ↓
   Data Validation
          ↓
  Vectorized Calculations (NumPy/Pandas)
          ↓
   Result Transformation
          ↓
   Type-Safe Output (Series/DataFrame)
```

---

## 🎓 Use Cases

- **Algorithmic Trading** - Build reliable trading strategies with accurate indicators
- **Financial Analysis** - Perform technical analysis on stocks, crypto, and forex
- **Data Science** - Integrate with data pipelines for market research
- **Backtesting** - Test trading strategies with precise historical calculations
- **Education** - Learn technical analysis with production-grade implementations

---

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

The library includes 12+ comprehensive test suites covering:
- Indicator accuracy validation
- Edge case handling
- Data type validation
- Performance benchmarks

---

## 🤝 Contributing

Contributions are welcome! This project follows professional development standards:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-indicator`)
3. Write tests for your changes
4. Ensure all tests pass (`pytest`)
5. Follow PEP 8 style guidelines
6. Submit a pull request

---

## 📄 License

[Your License Here]

---

## 🔗 Links

- **PyPI Package**: [tradingview-indicators](https://pypi.org/project/tradingview-indicators/)
- **GitHub Repository**: [m-marqx/TradingView-Indicators](https://github.com/m-marqx/TradingView-Indicators)
- **Issue Tracker**: [GitHub Issues](https://github.com/m-marqx/TradingView-Indicators/issues)

---

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub.

---

## 🙏 Acknowledgments

This project was created to address accuracy issues in existing technical analysis libraries and provide a production-ready solution for the financial data analysis community.

**Technologies Used:**
- Python 3.11+
- Pandas (time-series analysis)
- NumPy (vectorized operations)
- pytest (testing framework)

---

Made with ❤️ for the financial data analysis community
