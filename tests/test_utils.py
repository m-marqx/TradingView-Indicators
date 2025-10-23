import unittest
import pandas as pd
import numpy as np
from src.tradingview_indicators.utils import DynamicTimeWarping, OHLC_finder
from src.tradingview_indicators.errors_exceptions import InvalidArgumentError


class TestDynamicTimeWarping(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(seed=1337)
        self.input_x = pd.Series(
            rng.uniform(10, 100, 20),
            name="signal_x"
        )
        self.input_y = pd.Series(
            rng.uniform(15, 95, 20),
            name="signal_y"
        )
        self.dtw = DynamicTimeWarping(self.input_x, self.input_y)

    def test_dtw_initialization(self):
        self.assertIsInstance(self.dtw, DynamicTimeWarping)
        self.assertEqual(self.dtw.column_x, "signal_x")
        self.assertEqual(self.dtw.column_y, "signal_y")
        pd.testing.assert_series_equal(self.dtw.input_x, self.input_x)
        pd.testing.assert_series_equal(self.dtw.input_y, self.input_y)

    def test_dtw_initialization_with_numpy_arrays(self):
        rng = np.random.default_rng(seed=2048)
        arr_x = rng.uniform(5, 50, 15)
        arr_y = rng.uniform(10, 55, 15)

        dtw = DynamicTimeWarping(arr_x, arr_y)

        self.assertEqual(dtw.column_x, "input_x")
        self.assertEqual(dtw.column_y, "input_y")

    def test_dtw_df_structure(self):
        dtw_df = self.dtw.dtw_df

        expected_columns = [
            "signal_x_path",
            "signal_y_path",
            "signal_x",
            "signal_y"
        ]

        self.assertIsInstance(dtw_df, pd.DataFrame)
        self.assertListEqual(list(dtw_df.columns), expected_columns)
        self.assertGreater(len(dtw_df), 0)

    def test_dtw_df_values(self):
        dtw_df = self.dtw.dtw_df

        self.assertTrue(dtw_df["signal_x_path"].max() < len(self.input_x))
        self.assertTrue(dtw_df["signal_y_path"].max() < len(self.input_y))
        self.assertTrue(dtw_df["signal_x_path"].min() >= 0)
        self.assertTrue(dtw_df["signal_y_path"].min() >= 0)

    def test_calculate_dtw_distance_ratio(self):
        result = self.dtw.calculate_dtw_distance(method="ratio")

        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.dtw.dtw_df))
        self.assertTrue((result > 0).all())

    def test_calculate_dtw_distance_absolute(self):
        result = self.dtw.calculate_dtw_distance(method="absolute")

        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.dtw.dtw_df))

    def test_calculate_dtw_distance_invalid_method(self):
        with self.assertRaises(InvalidArgumentError) as context:
            self.dtw.calculate_dtw_distance(method="invalid_method")

        self.assertIn("method must be 'ratio' or 'absolute'", str(context.exception))

    def test_calculate_dtw_distance_with_alignment(self):
        result = self.dtw.calculate_dtw_distance(
            method="ratio",
            align_sequences=True
        )

        self.assertIsInstance(result, pd.Series)
        self.assertGreater(len(result), 0)

    def test_align_dtw_distance_equal_length(self):
        rng = np.random.default_rng(seed=4096)
        x = pd.Series(rng.uniform(20, 80, 25), name="x")
        y = pd.Series(rng.uniform(25, 75, 25), name="y")

        dtw = DynamicTimeWarping(x, y)
        x_aligned, y_aligned = dtw.align_dtw_distance()

        self.assertIsInstance(x_aligned, pd.Series)
        self.assertIsInstance(y_aligned, pd.Series)
        self.assertEqual(len(x_aligned), len(y_aligned))

    def test_align_dtw_distance_different_length_x_longer(self):
        rng = np.random.default_rng(seed=8192)
        x = pd.Series(rng.uniform(10, 90, 30), name="x_long")
        y = pd.Series(rng.uniform(15, 85, 20), name="y_short")

        dtw = DynamicTimeWarping(x, y)
        x_aligned, y_aligned = dtw.align_dtw_distance()

        self.assertEqual(len(x_aligned), len(y_aligned))
        self.assertEqual(len(x_aligned), len(y))

    def test_align_dtw_distance_different_length_y_longer(self):
        rng = np.random.default_rng(seed=16384)
        x = pd.Series(rng.uniform(10, 90, 18), name="x_short")
        y = pd.Series(rng.uniform(15, 85, 28), name="y_long")

        dtw = DynamicTimeWarping(x, y)
        x_aligned, y_aligned = dtw.align_dtw_distance()

        self.assertEqual(len(x_aligned), len(y_aligned))
        self.assertEqual(len(y_aligned), len(x))

    def test_dtw_with_similar_patterns(self):
        rng = np.random.default_rng(seed=32768)
        t = np.linspace(0, 4 * np.pi, 50)
        x = pd.Series(np.sin(t) + rng.normal(0, 0.1, 50), name="sine_x")
        y = pd.Series(np.sin(t + 0.5) + rng.normal(0, 0.1, 50), name="sine_y")

        dtw = DynamicTimeWarping(x, y)
        ratio = dtw.calculate_dtw_distance(method="ratio", align_sequences=True)

        mean_ratio = np.nanmean(np.abs(ratio.values))
        self.assertIsNotNone(mean_ratio)

    def test_dtw_with_single_element(self):
        rng = np.random.default_rng(seed=65536)
        x = pd.Series([rng.uniform(10, 100)], name="single_x")
        y = pd.Series([rng.uniform(10, 100)], name="single_y")

        dtw = DynamicTimeWarping(x, y)
        dtw_df = dtw.dtw_df

        self.assertEqual(len(dtw_df), 1)


class TestOHLCFinder(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(seed=42069)

        base_price = 100
        self.n_rows = 50

        close_prices = base_price + rng.normal(0, 5, self.n_rows).cumsum()
        high_prices = close_prices + rng.uniform(0.5, 3, self.n_rows)
        low_prices = close_prices - rng.uniform(0.5, 3, self.n_rows)
        open_prices = close_prices + rng.uniform(-1, 1, self.n_rows)

        self.df_lowercase = pd.DataFrame({
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
        })

        self.df_uppercase = pd.DataFrame({
            "Open": open_prices,
            "High": high_prices,
            "Low": low_prices,
            "Close": close_prices,
        })

        self.df_custom = pd.DataFrame({
            "price_open": open_prices,
            "price_high": high_prices,
            "price_low": low_prices,
            "price_close": close_prices,
        })

    def test_ohlc_finder_lowercase_columns(self):
        open_p, high_p, low_p, close_p = OHLC_finder(self.df_lowercase)

        pd.testing.assert_series_equal(open_p, self.df_lowercase["open"])
        pd.testing.assert_series_equal(high_p, self.df_lowercase["high"])
        pd.testing.assert_series_equal(low_p, self.df_lowercase["low"])
        pd.testing.assert_series_equal(close_p, self.df_lowercase["close"])

    def test_ohlc_finder_uppercase_columns(self):
        open_p, high_p, low_p, close_p = OHLC_finder(self.df_uppercase)

        pd.testing.assert_series_equal(open_p, self.df_uppercase["Open"])
        pd.testing.assert_series_equal(high_p, self.df_uppercase["High"])
        pd.testing.assert_series_equal(low_p, self.df_uppercase["Low"])
        pd.testing.assert_series_equal(close_p, self.df_uppercase["Close"])

    def test_ohlc_finder_custom_columns(self):
        open_p, high_p, low_p, close_p = OHLC_finder(
            self.df_custom,
            Open="price_open",
            High="price_high",
            Low="price_low",
            Close="price_close"
        )

        pd.testing.assert_series_equal(open_p, self.df_custom["price_open"])
        pd.testing.assert_series_equal(high_p, self.df_custom["price_high"])
        pd.testing.assert_series_equal(low_p, self.df_custom["price_low"])
        pd.testing.assert_series_equal(close_p, self.df_custom["price_close"])

    def test_ohlc_finder_partial_custom_columns(self):
        df_mixed = self.df_lowercase.copy()
        df_mixed["volume"] = np.random.default_rng(seed=31415).uniform(1000, 10000, self.n_rows)

        open_p, high_p, low_p, close_p = OHLC_finder(
            df_mixed,
            Close="close"
        )

        pd.testing.assert_series_equal(open_p, df_mixed["open"])
        pd.testing.assert_series_equal(high_p, df_mixed["high"])
        pd.testing.assert_series_equal(low_p, df_mixed["low"])
        pd.testing.assert_series_equal(close_p, df_mixed["close"])

    def test_ohlc_finder_missing_columns_raises_error(self):
        df_no_ohlc = pd.DataFrame({
            "price": np.random.default_rng(seed=27182).uniform(90, 110, 20),
            "volume": np.random.default_rng(seed=27183).uniform(1000, 10000, 20)
        })

        with self.assertRaises(ValueError) as context:
            OHLC_finder(df_no_ohlc)

        self.assertIn("OHLC columns not found", str(context.exception))

    def test_ohlc_finder_not_dataframe_raises_error(self):
        not_a_df = [1, 2, 3, 4]

        with self.assertRaises(ValueError) as context:
            OHLC_finder(not_a_df)

        self.assertIn("dataframe param must be a DataFrame", str(context.exception))

    def test_ohlc_finder_with_none_values(self):
        open_p, high_p, low_p, close_p = OHLC_finder(
            self.df_uppercase,
            Open=None,
            High=None,
            Low=None,
            Close=None
        )

        pd.testing.assert_series_equal(open_p, self.df_uppercase["Open"])
        pd.testing.assert_series_equal(high_p, self.df_uppercase["High"])
        pd.testing.assert_series_equal(low_p, self.df_uppercase["Low"])
        pd.testing.assert_series_equal(close_p, self.df_uppercase["Close"])

    def test_ohlc_finder_uppercase_priority(self):
        rng = np.random.default_rng(seed=12345)
        df_both = pd.DataFrame({
            "Open": rng.uniform(100, 110, 10),
            "High": rng.uniform(110, 120, 10),
            "Low": rng.uniform(90, 100, 10),
            "Close": rng.uniform(100, 110, 10),
            "open": rng.uniform(50, 60, 10),
            "high": rng.uniform(60, 70, 10),
            "low": rng.uniform(40, 50, 10),
            "close": rng.uniform(50, 60, 10),
        })

        open_p, high_p, low_p, close_p = OHLC_finder(df_both)

        pd.testing.assert_series_equal(open_p, df_both["Open"])
        pd.testing.assert_series_equal(high_p, df_both["High"])
        pd.testing.assert_series_equal(low_p, df_both["Low"])
        pd.testing.assert_series_equal(close_p, df_both["Close"])

    def test_ohlc_finder_with_additional_columns(self):
        rng = np.random.default_rng(seed=98765)
        df_extra = self.df_lowercase.copy()
        df_extra["volume"] = rng.uniform(1000, 10000, self.n_rows)
        df_extra["timestamp"] = pd.date_range("2023-01-01", periods=self.n_rows)
        df_extra["symbol"] = "BTC/USD"

        open_p, high_p, low_p, close_p = OHLC_finder(df_extra)

        pd.testing.assert_series_equal(open_p, df_extra["open"])
        pd.testing.assert_series_equal(high_p, df_extra["high"])
        pd.testing.assert_series_equal(low_p, df_extra["low"])
        pd.testing.assert_series_equal(close_p, df_extra["close"])

    def test_ohlc_finder_single_row(self):
        rng = np.random.default_rng(seed=11111)
        df_single = pd.DataFrame({
            "open": [rng.uniform(100, 110)],
            "high": [rng.uniform(110, 120)],
            "low": [rng.uniform(90, 100)],
            "close": [rng.uniform(100, 110)],
        })

        open_p, high_p, low_p, close_p = OHLC_finder(df_single)

        self.assertEqual(len(open_p), 1)
        self.assertEqual(len(high_p), 1)
        self.assertEqual(len(low_p), 1)
        self.assertEqual(len(close_p), 1)

    def test_ohlc_finder_preserves_index(self):
        custom_index = pd.date_range("2023-01-01", periods=self.n_rows, freq="D")
        df_indexed = self.df_lowercase.copy()
        df_indexed.index = custom_index

        open_p, high_p, low_p, close_p = OHLC_finder(df_indexed)

        pd.testing.assert_index_equal(open_p.index, custom_index)
        pd.testing.assert_index_equal(high_p.index, custom_index)
        pd.testing.assert_index_equal(low_p.index, custom_index)
        pd.testing.assert_index_equal(close_p.index, custom_index)

