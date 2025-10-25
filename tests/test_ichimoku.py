import unittest
import pandas as pd
import numpy as np
from src.tradingview_indicators.ichimoku import Ichimoku


class TestIchimoku(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(seed=54321)

        base_price = 30000
        self.n_rows = 100

        close_prices = base_price + rng.normal(0, 300, self.n_rows).cumsum()
        high_prices = close_prices + rng.uniform(50, 400, self.n_rows)
        low_prices = close_prices - rng.uniform(50, 400, self.n_rows)
        open_prices = close_prices + rng.uniform(-150, 150, self.n_rows)

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

        self.conversion_periods = 9
        self.base_periods = 26
        self.lagging_span_2_periods = 52
        self.displacement = 26

    def test_ichimoku_returns_dataframe(self):
        result = Ichimoku(
            self.df_lowercase,
            self.conversion_periods,
            self.base_periods,
            self.lagging_span_2_periods,
            self.displacement
        )

        self.assertIsInstance(result, pd.DataFrame)

    def test_ichimoku_column_structure(self):
        result = Ichimoku(
            self.df_lowercase,
            self.conversion_periods,
            self.base_periods,
            self.lagging_span_2_periods,
            self.displacement
        )

        expected_columns = [
            "conversion_line",
            "base_line",
            "lagging_span",
            "lead_line1",
            "lead_line2",
            "leading_span_a",
            "leading_span_b"
        ]

        self.assertListEqual(list(result.columns), expected_columns)

    def test_ichimoku_output_length(self):
        result = Ichimoku(
            self.df_uppercase,
            self.conversion_periods,
            self.base_periods,
            self.lagging_span_2_periods,
            self.displacement
        )

        self.assertEqual(len(result), len(self.df_uppercase))

    def test_ichimoku_with_standard_parameters(self):
        result = Ichimoku(
            self.df_lowercase,
            conversion_periods=9,
            base_periods=26,
            lagging_span_2_periods=52,
            displacement=26
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result.columns), 7)

    def test_ichimoku_conversion_line_calculation(self):
        result = Ichimoku(
            self.df_lowercase,
            self.conversion_periods,
            self.base_periods,
            self.lagging_span_2_periods,
            self.displacement
        )

        conversion_line = result["conversion_line"]

        high = self.df_lowercase["high"]
        low = self.df_lowercase["low"]

        valid_indices = conversion_line.dropna().index

        for idx in valid_indices:
            start_idx = max(0, idx - self.conversion_periods + 1)
            window_high = high.iloc[start_idx:idx+1].max()
            window_low = low.iloc[start_idx:idx+1].min()

            self.assertGreaterEqual(conversion_line.loc[idx], window_low)
            self.assertLessEqual(conversion_line.loc[idx], window_high)

    def test_ichimoku_base_line_calculation(self):
        result = Ichimoku(
            self.df_lowercase,
            self.conversion_periods,
            self.base_periods,
            self.lagging_span_2_periods,
            self.displacement
        )

        base_line = result["base_line"]

        self.assertTrue(base_line.notna().any())

    def test_ichimoku_lead_line1_relationship(self):
        result = Ichimoku(
            self.df_lowercase,
            self.conversion_periods,
            self.base_periods,
            self.lagging_span_2_periods,
            self.displacement
        )

        conversion_line = result["conversion_line"]
        base_line = result["base_line"]
        lead_line1 = result["lead_line1"]

        valid_mask = conversion_line.notna() & base_line.notna()
        expected_lead1 = (conversion_line + base_line) / 2

        pd.testing.assert_series_equal(
            lead_line1[valid_mask],
            expected_lead1[valid_mask],
            check_names=False
        )

    def test_ichimoku_leading_span_a_displacement(self):
        result = Ichimoku(
            self.df_lowercase,
            self.conversion_periods,
            self.base_periods,
            self.lagging_span_2_periods,
            self.displacement
        )

        lead_line1 = result["lead_line1"]
        leading_span_a = result["leading_span_a"]

        expected_leading_a = lead_line1.shift(self.displacement - 1)

        pd.testing.assert_series_equal(
            leading_span_a.dropna(),
            expected_leading_a.dropna(),
            check_names=False
        )

    def test_ichimoku_leading_span_b_displacement(self):
        result = Ichimoku(
            self.df_lowercase,
            self.conversion_periods,
            self.base_periods,
            self.lagging_span_2_periods,
            self.displacement
        )

        lead_line2 = result["lead_line2"]
        leading_span_b = result["leading_span_b"]

        expected_leading_b = lead_line2.shift(self.displacement - 1)

        pd.testing.assert_series_equal(
            leading_span_b.dropna(),
            expected_leading_b.dropna(),
            check_names=False
        )

    def test_ichimoku_lagging_span_displacement(self):
        result = Ichimoku(
            self.df_lowercase,
            self.conversion_periods,
            self.base_periods,
            self.lagging_span_2_periods,
            self.displacement
        )

        lagging_span = result["lagging_span"]
        close = self.df_lowercase["close"]

        expected_lagging = close.shift(-self.displacement + 1)

        pd.testing.assert_series_equal(
            lagging_span.dropna(),
            expected_lagging.dropna(),
            check_names=False
        )

    def test_ichimoku_with_custom_parameters(self):
        result = Ichimoku(
            self.df_uppercase,
            conversion_periods=12,
            base_periods=30,
            lagging_span_2_periods=60,
            displacement=30
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result.columns), 7)

    def test_ichimoku_with_short_periods(self):
        result = Ichimoku(
            self.df_lowercase,
            conversion_periods=3,
            base_periods=6,
            lagging_span_2_periods=12,
            displacement=6
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result["conversion_line"].notna().sum() > 90)

    def test_ichimoku_with_long_periods(self):
        result = Ichimoku(
            self.df_uppercase,
            conversion_periods=20,
            base_periods=50,
            lagging_span_2_periods=80,
            displacement=50
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result["conversion_line"].notna().sum() < len(self.df_uppercase))

    def test_ichimoku_with_minimal_data(self):
        rng = np.random.default_rng(seed=135790)
        n_rows = 60

        close_prices = 100 + rng.normal(0, 5, n_rows).cumsum()
        high_prices = close_prices + rng.uniform(0.5, 2, n_rows)
        low_prices = close_prices - rng.uniform(0.5, 2, n_rows)
        open_prices = close_prices + rng.uniform(-1, 1, n_rows)

        df_minimal = pd.DataFrame({
            "open": open_prices,
            "close": close_prices,
            "high": high_prices,
            "low": low_prices,
        })

        result = Ichimoku(
            df_minimal,
            conversion_periods=9,
            base_periods=26,
            lagging_span_2_periods=52,
            displacement=26
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), n_rows)

    def test_ichimoku_preserves_dataframe_index(self):
        custom_index = pd.date_range("2023-01-01", periods=self.n_rows, freq="h")
        df_indexed = self.df_lowercase.copy()
        df_indexed.index = custom_index

        result = Ichimoku(
            df_indexed,
            self.conversion_periods,
            self.base_periods,
            self.lagging_span_2_periods,
            self.displacement
        )

        pd.testing.assert_index_equal(result.index, custom_index)

    def test_ichimoku_with_uppercase_columns(self):
        result = Ichimoku(
            self.df_uppercase,
            self.conversion_periods,
            self.base_periods,
            self.lagging_span_2_periods,
            self.displacement
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result.columns), 7)

    def test_ichimoku_cloud_lines_order(self):
        result = Ichimoku(
            self.df_lowercase,
            self.conversion_periods,
            self.base_periods,
            self.lagging_span_2_periods,
            self.displacement
        )

        leading_span_a = result["leading_span_a"]
        leading_span_b = result["leading_span_b"]

        self.assertTrue(leading_span_a.notna().any())
        self.assertTrue(leading_span_b.notna().any())

        self.assertTrue(pd.api.types.is_numeric_dtype(leading_span_a))
        self.assertTrue(pd.api.types.is_numeric_dtype(leading_span_b))

    def test_ichimoku_with_trending_data(self):
        rng = np.random.default_rng(seed=246810)
        n_rows = 100

        trend = np.linspace(0, 100, n_rows)
        close_prices = 1000 + trend + rng.normal(0, 5, n_rows)
        high_prices = close_prices + rng.uniform(1, 5, n_rows)
        low_prices = close_prices - rng.uniform(1, 5, n_rows)
        open_prices = close_prices + rng.uniform(-2, 2, n_rows)

        df_trending = pd.DataFrame({
            "open": open_prices,
            "close": close_prices,
            "high": high_prices,
            "low": low_prices,
        })

        result = Ichimoku(
            df_trending,
            self.conversion_periods,
            self.base_periods,
            self.lagging_span_2_periods,
            self.displacement
        )

        conversion_line = result["conversion_line"].dropna()
        self.assertTrue(len(conversion_line) > 0)

    def test_ichimoku_values_within_reasonable_range(self):
        result = Ichimoku(
            self.df_lowercase,
            self.conversion_periods,
            self.base_periods,
            self.lagging_span_2_periods,
            self.displacement
        )

        high_max = self.df_lowercase["high"].max()
        low_min = self.df_lowercase["low"].min()

        conversion_line = result["conversion_line"].dropna()
        base_line = result["base_line"].dropna()

        self.assertTrue((conversion_line >= low_min).all())
        self.assertTrue((conversion_line <= high_max).all())
        self.assertTrue((base_line >= low_min).all())
        self.assertTrue((base_line <= high_max).all())

    def test_ichimoku_with_zero_displacement(self):
        result = Ichimoku(
            self.df_lowercase,
            conversion_periods=9,
            base_periods=26,
            lagging_span_2_periods=52,
            displacement=0
        )

        self.assertIsInstance(result, pd.DataFrame)

    def test_ichimoku_with_displacement_one(self):
        result = Ichimoku(
            self.df_uppercase,
            conversion_periods=9,
            base_periods=26,
            lagging_span_2_periods=52,
            displacement=1
        )

        lead_line1 = result["lead_line1"]
        leading_span_a = result["leading_span_a"]

        pd.testing.assert_series_equal(
            leading_span_a.dropna(),
            lead_line1.dropna(),
            check_names=False
        )

    def test_ichimoku_column_names_are_strings(self):
        result = Ichimoku(
            self.df_lowercase,
            self.conversion_periods,
            self.base_periods,
            self.lagging_span_2_periods,
            self.displacement
        )

        for col in result.columns:
            self.assertIsInstance(col, str)

    def test_ichimoku_no_infinite_values(self):
        result = Ichimoku(
            self.df_lowercase,
            self.conversion_periods,
            self.base_periods,
            self.lagging_span_2_periods,
            self.displacement
        )

        for col in result.columns:
            self.assertFalse(np.isinf(result[col]).any())

    def test_ichimoku_values(self):
        rng = np.random.default_rng(seed=369258)
        n_rows = 20

        close_prices = rng.integers(10, 40, n_rows)
        high_prices = close_prices + rng.integers(10, 40, n_rows)
        low_prices = close_prices - rng.integers(10, 40, n_rows)
        open_prices = close_prices + rng.integers(10, 40, n_rows)

        df_volatile = pd.DataFrame(
            {
                "open": open_prices,
                "close": close_prices,
                "high": high_prices,
                "low": low_prices,
            }
        )

        result = Ichimoku(
            df_volatile,
            8,
            12,
            4,
            8,
        )

        ref_values = pd.DataFrame(
            [
                [np.nan, np.nan, 12.0, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, 12.0, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, 13.0, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, 36.0, np.nan, 15.0, np.nan, np.nan],
                [np.nan, np.nan, 19.0, np.nan, 16.5, np.nan, np.nan],
                [np.nan, np.nan, 22.0, np.nan, 28.0, np.nan, np.nan],
                [np.nan, np.nan, 33.0, np.nan, 30.5, np.nan, np.nan],
                [16.5, np.nan, 22.0, np.nan, 27.0, np.nan, np.nan],
                [16.5, np.nan, 19.0, np.nan, 18.0, np.nan, np.nan],
                [20.0, np.nan, 15.0, np.nan, 15.5, np.nan, np.nan],
                [25.0, np.nan, 29.0, np.nan, 25.0, np.nan, 15.0],
                [25.0, 21.5, 38.0, 23.25, 25.0, np.nan, 16.5],
                [25.0, 21.5, 28.0, 23.25, 25.0, np.nan, 28.0],
                [25.0, 25.0, np.nan, 25.0, 26.0, np.nan, 30.5],
                [25.0, 25.0, np.nan, 25.0, 19.5, np.nan, 27.0],
                [25.0, 25.0, np.nan, 25.0, 19.5, np.nan, 18.0],
                [21.0, 21.0, np.nan, 21.0, 14.0, np.nan, 15.5],
                [21.0, 21.0, np.nan, 21.0, 21.0, np.nan, 25.0],
                [21.0, 21.0, np.nan, 21.0, 21.0, 23.25, 25.0],
                [21.0, 21.0, np.nan, 21.0, 21.0, 23.25, 25.0],
            ],
            columns=[
                "conversion_line",
                "base_line",
                "lagging_span",
                "lead_line1",
                "lead_line2",
                "leading_span_a",
                "leading_span_b",
            ],
            index=result.index,
        )

        pd.testing.assert_frame_equal(result, ref_values)


if __name__ == "__main__":
    unittest.main()
