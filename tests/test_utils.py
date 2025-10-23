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
