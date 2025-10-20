import unittest
import pandas as pd
import numpy as np

from src.tradingview_indicators.bollinger import bollinger_trends
from src.tradingview_indicators.errors_exceptions import InvalidArgumentError

class TestBollingerBands(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.source = pd.Series(np.random.randint(1, 500, 50))
        self.short_length = 10
        self.stdev = 2

    def test_bollinger_trends_normal(self):
        ref_values = [
            -73.09644670050763,
            -75.62002807674314,
            -84.45907635754708,
            87.6712328767123,
            131.4508264920836,
            -64.63473633284956,
            5.320945945945931,
            -7.547169811320736,
            24.2933537051184,
            -1.7156094986853856,
        ]

        ref_df = pd.Series(ref_values, index=range(9, 19), name="Bollinger Trend")

        test_values = bollinger_trends(
            self.source,
            self.short_length,
            self.stdev,
            stdev_method="absolute",
            diff_method="normal",
        ).iloc[9:19]

        pd.testing.assert_series_equal(ref_df, test_values)

    def test_bollinger_trends_absolute(self):
        ref_values = [
            -17859.6,
            -21531.6,
            -19321.82359030106,
            -18813.6,
            -20585.793575937514,
            -20803.6,
            -23667.4,
            -22807.2,
            -26116.4,
            -26674.575530532995,
        ]

        ref_df = pd.Series(
            ref_values,
            index=range(9, 19),
            name="Bollinger Trend",
        )

        test_values = bollinger_trends(
            self.source,
            self.short_length,
            self.stdev,
            stdev_method="absolute",
            diff_method="absolute",
        ).iloc[9:19]

        pd.testing.assert_series_equal(ref_df, test_values)

    def test_bollinger_trends_ratio(self):
        ref_values = [
            0.18461994200487786,
            0.3663345672678181,
            4.249053707103651,
            0.37280623343162145,
            0.2683303864537662,
            0.33307423920239476,
            0.23486902138749785,
            0.33951568314334674,
            0.12994393707976107,
            0.28554727862120927,
        ]

        ref_df = pd.Series(
            ref_values,
            index=range(9, 19),
            name="Bollinger Trend",
        )

        test_values = bollinger_trends(
            self.source,
            self.short_length,
            self.stdev,
            stdev_method="ratio",
            diff_method="ratio",
        ).iloc[9:19]

        pd.testing.assert_series_equal(ref_df, test_values)

    def test_bollinger_trends_dtw(self):
        ref_values = [
            18.461994200487784,
            9.390672023352204,
            96.56185881046612,
            93.60233380902268,
            104.61804735521858,
            109.2132021342806,
            90.07465530800913,
            74.30953900806698,
            60.265145210540304,
            43.246995578495266,
        ]

        ref_df = pd.Series(
            ref_values,
            index=range(9, 19),
            name="Bollinger Trend",
        )

        test_values = bollinger_trends(
            self.source,
            self.short_length,
            self.stdev,
            stdev_method="dtw",
            diff_method="dtw",
        ).iloc[9:19]

        pd.testing.assert_series_equal(ref_df, test_values)

    def test_bollinger_bands_invalid_ma_method(self):
        with self.assertRaises(InvalidArgumentError) as context:
            bollinger_trends(
                self.source,
                self.short_length,
                self.stdev,
                ma_method="invalid_method",
            )

        self.assertEqual(
            str(context.exception),
            "ma_method must be 'sma', 'ema', 'dema', 'tema', or 'rma', got 'invalid_method'.",
        )