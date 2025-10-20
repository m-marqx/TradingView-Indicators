import unittest
import pandas as pd
import numpy as np
from src.tradingview_indicators import RSI
from src.tradingview_indicators.errors_exceptions import InvalidArgumentError

class TestRSI(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        source = pd.read_csv("example/BTCUSDT_1d_spot.csv", index_col=0).iloc[:-1]
        self.short_source = source.iloc[:25]
        self.medium_source = source.iloc[:40]
        self.long_source = source.iloc[:50]

        self.length = 14

    def test_rsi_sma(self):
        indexes = pd.Index(
            [
                "2017-08-31",
                "2017-09-01",
                "2017-09-02",
                "2017-09-03",
                "2017-09-04",
                "2017-09-05",
                "2017-09-06",
                "2017-09-07",
                "2017-09-08",
                "2017-09-09",
                "2017-09-10",
            ],
            name="open_time",
        )
        ref_values = [
            67.85827398305986,
            81.19000601013147,
            61.102643295495625,
            64.29204048380448,
            52.31352356432811,
            57.92345217581246,
            61.292108741471104,
            58.90351209891528,
            50.04269442072532,
            48.39529957265133,
            46.47902783222265,
        ]

        ref_df = pd.Series(ref_values, index=indexes, name="RSI")

        test_rsi = RSI(self.short_source["close"], self.length, "sma")

        pd.testing.assert_series_equal(test_rsi, ref_df)

    def test_rsi_ema(self):
        indexes = pd.Index(
            [
                "2017-08-31",
                "2017-09-01",
                "2017-09-02",
                "2017-09-03",
                "2017-09-04",
                "2017-09-05",
                "2017-09-06",
                "2017-09-07",
                "2017-09-08",
                "2017-09-09",
                "2017-09-10",
            ],
            name="open_time",
        )
        ref_values = [
            67.85827398305986,
            73.04536784462798,
            45.257584105527584,
            47.59970028227267,
            30.778063974083167,
            45.30402437158805,
            55.541061793939676,
            58.106962290503134,
            42.13850551761739,
            41.368730847300505,
            37.17367331545027,
        ]

        ref_df = pd.Series(ref_values, index=indexes, name="RSI")

        test_rsi = RSI(self.short_source["close"], self.length, "ema")

        pd.testing.assert_series_equal(test_rsi, ref_df)

    def test_rsi_dema(self):
        indexes = pd.Index(
            [
                "2017-09-13",
                "2017-09-14",
                "2017-09-15",
                "2017-09-16",
                "2017-09-17",
                "2017-09-18",
                "2017-09-19",
                "2017-09-20",
                "2017-09-21",
                "2017-09-22",
                "2017-09-23",
                "2017-09-24",
                "2017-09-25",
            ],
            name="open_time",
        )
        ref_values = [
            45.473351418710536,
            40.17643541871139,
            39.896811355690666,
            39.80032881737036,
            39.67532772860775,
            41.46710248805541,
            42.28409714307008,
            42.85444071788075,
            42.13904474409599,
            41.540849537417294,
            42.09679826319482,
            42.04121490078051,
            43.452858299751824,
        ]

        ref_df = pd.Series(ref_values, index=indexes, name="RSI")

        test_rsi = RSI(self.medium_source["close"], self.length, "dema")

        pd.testing.assert_series_equal(test_rsi, ref_df)

    def test_rsi_tema(self):
        indexes = pd.Index(
            [
                "2017-09-26",
                "2017-09-27",
                "2017-09-28",
                "2017-09-29",
                "2017-09-30",
                "2017-10-01",
                "2017-10-02",
                "2017-10-03",
                "2017-10-04",
                "2017-10-05",
            ],
            name="open_time",
        )
        ref_values = [
            68.13395822544877,
            89.49296234709793,
            90.70815731715388,
            99.6630779363396,
            106.67669095745168,
            118.35093396118103,
            142.8239608552019,
            78.25256770835128,
            25.629678467397426,
            56.56228144510478,
        ]

        ref_df = pd.Series(ref_values, index=indexes, name="RSI")

        test_rsi = RSI(self.long_source["close"], self.length, "tema")

        pd.testing.assert_series_equal(test_rsi, ref_df)

    def test_rsi_rma(self):
        indexes = pd.Index(
            [
                "2017-08-31",
                "2017-09-01",
                "2017-09-02",
                "2017-09-03",
                "2017-09-04",
                "2017-09-05",
                "2017-09-06",
                "2017-09-07",
                "2017-09-08",
                "2017-09-09",
                "2017-09-10",
            ],
            name="open_time",
        )
        ref_values = [
            67.85827398305986,
            70.67946559078209,
            53.884720079742976,
            55.05584599074014,
            42.26003453900124,
            50.3531915310706,
            56.585226125923334,
            58.18829511861674,
            47.450898785420655,
            46.90392568983139,
            43.98085118876622,
        ]

        ref_df = pd.Series(ref_values, index=indexes, name="RSI")

        test_rsi = RSI(self.short_source["close"], self.length, "rma")

        pd.testing.assert_series_equal(test_rsi, ref_df)

    def test_rsi_invalid_ma_method(self):
        with self.assertRaises(InvalidArgumentError) as context:
            RSI(self.short_source["close"], self.length, "invalid_method")

        self.assertEqual(
            str(context.exception),
            "ma_method must be 'sma', 'ema', 'dema', 'tema', or 'rma', got 'invalid_method'.",
        )