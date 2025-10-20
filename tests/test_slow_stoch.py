import unittest
import pandas as pd
import numpy as np
from src.tradingview_indicators.slow_stoch import slow_stoch
from src.tradingview_indicators.errors_exceptions import InvalidArgumentError

class TestSlowStoch(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        source = (
            pd.read_csv("example/BTCUSDT_1d_spot.csv", index_col=0)
            .iloc[:-1]
        )
        self.short_source = source.iloc[:25]
        self.medium_source = source.iloc[:83]
        self.long_source = source.iloc[:119]

        self.k_length = 14
        self.k_smoothing = 2
        self.d_smoothing = 3

    def test_slow_stoch_sma(self):
        ref_k_indexes = pd.Index(
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

        ref_d_indexes = pd.Index(
            [
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

        ref_k_values = pd.Series(
            [
                95.53486648926238,
                97.53262198266285,
                83.12363967543837,
                70.856099636822,
                58.77084700394364,
                51.311722378487474,
                66.61627463160184,
                78.7829575135273,
                66.17359806614331,
                49.97829649974933,
                44.274392114893864,
            ],
            index=ref_k_indexes,
            name="%K",
        )

        ref_d_values = pd.Series(
            [
                92.06370938245453,
                83.8374537649744,
                70.91686210540134,
                60.31288967308436,
                58.89961467134432,
                65.57031817453887,
                70.52427673709082,
                64.9782840264733,
                53.475428893595506,
            ],
            index=ref_d_indexes,
            name="%D",
        )

        test_k, test_d = slow_stoch(
            self.short_source["close"],
            self.short_source["high"],
            self.short_source["low"],
            self.k_length,
            self.k_smoothing,
            self.d_smoothing,
            smoothing_method="sma",
        )

        pd.testing.assert_series_equal(test_k, ref_k_values)
        pd.testing.assert_series_equal(test_d, ref_d_values)

    def test_slow_stoch_ema(self):
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
            [95.53486648926238, np.nan],
            [96.2390631781997, np.nan],
            [78.51709961153882, 90.09700975966696],
            [74.20975416747001, 82.15338196356848],
            [55.06032643079112, 68.6068541971798],
            [56.44533027327922, 62.52609223522951],
            [69.54492147021335, 66.03550685272143],
            [77.49577246232057, 71.765639657521],
            [59.74925627004848, 65.75744796378474],
            [52.63681530707365, 59.197131635429194],
            [43.85773137182555, 51.52743150362737],
        ]

        ref_df = pd.DataFrame(ref_values, index=indexes, columns=["%K", "%D"])

        test_k, test_d = slow_stoch(
            self.short_source["close"],
            self.short_source["high"],
            self.short_source["low"],
            self.k_length,
            self.k_smoothing,
            self.d_smoothing,
            smoothing_method="ema",
        )

        test_df = pd.concat([test_k, test_d], axis=1)

        pd.testing.assert_frame_equal(test_df, ref_df)

    def test_slow_stoch_dema(self):
        indexes = pd.Index(
            [
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
            [95.88696483373104, np.nan],
            [84.30705468560289, np.nan],
            [77.57552100684764, np.nan],
            [62.56539128947662, np.nan],
            [58.485350612011686, 75.51076136001751],
            [65.85839785081279, 70.81118468576781],
            [73.61664759181798, 70.33765624371793],
            [64.37172004397164, 68.72779008326889],
            [56.54845021937298, 65.28048857718265],
            [48.087970987674694, 60.120533803284104],
        ]

        ref_df = pd.DataFrame(ref_values, index=indexes, columns=["%K", "%D"])

        test_k, test_d = slow_stoch(
            self.short_source["close"],
            self.short_source["high"],
            self.short_source["low"],
            self.k_length,
            self.k_smoothing,
            self.d_smoothing,
            smoothing_method="dema",
        )

        test_df = pd.concat([test_k, test_d], axis=1)

        pd.testing.assert_frame_equal(test_df, ref_df)

    def test_slow_stoch_tema(self):
        test_k, test_d = slow_stoch(
            self.short_source["close"],
            self.short_source["high"],
            self.short_source["low"],
            self.k_length,
            self.k_smoothing,
            self.d_smoothing,
            smoothing_method="tema",
        )

        test_df = pd.concat([test_k, test_d], axis=1)

        indexes = pd.Index(
            [
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
            [72.72714453747477, np.nan],
            [71.65205007298786, np.nan],
            [46.44484981396816, np.nan],
            [55.85685418848528, np.nan],
            [75.62414116030445, np.nan],
            [82.23666310675402, np.nan],
            [52.58018487262703, 58.190493820101594],
            [48.11325414081615, 47.974328438514036],
            [39.31731477014041, 38.73704507119598],
        ]

        ref_df = pd.DataFrame(ref_values, index=indexes, columns=["%K", "%D"])

        pd.testing.assert_frame_equal(test_df, ref_df)

    def test_slow_stoch_rma(self):
        ref_k_indexes = pd.Index(
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

        ref_d_indexes = pd.Index(
            [
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

        ref_k_values = pd.Series(
            [
                95.53486648926238,
                96.06301400596537,
                82.85956591708688,
                77.45782368126125,
                61.471718121856455,
                59.30477515818986,
                67.69974611343514,
                74.58547203590467,
                62.73073510490855,
                55.90566496524739,
                47.68692718472444,
            ],
            index=ref_k_indexes,
            name="%K",
        )

        ref_d_values = pd.Series(
            [
                91.48581547077156,
                86.80981820760145,
                78.36378484568645,
                72.01078161652093,
                70.573769782159,
                71.9110038667409,
                68.85091427946345,
                64.53583117472476,
                58.91952984472466,
            ],
            index=ref_d_indexes,
            name="%D",
        )

        test_k, test_d = slow_stoch(
            self.short_source["close"],
            self.short_source["high"],
            self.short_source["low"],
            self.k_length,
            self.k_smoothing,
            self.d_smoothing,
            smoothing_method="rma",
        )

        pd.testing.assert_series_equal(test_k, ref_k_values)
        pd.testing.assert_series_equal(test_d, ref_d_values)

    def test_slow_stoch_invalid_method(self):
        with self.assertRaises(InvalidArgumentError) as context:
            slow_stoch(
                self.short_source["close"],
                self.short_source["high"],
                self.short_source["low"],
                self.k_length,
                self.k_smoothing,
                self.d_smoothing,
                smoothing_method="invalid",
            )
        self.assertEqual(
            str(context.exception),
            "smoothing_method must be 'sma', 'ema', 'dema', 'tema', or 'rma', got 'invalid'.",
        )