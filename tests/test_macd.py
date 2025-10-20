import unittest

import pandas as pd
import numpy as np
from src.tradingview_indicators import MACD
from src.tradingview_indicators.errors_exceptions import InvalidArgumentError


class TestMACD(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.source_10 = pd.Series(np.random.randint(1, 500, 10))
        self.source_20 = self.source = pd.Series(np.random.randint(1, 500, 20))
        self.fast_length = 3
        self.slow_length = 5
        self.signal_length = 2
        self.diff_method = "absolute"
        self.indexes = range(4, 10)
        self.columns = ["macd", "signal", "histogram"]

    def test_MACD_ma_sma(self):
        ref_values = [
            [-10.866666666666646, np.nan, np.nan],
            [-97.0, -53.93333333333332, -43.06666666666668],
            [-74.93333333333332, -67.93333333333332, -7.0],
            [-38.0, -47.977777777777774, 9.977777777777774],
            [5.933333333333323, -12.037037037037045, 17.970370370370368],
            [-19.400000000000006, -16.945679012345686, -2.45432098765432],
        ]

        ref_df = pd.DataFrame(
            ref_values,
            index=self.indexes,
            columns=self.columns,
        )

        macd_df = MACD(
            self.source_10,
            self.fast_length,
            self.slow_length,
            self.signal_length,
            ma_method="sma",
        )

        pd.testing.assert_frame_equal(ref_df, macd_df)

    def test_MACD_ma_ema(self):
        ref_values = [
            [-57.94999999999999, np.nan, np.nan],
            [-59.17500000000001, -58.5625, -0.6125000000000114],
            [-30.22083333333336, -39.668055555555576, 9.447222222222216],
            [-43.532638888888926, -42.244444444444476, -1.28819444444445],
            [-27.047800925925955, -32.11334876543213, 5.0655478395061735],
            [-13.878221450617303, -19.956597222222246, 6.078375771604943],
        ]

        ref_df = pd.DataFrame(
            ref_values,
            index=self.indexes,
            columns=self.columns,
        )

        macd_df = MACD(
            self.source_10,
            self.fast_length,
            self.slow_length,
            self.signal_length,
            ma_method="ema",
        )

        pd.testing.assert_frame_equal(ref_df, macd_df)

    def test_MACD_ma_rma(self):
        ref_values = [
            [-25.75555555555553, np.nan, np.nan],
            [-41.33037037037036, -33.542962962962946, -7.787407407407414],
            [-31.28158024691359, -32.035374485596705, 0.7537942386831133],
            [-46.23678683127572, -41.50298271604938, -4.7338041152263415],
            [-40.19711122085047, -40.63240171925011, 0.4352904983996382],
            [-31.762810147233637, -34.71934067123913, 2.956530524005494],
        ]

        ref_df = pd.DataFrame(
            ref_values,
            index=self.indexes,
            columns=self.columns,
        )

        macd_df = MACD(
            self.source_10,
            self.fast_length,
            self.slow_length,
            self.signal_length,
            ma_method="rma",
        )

        pd.testing.assert_frame_equal(ref_df, macd_df)

    def test_MACD_ma_dema(self):
        ref_values = [
            [-42.44961419753088, np.nan, np.nan],
            [-50.758965620713354, -46.604289909122116, -4.1546757115912385],
            [-50.19715649291271, -48.999534298315844, -1.1976221945968675],
            [-22.64577814214681, -31.43036352753649, 8.78458538538968],
            [-6.147375570765433, -14.575038223022453, 8.42766265225702],
            [14.368826030936646, 4.7208712796169445, 9.647954751319702],
            [47.34574003489149, 33.13745044979997, 14.20828958509152],
            [55.9048460905463, 48.31571421029753, 7.589131880248772],
            [36.13013000578047, 40.19199140728616, -4.061861401505688],
            [28.946738131358188, 32.69515589000085, -3.7484177586426597],
            [-2.111687694226589, 9.49059350051589, -11.60228119474248],
            [9.431405719319116, 9.451134979718042, -0.01972926039892542],
        ]

        ref_df = pd.DataFrame(
            ref_values,
            index=range(8, 20),
            columns=self.columns,
        )

        macd_df = MACD(
            self.source_20,
            self.fast_length,
            self.slow_length,
            self.signal_length,
            ma_method="dema",
        )

        macd_df.columns = self.columns
        ref_df.columns = self.columns

        pd.testing.assert_frame_equal(ref_df, macd_df)

    def test_MACD_ma_tema(self):
        np.random.seed(42)
        self.source = pd.Series(np.random.randint(1, 500, 20))
        self.fast_length = 3
        self.slow_length = 5
        self.signal_length = 2

        ref_values = [
            [14.406818702409794, np.nan, np.nan],
            [14.653630057130442, 14.530224379770118, 0.12340567736032426],
            [25.05524542319705, 21.546905075388075, 3.508340347808975],
            [-13.94962795816707, -2.1174502803153548, -11.832177677851716],
            [-43.379868203303545, -29.625728895640815, -13.75413930766273],
            [-11.782574841064957, -17.730292859256913, 5.9477180181919564],
            [-37.18720195785079, -30.70156559165283, -6.485636366197959],
            [28.805414513595736, 8.969754478512879, 19.835660035082856],
        ]

        ref_df = pd.DataFrame(
            ref_values,
            index=range(12, 20),
            columns=self.columns,
        )

        macd_df = MACD(
            self.source_20,
            self.fast_length,
            self.slow_length,
            self.signal_length,
            ma_method="tema",
        )
        macd_df.columns = self.columns

        pd.testing.assert_frame_equal(ref_df, macd_df)

    def test_MACD_invalid_ma_method(self):
        with self.assertRaises(InvalidArgumentError) as context:
            MACD(
                self.source_10,
                self.fast_length,
                self.slow_length,
                self.signal_length,
                ma_method="invalid_ma",
            )
        self.assertEqual(
            str(context.exception),
            "ma_method must be 'sma', 'ema', 'dema', 'tema', or 'rma', got 'invalid_ma'.",
        )
    def test_MACD_invalid_diff_method(self):
        with self.assertRaises(InvalidArgumentError) as context:
            MACD(
                self.source_10,
                self.fast_length,
                self.slow_length,
                self.signal_length,
                diff_method="invalid_diff",
            )
        self.assertEqual(
            str(context.exception),
            "diff_method must be 'absolute', 'ratio', or 'dtw', got 'invalid_diff'.",
        )
    def test_MACD_invalid_signal_method(self):
        with self.assertRaises(InvalidArgumentError) as context:
            MACD(
                self.source_10,
                self.fast_length,
                self.slow_length,
                self.signal_length,
                signal_method="invalid_signal",
            )
        self.assertEqual(
            str(context.exception),
            "signal_method must be 'sma', 'ema', 'dema', 'tema', or 'rma', got 'invalid_signal'.",
        )