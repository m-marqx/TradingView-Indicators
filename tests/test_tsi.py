import unittest
import pandas as pd
import numpy as np
from src.tradingview_indicators.tsi import tsi
from src.tradingview_indicators.errors_exceptions import InvalidArgumentError


class TestTsi(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.source = pd.Series(np.random.randint(1, 500, 47))
        self.dema_source = pd.Series(np.random.randint(1, 500, 83))
        self.tema_source = pd.Series(np.random.randint(1, 500, 119))
        self.short_length = 13
        self.long_length = 25

    def test_tsi_sma(self):
        expected_result = pd.Series(
            [
                0.019328469383857287,
                0.005515019425755706,
                0.015344311377245507,
                0.008646595748603454,
                0.011284785625593454,
                0.007186170694180467,
                0.004095268886733483,
                0.010450859667488774,
                0.004747140329672252,
                -0.0010230090336899435,
            ],
            index=range(37, 47),
            name="TSI",
        )

        result = tsi(self.source, self.short_length, self.long_length, "sma")

        pd.testing.assert_series_equal(result, expected_result)

    def test_tsi_ema(self):
        expected_result = pd.Series(
            [
                0.002335746884832184,
                -0.013084348062166937,
                0.0032651357187076743,
                -0.010880455657194829,
                -0.014534589993399067,
                0.0014426670003598614,
                -0.003504231189196725,
                0.008675016842811072,
                0.00693257143011235,
                0.0010370377682753253,
            ],
            index=range(37, 47),
            name="TSI",
        )

        result = tsi(self.source, self.short_length, self.long_length, "ema")

        pd.testing.assert_series_equal(result, expected_result)

    def test_tsi_dema(self):
        expected_result = pd.Series(
            [
                -0.0039715831415075625,
                -0.003936666913911443,
                -0.0039799279148078435,
                -0.003994711627304258,
                -0.003831370984631432,
                -0.003721103181496938,
                -0.003731082310130517,
                -0.004003580330772648,
                -0.004404342646217298,
                -0.005012800877856322,
            ],
            index=range(73, 83),
            name="TSI",
        )

        result = tsi(self.dema_source, self.short_length, self.long_length, "dema")

        pd.testing.assert_series_equal(result, expected_result)

    def test_tsi_tema(self):
        expected_result = pd.Series(
            [
                -0.0019853412775270303,
                -0.04092061417144905,
                -0.05047342116419206,
                -0.002564465587187089,
                -0.0590057259527061,
                0.044402201046715474,
                0.0835918474059937,
                -0.03852246192266579,
                -0.021019087337841195,
                -0.08148354659495585,
            ],
            index=range(109, 119),
            name="TSI",
        )

        result = tsi(self.tema_source, self.short_length, self.long_length, "tema")

        pd.testing.assert_series_equal(result, expected_result)

    def test_tsi_rma(self):
        expected_result = pd.Series(
            [
                0.03834363657906433,
                0.03275677457437066,
                0.03567266461888342,
                0.030120401015458872,
                0.027299279473885676,
                0.030331684808031136,
                0.02755343723614153,
                0.02998261655449017,
                0.028544027239623307,
                0.025818486811049805,
            ],
            index=range(37, 47),
            name="TSI",
        )

        result = tsi(self.source, self.short_length, self.long_length, "rma")

        pd.testing.assert_series_equal(result, expected_result)

    def test_tsi_invalid_method(self):
        with self.assertRaises(InvalidArgumentError) as context:
            tsi(self.source, self.short_length, self.long_length, "invalid")

        self.assertEqual(
            str(context.exception),
            "ma_method must be 'sma', 'ema', 'dema', 'tema', or 'rma', got 'invalid'.",
        )