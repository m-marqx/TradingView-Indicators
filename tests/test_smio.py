import unittest
import pandas as pd
import numpy as np
from src.tradingview_indicators.SMIO import SMIO
from src.tradingview_indicators.errors_exceptions import InvalidArgumentError


class TestSMIO(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.source = pd.Series(np.random.randint(1, 500, 37))
        self.dema_source = pd.Series(np.random.randint(1, 500, 42))
        self.tema_source = pd.Series(np.random.randint(1, 500, 52))
        self.long_length = 20
        self.short_length = 5
        self.signal_length = 5

    def test_smio_sma(self):
        expected_result = pd.Series(
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                -0.03702186972261484,
                -0.005783453565452512,
                -0.016429866077644253,
                -0.04027520292196353,
                0.005410234847397045,
                -0.0022208095652088804,
                -0.04643814522873069,
                -0.025121318783773383,
                -0.018804672815918785,
            ],
            index=range(24, 37),
            name="SMIO",
        )

        result = SMIO(
            self.source,
            long_length=self.long_length,
            short_length=self.short_length,
            signal_length=self.signal_length,
            ma_method="sma",
        )

        pd.testing.assert_series_equal(result, expected_result)

    def test_smio_ema(self):
        expected_result = pd.Series(
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                -0.03702186972261484,
                -0.006788511334759897,
                -0.0185668188832111,
                -0.037422277778225675,
                0.000891031289350537,
                -0.004603580707847523,
                -0.04452781559367561,
                -0.023386574709263256,
                -0.01395680596492259,
            ],
            index=range(24, 37),
            name="SMIO",
        )

        result = SMIO(
            self.source,
            long_length=self.long_length,
            short_length=self.short_length,
            signal_length=self.signal_length,
            ma_method="ema",
        )

        pd.testing.assert_series_equal(result, expected_result)

    def test_smio_dema(self):
        expected_result = pd.Series(
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.0267577196807172,
                -0.005846336095866194,
                0.02265095765153298,
                0.029665062791784838,
                -0.0031684075739063684,
                0.017273292479976096,
                0.03715191104254778,
                0.04659186566896591,
                0.04402397777194027,
                0.012004380888481284,
            ],
            index=range(24, 42),
            name="SMIO",
        )

        result = SMIO(
            self.dema_source,
            long_length=self.long_length,
            short_length=self.short_length,
            signal_length=self.signal_length,
            ma_method="dema",
        )

        pd.testing.assert_series_equal(result, expected_result)

    def test_smio_tema(self):
        expected_result = pd.Series(
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.011979480399899049,
                0.010364829627636777,
                -0.022437975652883912,
                0.0013571057232135628,
                0.015116558624286645,
                -0.0057093778245332575,
                -0.005605140249119078,
                -0.008611786322395995,
                0.009662062705362694,
                0.016109361624092232,
                -0.026551542376946853,
                -0.012987557415279204,
                -0.01146411314267011,
                0.01723487483428201,
                0.0027117296369333563,
                0.017728337426081108,
            ],
            index=range(24, 52),
            name="SMIO",
        )

        result = SMIO(
            self.tema_source,
            long_length=self.long_length,
            short_length=self.short_length,
            signal_length=self.signal_length,
            ma_method="tema",
        )

        pd.testing.assert_series_equal(result, expected_result)

    def test_smio_rma(self):
        expected_result = pd.Series(
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                -0.03702186972261484,
                -0.008146213601711881,
                -0.0233663444734149,
                -0.04874635380603385,
                -0.007990023275025897,
                -0.012629140500918175,
                -0.059853826546867245,
                -0.040324698413669216,
                -0.03029866612143188,
            ],
            index=range(24, 37),
            name="SMIO",
        )

        result = SMIO(
            self.source,
            long_length=self.long_length,
            short_length=self.short_length,
            signal_length=self.signal_length,
            ma_method="rma",
        )

        pd.testing.assert_series_equal(result, expected_result)

    def test_smio_invalid_method(self):
        with self.assertRaises(InvalidArgumentError) as context:
            SMIO(
                self.source,
                long_length=self.long_length,
                short_length=self.short_length,
                signal_length=self.signal_length,
                ma_method="invalid",
            )
        self.assertEqual(
            str(context.exception),
            "ma_method must be 'sma', 'ema', 'dema', 'tema', or 'rma', got 'invalid'.",
        )
