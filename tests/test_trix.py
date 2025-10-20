import unittest
import pandas as pd
import numpy as np
from src.tradingview_indicators.TRIX import TRIX
from src.tradingview_indicators.errors_exceptions import InvalidArgumentError


class TestTrix(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.source = pd.Series(np.random.randint(1, 500, 60))
        self.dema_source = pd.Series(np.random.randint(1, 500, 112))
        self.tema_source = pd.Series(np.random.randint(1, 500, 163))
        self.length = 18
        self.signal_length = 1

    def test_trix_sma(self):
        expected_result = pd.Series(
            [
                np.nan,
                12.192185870230077,
                -1.6630617671520298,
                -15.837688425390795,
                -37.05532983268256,
                -51.953489146878695,
                -68.84957653841006,
                -77.75747400020627,
                -85.15236395226111,
            ],
            index=range(51, 60),
            name="TRIX",
        )

        result = TRIX(self.source, self.length, self.signal_length, "sma")

        pd.testing.assert_series_equal(result, expected_result)

    def test_trix_ema(self):
        expected_result = pd.Series(
            [
                np.nan,
                8.86050763307722,
                -1.9272532544878374,
                -4.833850140055773,
                -12.416765107685634,
                -7.833433279014201,
                -0.0018611586405370417,
                0.3318962104348344,
                -26.418522373559128,
            ],
            index=range(51, 60),
            name="TRIX",
        )

        result = TRIX(self.source, self.length, self.signal_length, "ema")

        pd.testing.assert_series_equal(result, expected_result)

    def test_trix_dema(self):
        expected_result = pd.Series(
            [
                np.nan,
                -37.34538675591814,
                -37.059518788655765,
                -36.94188499935969,
                -36.96795603815417,
                -37.090665032852854,
                -37.27604420752151,
                -37.48562142520839,
                -37.67782830019328,
                -37.82084645590267,
            ],
            index=range(102, 112),
            name="TRIX",
        )

        result = TRIX(self.dema_source, self.length, self.signal_length, "dema")

        pd.testing.assert_series_equal(result, expected_result)

    def test_trix_tema(self):
        expected_result = pd.Series(
            [
                np.nan,
                542.0515481781241,
                655.0753354089434,
                768.3775937297987,
                601.8218990305346,
                301.260699599446,
                157.91091562776137,
                -21.10271369721417,
                -99.46466831257439,
                -10.863910790561704,
            ],
            index=range(153, 163),
            name="TRIX",
        )

        result = TRIX(self.tema_source, self.length, self.signal_length, "tema")

        pd.testing.assert_series_equal(result, expected_result)

    def test_trix_rma(self):
        expected_result = pd.Series(
            [
                np.nan,
                9.90286601443735,
                8.71416291960081,
                8.408137074118471,
                7.315564899252891,
                7.813306810335874,
                8.860861098760608,
                8.951996463011369,
                5.044486316796437,
            ],
            index=range(51, 60),
            name="TRIX",
        )

        result = TRIX(self.source, self.length, self.signal_length, "rma")

        pd.testing.assert_series_equal(result, expected_result)

    def test_trix_invalid_method(self):
        with self.assertRaises(InvalidArgumentError) as context:
            TRIX(self.source, self.length, self.signal_length, "invalid")

        self.assertEqual(
            str(context.exception),
            "ma_method must be 'sma', 'ema', 'dema', 'tema', or 'rma', got 'invalid'.",
        )