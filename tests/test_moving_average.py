import unittest

import pandas as pd
import numpy as np
from src.tradingview_indicators.moving_average import sma, ema, sema, rma
from src.tradingview_indicators.errors_exceptions import InvalidArgumentError


class TestMovingAverage(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.source = pd.Series(np.random.randint(1, 500, 10))
        self.length = 5

    def test_sma(self):
        expected_result = pd.Series([253.2, 247.0, 197.6, 132.0, 98.4, 101.4])

        result = sma(self.source, self.length).reset_index(drop=True)

        pd.testing.assert_series_equal(result, expected_result)

    def test_ema(self):
        expected_result = pd.Series(
            [
                253.2,
                192.8,
                191.53333333333336,
                134.68888888888893,
                124.12592592592596,
                123.4172839506173,
            ]
        )

        result = ema(self.source, self.length).reset_index(drop=True)

        pd.testing.assert_series_equal(result, expected_result)

    def test_sema(self):
        smooth = 2

        expected_result = (
            pd.Series([179.26962962962963, 160.65218106995889])
            .rename("sema")
        )

        result = sema(self.source, self.length, smooth).reset_index(drop=True)

        pd.testing.assert_series_equal(result, expected_result)

    def test_rma_numpy(self):
        expected_result = pd.Series(
            [
                253.2,
                216.96,
                211.36800000000002,
                173.29440000000002,
                159.23552,
                151.788416,
            ]
        ).rename("RMA")

        result = rma(self.source, self.length, method="numpy").reset_index(drop=True)

        pd.testing.assert_series_equal(result, expected_result)

    def test_rma_pandas(self):
        expected_result = pd.Series(
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                253.2,
                152.53333333333333,
                167.47868852459015,
                117.85853658536587,
                113.43845787720134,
                115.75911180501345,
            ]
        ).rename("RMA")

        result = rma(self.source, self.length, method="pandas")

        pd.testing.assert_series_equal(result, expected_result)

    def test_rma_invalid_method(self):
        with self.assertRaises(InvalidArgumentError) as context:
            rma(self.source, self.length, method="invalid")
        self.assertEqual(
            str(context.exception),
            "method must be 'numpy' or 'pandas', got 'invalid'.",
        )
