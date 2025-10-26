import unittest
import pandas as pd
import numpy as np
from src.tradingview_indicators.DMI import DMI


class TestDMI(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(seed=13370)

        base_price = 50000
        self.n_rows = 100

        close_prices = base_price + rng.normal(0, 500, self.n_rows).cumsum()
        high_prices = close_prices + rng.uniform(100, 800, self.n_rows)
        low_prices = close_prices - rng.uniform(100, 800, self.n_rows)
        open_prices = close_prices + rng.uniform(-300, 300, self.n_rows)

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

    def test_dmi_initialization_lowercase(self):
        dmi = DMI(self.df_lowercase)

        self.assertIsInstance(dmi, DMI)
        pd.testing.assert_series_equal(dmi.close, self.df_lowercase["close"])
        pd.testing.assert_series_equal(dmi.high, self.df_lowercase["high"])
        pd.testing.assert_series_equal(dmi.low, self.df_lowercase["low"])

    def test_dmi_initialization_uppercase(self):
        dmi = DMI(self.df_uppercase)

        self.assertIsInstance(dmi, DMI)
        pd.testing.assert_series_equal(dmi.close, self.df_uppercase["Close"])
        pd.testing.assert_series_equal(dmi.high, self.df_uppercase["High"])
        pd.testing.assert_series_equal(dmi.low, self.df_uppercase["Low"])

    def test_dmi_initialization_custom_columns(self):
        dmi = DMI(
            self.df_custom,
            close="price_close",
            high="price_high",
            low="price_low"
        )

        self.assertIsInstance(dmi, DMI)
        pd.testing.assert_series_equal(dmi.close, self.df_custom["price_close"])
        pd.testing.assert_series_equal(dmi.high, self.df_custom["price_high"])
        pd.testing.assert_series_equal(dmi.low, self.df_custom["price_low"])

    def test_dmi_initialization_not_dataframe_raises_error(self):
        not_a_df = [1, 2, 3, 4]

        with self.assertRaises(ValueError) as context:
            DMI(not_a_df)

        self.assertIn("dataframe param must be a DataFrame", str(context.exception))

    def test_dmi_initialization_missing_columns_raises_error(self):
        df_no_ohlc = pd.DataFrame({
            "price": np.random.default_rng(seed=77777).uniform(90, 110, 20),
            "volume": np.random.default_rng(seed=77778).uniform(1000, 10000, 20)
        })

        with self.assertRaises(ValueError) as context:
            DMI(df_no_ohlc)

        self.assertIn("OHLC columns not found", str(context.exception))

    def test_dmi_initialization_with_not_none_columns_fallback(self):
        df_test = pd.DataFrame({
            "close": np.random.default_rng(seed=11111).uniform(90, 110, 20),
            "high": np.random.default_rng(seed=11112).uniform(95, 115, 20),
            "low": np.random.default_rng(seed=11113).uniform(85, 105, 20),
        })

        dmi = DMI(df_test)

        self.assertIsNotNone(dmi.close)
        self.assertIsNotNone(dmi.high)
        self.assertIsNotNone(dmi.low)
        pd.testing.assert_series_equal(dmi.close, df_test["close"])
        pd.testing.assert_series_equal(dmi.high, df_test["high"])
        pd.testing.assert_series_equal(dmi.low, df_test["low"])

    def test_true_range_calculation(self):
        dmi = DMI(self.df_lowercase)
        tr = dmi.true_range()

        self.assertIsInstance(tr, pd.Series)
        self.assertEqual(len(tr), len(self.df_lowercase))
        self.assertTrue((tr.dropna() >= 0).all())
        non_na_count = np.sum(~np.isnan(tr.values))
        self.assertTrue(non_na_count > len(tr) * 0.9)

    def test_true_range_minimum_value(self):
        dmi = DMI(self.df_lowercase)
        tr = dmi.true_range()

        high_low_range = dmi.high - dmi.low

        valid_indices = tr.iloc[1:].dropna().index
        for idx in valid_indices:
            self.assertGreaterEqual(tr.loc[idx], high_low_range.loc[idx])

    def test_adx_calculation(self):
        rng = np.random.default_rng(seed=13370)

        n_rows = 35

        close_prices = rng.integers(10, 40, n_rows)
        high_prices = close_prices + rng.integers(10, 40, n_rows)
        low_prices = close_prices - rng.integers(10, 40, n_rows)
        open_prices = close_prices + rng.integers(10, 40, n_rows)

        df_lowercase = pd.DataFrame(
            {
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
            }
        )

        dmi = DMI(df_lowercase)
        adx, plus_di, minus_di = dmi.adx()
        results = pd.concat([plus_di, minus_di, adx], axis=1)

        ref_values = pd.DataFrame(
            [
                [15.69767441860465, 10.029069767441861, np.nan],
                [15.129310344827585, 9.665948275862068, np.nan],
                [16.295521653543304, 8.966689222440944, np.nan],
                [15.325426670523575, 8.432889788834249, np.nan],
                [14.033548406511816, 9.829437502605735, np.nan],
                [13.000850013469176, 11.35879922769426, np.nan],
                [13.31509431506088, 10.462019081306504, np.nan],
                [12.437208622294852, 9.772241250853174, np.nan],
                [13.722090452938541, 9.124380325553886, np.nan],
                [12.788308989056922, 11.830335452737334, np.nan],
                [13.985305054746387, 10.635262432064467, np.nan],
                [13.039601287175893, 9.916092724165184, np.nan],
                [12.862906073078179, 9.437970825054178, np.nan],
                [12.023933464783074, 10.490912586627058, 15.988767530909048],
                [11.082547226113341, 11.175179577694452, 14.876439896395954],
                [12.197616172710154, 10.462664267327394, 14.360719669870223],
                [11.448929737633566, 14.11705009259875, 14.08039783314686],
                [13.78594168658026, 12.901776493829875, 13.311298279504419],
                [12.963639424374078, 12.132212815225156, 12.597134408265006],
                [14.409440750665965, 10.975273812843847, 12.663659348053047],
                [13.768722920411722, 10.487256703402142, 12.725432506427653],
            ],
            columns=["DI+", "DI-", "ADX"],
            index=range(14, 35),
        )
        pd.testing.assert_frame_equal(ref_values, results)

    def test_adx_calculation_with_custom_parameters(self):
        dmi = DMI(self.df_uppercase)
        adx, plus_di, minus_di = dmi.adx(adx_smoothing=20, di_length=20)

        self.assertIsInstance(adx, pd.Series)
        self.assertIsInstance(plus_di, pd.Series)
        self.assertIsInstance(minus_di, pd.Series)

        self.assertTrue((adx.dropna() >= 0).all())
        self.assertTrue((adx.dropna() <= 100).all())

    def test_adx_directional_indicators_positive(self):
        dmi = DMI(self.df_lowercase)
        _, plus_di, minus_di = dmi.adx()

        self.assertTrue((plus_di.dropna() >= 0).all())
        self.assertTrue((minus_di.dropna() >= 0).all())

    def test_adx_values_bounded(self):
        dmi = DMI(self.df_uppercase)
        adx, _, _ = dmi.adx()

        adx_clean = adx.dropna()
        self.assertTrue((adx_clean >= 0).all())
        self.assertTrue((adx_clean <= 100).all())

    def test_di_difference_calculation(self):
        dmi = DMI(self.df_lowercase)
        di_delta, di_ratio = dmi.di_difference()

        self.assertIsInstance(di_delta, pd.Series)
        self.assertIsInstance(di_ratio, pd.Series)

        self.assertEqual(di_delta.name, "DI_Delta")
        self.assertEqual(di_ratio.name, "DI_Ratio")

    def test_di_difference_with_custom_parameters(self):
        dmi = DMI(self.df_uppercase)
        di_delta, di_ratio = dmi.di_difference(adx_smoothing=21, di_length=21)

        self.assertIsInstance(di_delta, pd.Series)
        self.assertIsInstance(di_ratio, pd.Series)

    def test_di_delta_range(self):
        dmi = DMI(self.df_lowercase)
        di_delta, _ = dmi.di_difference()

        di_delta_clean = di_delta.dropna()
        self.assertTrue((di_delta_clean >= -100).all())
        self.assertTrue((di_delta_clean <= 100).all())

    def test_di_ratio_positive(self):
        dmi = DMI(self.df_lowercase)
        _, di_ratio = dmi.di_difference()

        di_ratio_clean = di_ratio.dropna()
        self.assertTrue((di_ratio_clean > 0).all())

    def test_dmi_with_minimal_data(self):
        rng = np.random.default_rng(seed=88888)
        n_rows = 30

        close_prices = 100 + rng.normal(0, 5, n_rows).cumsum()
        high_prices = close_prices + rng.uniform(0.5, 2, n_rows)
        low_prices = close_prices - rng.uniform(0.5, 2, n_rows)

        df_minimal = pd.DataFrame({
            "close": close_prices,
            "high": high_prices,
            "low": low_prices,
        })

        dmi = DMI(df_minimal)
        adx, plus_di, minus_di = dmi.adx(adx_smoothing=14, di_length=14)

        self.assertIsInstance(adx, pd.Series)
        self.assertIsInstance(plus_di, pd.Series)
        self.assertIsInstance(minus_di, pd.Series)

    def test_dmi_with_trending_data(self):
        rng = np.random.default_rng(seed=99999)
        n_rows = 100

        trend = np.linspace(0, 50, n_rows)
        close_prices = 100 + trend + rng.normal(0, 2, n_rows)
        high_prices = close_prices + rng.uniform(0.5, 1.5, n_rows)
        low_prices = close_prices - rng.uniform(0.5, 1.5, n_rows)

        df_trending = pd.DataFrame({
            "close": close_prices,
            "high": high_prices,
            "low": low_prices,
        })

        dmi = DMI(df_trending)
        adx, plus_di, minus_di = dmi.adx()

        last_20_plus = np.nanmean(plus_di.iloc[-20:].values)
        last_20_minus = np.nanmean(minus_di.iloc[-20:].values)

        self.assertGreater(last_20_plus, last_20_minus)

    def test_dmi_with_ranging_data(self):
        rng = np.random.default_rng(seed=111111)
        n_rows = 100

        close_prices = 100 + 10 * np.sin(np.linspace(0, 4*np.pi, n_rows))
        close_prices += rng.normal(0, 1, n_rows)
        high_prices = close_prices + rng.uniform(0.5, 1.5, n_rows)
        low_prices = close_prices - rng.uniform(0.5, 1.5, n_rows)

        df_ranging = pd.DataFrame({
            "close": close_prices,
            "high": high_prices,
            "low": low_prices,
        })

        dmi = DMI(df_ranging)
        adx, _, _ = dmi.adx()

        self.assertIsInstance(adx, pd.Series)

    def test_dmi_preserves_dataframe_index(self):
        custom_index = pd.date_range("2023-01-01", periods=self.n_rows, freq="D")
        df_indexed = self.df_lowercase.copy()
        df_indexed.index = custom_index

        dmi = DMI(df_indexed)
        adx, plus_di, minus_di = dmi.adx()

        self.assertTrue(adx.dropna().index.isin(custom_index).all())
        self.assertTrue(plus_di.dropna().index.isin(custom_index).all())
        self.assertTrue(minus_di.dropna().index.isin(custom_index).all())

    def test_dmi_consistency_between_methods(self):
        dmi = DMI(self.df_lowercase)

        adx1, plus1, minus1 = dmi.adx(adx_smoothing=14, di_length=14)
        di_delta, di_ratio = dmi.di_difference(adx_smoothing=14, di_length=14)

        expected_delta = plus1 - minus1
        pd.testing.assert_series_equal(
            di_delta.dropna(),
            expected_delta.dropna(),
            check_names=False
        )

        expected_ratio = plus1 / minus1
        pd.testing.assert_series_equal(
            di_ratio.dropna(),
            expected_ratio.dropna(),
            check_names=False
        )

    def test_dmi_short_period_calculations(self):
        dmi = DMI(self.df_lowercase)
        adx, plus_di, minus_di = dmi.adx(adx_smoothing=5, di_length=5)

        self.assertTrue(adx.dropna().notna().any())
        self.assertTrue(plus_di.dropna().notna().any())
        self.assertTrue(minus_di.dropna().notna().any())

    def test_dmi_long_period_calculations(self):
        dmi = DMI(self.df_uppercase)
        adx, plus_di, minus_di = dmi.adx(adx_smoothing=30, di_length=30)

        self.assertTrue(adx.dropna().notna().any())
        self.assertTrue(plus_di.dropna().notna().any())
        self.assertTrue(minus_di.dropna().notna().any())
