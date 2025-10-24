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
        result = self.dtw.dtw_df
        ref_values = pd.DataFrame(
            [
                [0.0, 0.0, 89.02917103124065, 81.86562572425268],
                [1.0, 1.0, 26.697516547383408, 51.76787324092605],
                [1.0, 2.0, 26.697516547383408, 18.234535435057467],
                [2.0, 3.0, 92.88104093834257, 80.03922427076971],
                [3.0, 4.0, 95.19092773907751, 77.10829224999844],
                [4.0, 5.0, 88.70572813461894, 72.37030574037911],
                [5.0, 6.0, 20.41684866220504, 20.233374416809955],
                [6.0, 6.0, 27.435849612899734, 20.233374416809955],
                [7.0, 6.0, 40.756347776588626, 20.233374416809955],
                [8.0, 7.0, 54.621181023514474, 65.30321113244254],
                [9.0, 8.0, 90.85341095245559, 93.89437194840755],
                [9.0, 9.0, 90.85341095245559, 76.70511625350485],
                [10.0, 10.0, 10.581275727957076, 32.126423797607735],
                [11.0, 11.0, 30.4670320304519, 21.86988537114294],
                [12.0, 12.0, 81.42894571654963, 92.89051190151869],
                [12.0, 13.0, 81.42894571654963, 74.23930474300084],
                [12.0, 14.0, 81.42894571654963, 93.30138743862244],
                [12.0, 15.0, 81.42894571654963, 77.8937168154844],
                [13.0, 16.0, 47.28580460255, 25.684540472516353],
                [14.0, 17.0, 17.543296280808633, 24.55121680553762],
                [15.0, 18.0, 45.28860720316547, 32.7526647652443],
                [16.0, 18.0, 40.56141040572983, 32.7526647652443],
                [17.0, 18.0, 12.007222006386003, 32.7526647652443],
                [18.0, 19.0, 90.13944843440439, 65.07208418881117],
                [19.0, 19.0, 12.14827144082536, 65.07208418881117],
            ],
            columns=["signal_x_path", "signal_y_path", "signal_x", "signal_y"],
            index=range(25),
        )

        ref_values["signal_x_path"] = ref_values["signal_x_path"].astype(
            "int64"
        )
        ref_values["signal_y_path"] = ref_values["signal_y_path"].astype(
            "int64"
        )
        ref_values["signal_x"] = ref_values["signal_x"].astype("float64")
        ref_values["signal_y"] = ref_values["signal_y"].astype("float64")

        pd.testing.assert_frame_equal(result, ref_values)

    def test_calcualte_dtw_alligned_distance(self):
        results = pd.concat(self.dtw.align_dtw_distance(), axis=1)
        ref_values = pd.DataFrame(
            [
                [89.02917103124065, 81.86562572425268],
                [26.697516547383408, 51.76787324092605],
                [26.697516547383408, 18.234535435057467],
                [92.88104093834257, 80.03922427076971],
                [95.19092773907751, 77.10829224999844],
                [88.70572813461894, 72.37030574037911],
                [20.41684866220504, 20.233374416809955],
                [27.435849612899734, 20.233374416809955],
                [40.756347776588626, 20.233374416809955],
                [54.621181023514474, 65.30321113244254],
                [90.85341095245559, 93.89437194840755],
                [90.85341095245559, 76.70511625350485],
                [10.581275727957076, 32.126423797607735],
                [30.4670320304519, 21.86988537114294],
                [81.42894571654963, 92.89051190151869],
                [81.42894571654963, 74.23930474300084],
                [81.42894571654963, 93.30138743862244],
                [81.42894571654963, 77.8937168154844],
                [47.28580460255, 25.684540472516353],
                [17.543296280808633, 24.55121680553762],
            ],
            columns=["x", "y"],
            index=range(20),
        )
        pd.testing.assert_frame_equal(results, ref_values)

    def test_calculate_dtw_distance_ratio(self):
        result = self.dtw.calculate_dtw_distance(method="ratio")
        ref_values = pd.Series(
            [
                1.087503701872589,
                0.5157159233321023,
                1.4641182739459944,
                1.1604440420877828,
                1.2345096092966503,
                1.2257199583050187,
                1.0090679014590198,
                1.3559700447250134,
                2.01431293352275,
                0.8364241218205386,
                0.9676129577007779,
                1.1844504694079552,
                0.3293636352000374,
                1.3931043310658042,
                0.8766120893259749,
                1.0968441312649364,
                0.8727517130451785,
                1.0453852896689926,
                1.8410220207423216,
                0.7145591365089357,
                1.3827457255088376,
                1.2384155822573506,
                0.3666029036857955,
                1.3852245484078627,
                0.1866894474376494,
            ],
            index=range(25),
        )
        pd.testing.assert_series_equal(result, ref_values)

    def test_calculate_dtw_distance_absolute(self):
        result = self.dtw.calculate_dtw_distance(method="absolute")
        ref_values = pd.Series(
            [
                7.16354530698797,
                -25.070356693542642,
                8.462981112325942,
                12.841816667572857,
                18.082635489079067,
                16.335422394239828,
                0.18347424539508594,
                7.202475196089779,
                20.52297335977867,
                -10.682030108928068,
                -3.040960995951963,
                14.148294698950735,
                -21.545148069650658,
                8.597146659308962,
                -11.461566184969058,
                7.189640973548791,
                -11.872441722072807,
                3.5352289010652385,
                21.60126413003365,
                -7.007920524728988,
                12.535942437921172,
                7.808745640485533,
                -20.745442758858296,
                25.06736424559321,
                -52.923812747985814,
            ],
            index=range(25),
        )
        pd.testing.assert_series_equal(result, ref_values)

    def test_calculate_dtw_distance_absolute_aligned(self):
        result = self.dtw.calculate_dtw_distance(
            method="absolute", align_sequences=True
        )
        ref_values = pd.Series(
            [
                7.16354530698797,
                -25.070356693542642,
                8.462981112325942,
                12.841816667572857,
                18.082635489079067,
                16.335422394239828,
                0.18347424539508594,
                7.202475196089779,
                20.52297335977867,
                -10.682030108928068,
                -3.040960995951963,
                14.148294698950735,
                -21.545148069650658,
                8.597146659308962,
                -11.461566184969058,
                7.189640973548791,
                -11.872441722072807,
                3.5352289010652385,
                21.60126413003365,
                -7.007920524728988,
            ],
            index=range(20),
        )
        pd.testing.assert_series_equal(result, ref_values)

    def test_calculate_dtw_distance_ratio_aligned(self):
        result = self.dtw.calculate_dtw_distance(
            method="ratio", align_sequences=True
        )
        ref_values = pd.Series(
            [
                1.087503701872589,
                0.5157159233321023,
                1.4641182739459944,
                1.1604440420877828,
                1.2345096092966503,
                1.2257199583050187,
                1.0090679014590198,
                1.3559700447250134,
                2.01431293352275,
                0.8364241218205386,
                0.9676129577007779,
                1.1844504694079552,
                0.3293636352000374,
                1.3931043310658042,
                0.8766120893259749,
                1.0968441312649364,
                0.8727517130451785,
                1.0453852896689926,
                1.8410220207423216,
                0.7145591365089357,
            ],
            index=range(20),
        )
        pd.testing.assert_series_equal(result, ref_values)

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


class TestOHLCFinder(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(seed=42069)

        base_price = 100
        self.n_rows = 50

        close_prices = base_price + rng.normal(0, 5, self.n_rows).cumsum()
        high_prices = close_prices + rng.uniform(0.5, 3, self.n_rows)
        low_prices = close_prices - rng.uniform(0.5, 3, self.n_rows)
        open_prices = close_prices + rng.uniform(-1, 1, self.n_rows)

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

    def test_ohlc_finder_lowercase_columns(self):
        open_p, high_p, low_p, close_p = OHLC_finder(self.df_lowercase)

        pd.testing.assert_series_equal(open_p, self.df_lowercase["open"])
        pd.testing.assert_series_equal(high_p, self.df_lowercase["high"])
        pd.testing.assert_series_equal(low_p, self.df_lowercase["low"])
        pd.testing.assert_series_equal(close_p, self.df_lowercase["close"])

    def test_ohlc_finder_uppercase_columns(self):
        open_p, high_p, low_p, close_p = OHLC_finder(self.df_uppercase)

        pd.testing.assert_series_equal(open_p, self.df_uppercase["Open"])
        pd.testing.assert_series_equal(high_p, self.df_uppercase["High"])
        pd.testing.assert_series_equal(low_p, self.df_uppercase["Low"])
        pd.testing.assert_series_equal(close_p, self.df_uppercase["Close"])

    def test_ohlc_finder_custom_columns(self):
        open_p, high_p, low_p, close_p = OHLC_finder(
            self.df_custom,
            Open="price_open",
            High="price_high",
            Low="price_low",
            Close="price_close"
        )

        pd.testing.assert_series_equal(open_p, self.df_custom["price_open"])
        pd.testing.assert_series_equal(high_p, self.df_custom["price_high"])
        pd.testing.assert_series_equal(low_p, self.df_custom["price_low"])
        pd.testing.assert_series_equal(close_p, self.df_custom["price_close"])

    def test_ohlc_finder_partial_custom_columns(self):
        df_mixed = self.df_lowercase.copy()
        df_mixed["volume"] = np.random.default_rng(seed=31415).uniform(1000, 10000, self.n_rows)

        open_p, high_p, low_p, close_p = OHLC_finder(
            df_mixed,
            Close="close"
        )

        pd.testing.assert_series_equal(open_p, df_mixed["open"])
        pd.testing.assert_series_equal(high_p, df_mixed["high"])
        pd.testing.assert_series_equal(low_p, df_mixed["low"])
        pd.testing.assert_series_equal(close_p, df_mixed["close"])

    def test_ohlc_finder_missing_columns_raises_error(self):
        df_no_ohlc = pd.DataFrame({
            "price": np.random.default_rng(seed=27182).uniform(90, 110, 20),
            "volume": np.random.default_rng(seed=27183).uniform(1000, 10000, 20)
        })

        with self.assertRaises(ValueError) as context:
            OHLC_finder(df_no_ohlc)

        self.assertIn("OHLC columns not found", str(context.exception))

    def test_ohlc_finder_not_dataframe_raises_error(self):
        not_a_df = [1, 2, 3, 4]

        with self.assertRaises(ValueError) as context:
            OHLC_finder(not_a_df)

        self.assertIn("dataframe param must be a DataFrame", str(context.exception))

    def test_ohlc_finder_with_none_values(self):
        open_p, high_p, low_p, close_p = OHLC_finder(
            self.df_uppercase,
            Open=None,
            High=None,
            Low=None,
            Close=None
        )

        pd.testing.assert_series_equal(open_p, self.df_uppercase["Open"])
        pd.testing.assert_series_equal(high_p, self.df_uppercase["High"])
        pd.testing.assert_series_equal(low_p, self.df_uppercase["Low"])
        pd.testing.assert_series_equal(close_p, self.df_uppercase["Close"])

    def test_ohlc_finder_uppercase_priority(self):
        rng = np.random.default_rng(seed=12345)
        df_both = pd.DataFrame({
            "Open": rng.uniform(100, 110, 10),
            "High": rng.uniform(110, 120, 10),
            "Low": rng.uniform(90, 100, 10),
            "Close": rng.uniform(100, 110, 10),
            "open": rng.uniform(50, 60, 10),
            "high": rng.uniform(60, 70, 10),
            "low": rng.uniform(40, 50, 10),
            "close": rng.uniform(50, 60, 10),
        })

        open_p, high_p, low_p, close_p = OHLC_finder(df_both)

        pd.testing.assert_series_equal(open_p, df_both["Open"])
        pd.testing.assert_series_equal(high_p, df_both["High"])
        pd.testing.assert_series_equal(low_p, df_both["Low"])
        pd.testing.assert_series_equal(close_p, df_both["Close"])

    def test_ohlc_finder_with_additional_columns(self):
        rng = np.random.default_rng(seed=98765)
        df_extra = self.df_lowercase.copy()
        df_extra["volume"] = rng.uniform(1000, 10000, self.n_rows)
        df_extra["timestamp"] = pd.date_range("2023-01-01", periods=self.n_rows)
        df_extra["symbol"] = "BTC/USD"

        open_p, high_p, low_p, close_p = OHLC_finder(df_extra)

        pd.testing.assert_series_equal(open_p, df_extra["open"])
        pd.testing.assert_series_equal(high_p, df_extra["high"])
        pd.testing.assert_series_equal(low_p, df_extra["low"])
        pd.testing.assert_series_equal(close_p, df_extra["close"])

    def test_ohlc_finder_single_row(self):
        rng = np.random.default_rng(seed=11111)
        df_single = pd.DataFrame({
            "open": [rng.uniform(100, 110)],
            "high": [rng.uniform(110, 120)],
            "low": [rng.uniform(90, 100)],
            "close": [rng.uniform(100, 110)],
        })

        open_p, high_p, low_p, close_p = OHLC_finder(df_single)

        self.assertEqual(len(open_p), 1)
        self.assertEqual(len(high_p), 1)
        self.assertEqual(len(low_p), 1)
        self.assertEqual(len(close_p), 1)

    def test_ohlc_finder_preserves_index(self):
        custom_index = pd.date_range("2023-01-01", periods=self.n_rows, freq="D")
        df_indexed = self.df_lowercase.copy()
        df_indexed.index = custom_index

        open_p, high_p, low_p, close_p = OHLC_finder(df_indexed)

        pd.testing.assert_index_equal(open_p.index, custom_index)
        pd.testing.assert_index_equal(high_p.index, custom_index)
        pd.testing.assert_index_equal(low_p.index, custom_index)
        pd.testing.assert_index_equal(close_p.index, custom_index)
