import unittest

import pandas as pd
import numpy as np
from src.tradingview_indicators.CCI import CCI
from src.tradingview_indicators.errors_exceptions import InvalidArgumentError

class TestCCI(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.source = pd.Series(np.random.randint(1, 500, 10))
        self.length = 5

    def test_CCI_sma(self):
        np_values = np.array(
            [
                [107, 118.56, 253.2, -82.20872695],
                [72, 126, 247, -92.59259259],
                [189, 89.92, 197.6, -6.37603796],
                [21, 78.4, 132, -94.3877551],
                [103, 41.52, 98.4, 7.38599872],
                [122, 43.92, 101.4, 31.26897389],
            ]
        )
        columns = ["source", "mad", "ma", "CCI"]
        indexes = [4, 5, 6, 7, 8, 9]

        ref_df = (
            pd.DataFrame(np_values, columns=columns, index=indexes)
            .astype("float64")
        )

        test_df = CCI(self.source, self.length, method="sma").astype("float64")

        pd.testing.assert_frame_equal(ref_df, test_df)

    def test_CCI_ema(self):
        np_values = np.array(
            [
                [107.0, 118.55999999999999, 253.2, -82.20872694556905],
                [72.0, 126.0, 192.8, -63.915343915343925],
                [189.0, 89.92, 191.53333333333336, -1.8782127323052786],
                [21.0, 78.4, 134.68888888888893, -96.67422524565386],
                [103.0, 41.519999999999996, 124.12592592592596, -33.92088298960494],
                [122.0, 43.92, 123.4172839506173, -2.15131140045128],
            ]
        )
        columns = ["source", "mad", "ma", "CCI"]
        indexes = [4, 5, 6, 7, 8, 9]

        ref_df = (
            pd.DataFrame(np_values, columns=columns, index=indexes)
            .astype("float64")
        )

        test_df = CCI(self.source, self.length, method="ema").astype("float64")

        pd.testing.assert_frame_equal(ref_df, test_df)

    def test_CCI_dema(self):
        np.random.seed(42)
        source = pd.Series(np.random.randint(1, 500, 20))
        length = 5

        np_values = np.array(
            [
                [107.0, 118.55999999999999, np.nan, np.nan],
                [72.0, 126.0, np.nan, np.nan],
                [189.0, 89.92, np.nan, np.nan],
                [21.0, 78.4, np.nan, np.nan],
                [103.0, 41.519999999999996, 179.26962962962963, -122.46247532053572],
                [122.0, 43.92, 160.65218106995889, -58.67058450206266],
                [467.0, 118.08000000000001, 186.41640603566532, 158.41440490307966],
                [215.0, 124.31999999999998, 201.0431275720165, 7.4843803238864846],
                [331.0, 121.12, 221.98354519128185, 60.004653681593],
                [459.0, 120.24000000000001, 257.6255591119241, 111.6513866090463],
                [88.0, 128.4, 254.61916984199573, -86.5104723997901],
                [373.0, 113.35999999999999, 266.43642240625024, 62.669711593595494],
                [100.0, 140.95999999999998, 253.19559883411335, -72.45346142362531],
                [360.0, 145.6, 259.1779440427066, 46.16394503539076],
                [152.0, 121.52000000000001, 249.9281036862251, -53.72399807232012],
                [131.0, 114.64000000000001, 232.602829784875, -59.085153398973596],
            ]
        )

        columns = ["source", "mad", "ma", "CCI"]
        indexes = list(range(4, 20))

        ref_df = (
            pd.DataFrame(np_values, columns=columns, index=indexes)
            .astype("float64")
        )

        test_df = CCI(source, length, method="dema").astype("float64")
        pd.testing.assert_frame_equal(ref_df, test_df)

    def test_CCI_tema(self):
        np.random.seed(42)

        source = pd.Series(np.random.randint(1, 500, 20))
        length = 5

        np_values = np.array(
            [
                [107.0, 118.55999999999999, np.nan, np.nan],
                [72.0, 126.0, np.nan, np.nan],
                [189.0, 89.92, np.nan, np.nan],
                [21.0, 78.4, np.nan, np.nan],
                [103.0, 41.519999999999996, np.nan, np.nan],
                [122.0, 43.92, np.nan, np.nan],
                [467.0, 118.08000000000001, np.nan, np.nan],
                [215.0, 124.31999999999998, np.nan, np.nan],
                [331.0, 121.12, 315.5154836153026, 8.522961462294898],
                [459.0, 120.24000000000001, 426.309255160968, 18.12527436184963],
                [88.0, 128.4, 208.47283541917164, -62.55079720621581],
                [373.0, 113.35999999999999, 310.723103546772, 36.62485088992473],
                [100.0, 140.95999999999998, 164.83331695271292, -30.662749220919846],
                [360.0, 145.6, 285.13889152281786, 34.2770643210541],
                [152.0, 121.52000000000001, 193.97353927069148, -23.026958125242196],
                [131.0, 114.64000000000001, 139.89768745991125, -5.174277424930945],
            ]
        )

        columns = ["source", "mad", "ma", "CCI"]
        indexes = list(range(4, 20))

        ref_df = (
            pd.DataFrame(np_values, columns=columns, index=indexes)
            .astype("float64")
        )

        test_df = CCI(source, length, method="tema").astype("float64")
        pd.testing.assert_frame_equal(ref_df, test_df)

    def test_CCI_rma(self):
        np_values = np.array(
            [
                [107.0, 118.55999999999999, 253.2, -82.20872694556905],
                [72.0, 126.0, 216.96, -76.69841269841271],
                [189.0, 89.92, 211.36800000000002, -16.583629893238452],
                [21.0, 78.4, 173.29440000000002, -129.50204081632657],
                [103.0, 41.519999999999996, 159.23552, -90.29466923570972],
                [122.0, 43.92, 151.788416, -45.21617486338799],
            ]
        )

        columns = ["source", "mad", "ma", "CCI"]
        indexes = [4, 5, 6, 7, 8, 9]
        ref_df = (
            pd.DataFrame(np_values, columns=columns, index=indexes)
            .astype("float64")
        )

        test_df = CCI(self.source, self.length, method="rma").astype("float64")

        pd.testing.assert_frame_equal(ref_df, test_df)

    def test_CCI_invalid_method(self):
        with self.assertRaises(InvalidArgumentError) as context:
            CCI(self.source, self.length, method="invalid_method")

        self.assertEqual(
            str(context.exception),
            "method must be 'sma', 'ema', 'sema', or 'rma', got 'invalid_method'.",
        )