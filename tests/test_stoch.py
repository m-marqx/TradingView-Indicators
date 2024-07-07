import unittest
import pandas as pd
import numpy as np
from src.tradingview_indicators.stoch import stoch


class TestStoch(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        source = pd.read_csv("example/BTCUSDT_1d_spot.csv", index_col=0).iloc[:-1]
        self.short_source = source.iloc[:23]
        self.medium_source = source.iloc[:83]
        self.long_source = source.iloc[:119]

        self.k_length = 14
        self.k_smoothing = 1
        self.d_smoothing = 3

    def test_stoch(self):
        ref_indexes = pd.Index(
            [
                "2017-08-30",
                "2017-08-31",
                "2017-09-01",
                "2017-09-02",
                "2017-09-03",
                "2017-09-04",
                "2017-09-05",
                "2017-09-06",
                "2017-09-07",
                "2017-09-08",
            ],
            name="open_time",
        )

        ref_values = pd.Series(
            [
                92.59565053586745,
                98.47408244265733,
                96.59116152266836,
                69.65611782820838,
                72.05608144543561,
                45.48561256245167,
                57.13783219452327,
                76.09471706868041,
                81.47119795837419,
                50.87599817391242,
            ],
            index=ref_indexes,
            name="stoch",
        )

        test_stoch = stoch(
            self.short_source["close"],
            self.short_source["high"],
            self.short_source["low"],
            self.k_length,
        ).dropna()

        pd.testing.assert_series_equal(test_stoch, ref_values)
