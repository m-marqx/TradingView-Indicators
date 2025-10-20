import unittest
import pandas as pd
import numpy as np
from src.tradingview_indicators.bollinger import bollinger_bands
from src.tradingview_indicators.errors_exceptions import InvalidArgumentError

class TestBollingerBands(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.source = pd.Series(np.random.randint(1, 500, 50))
        self.short_length = 10
        self.stdev = 2

    def test_bollinger_bands_sma(self):
        ref_basis = [
            177.3,
            213.7,
            191.6,
            189.8,
            208.6,
            206.7,
            236.8,
            227.9,
            261.8,
            266.7,
        ]

        ref_upper = [
            443.8683985605029,
            529.9545388343594,
            467.0700225674898,
            460.89948973270555,
            526.7225480149993,
            527.7331239399864,
            558.1455046104322,
            559.8831990394159,
            568.1256074470068,
            563.1441862401007,
        ]

        ref_lower = [
            -89.26839856050287,
            -102.55453883435939,
            -83.87002256748983,
            -81.29948973270552,
            -109.52254801499933,
            -114.33312393998642,
            -84.5455046104322,
            -104.08319903941592,
            -44.52560744700685,
            -29.74418624010076,
        ]

        ref_values = {
            "basis": ref_basis,
            "upper": ref_upper,
            "lower": ref_lower,
        }

        ref_df = pd.DataFrame(ref_values, index=range(9, 19))

        test_df = bollinger_bands(
            self.source,
            self.short_length,
            self.stdev,
            ma_method="sma",
        ).iloc[9:19]

        pd.testing.assert_frame_equal(ref_df, test_df)

    def test_bollinger_bands_ema(self):
        ref_basis = [
            177.3,
            229.97272727272727,
            227.25041322314047,
            246.11397445529673,
            284.82052455433364,
            249.03497463536388,
            271.5740701562068,
            240.3787846732601,
            262.12809655084914,
            242.10480626887653,
        ]

        ref_upper = [
            443.8683985605029,
            546.2272661070866,
            502.7204357906303,
            517.2134641880023,
            602.943072569333,
            570.0680985753503,
            592.919574766639,
            572.361983712676,
            568.453703997856,
            538.5489925089773,
        ]

        ref_lower = [
            -89.26839856050287,
            -86.28181156163211,
            -48.21960934434935,
            -24.98551527740881,
            -33.30202346066568,
            -71.99814930462253,
            -49.771434454225414,
            -91.60441436615582,
            -44.19751089615772,
            -54.33937997122422,
        ]

        ref_values = {
            "basis": ref_basis,
            "upper": ref_upper,
            "lower": ref_lower,
        }

        ref_df = pd.DataFrame(ref_values, index=range(9, 19))

        test_df = bollinger_bands(
            self.source,
            self.short_length,
            self.stdev,
            ma_method="ema",
        ).iloc[9:19]

        pd.testing.assert_frame_equal(ref_df, test_df)

    def test_bollinger_bands_dema(self):
        ref_basis = [
            243.06783717900544,
            239.21985449224226,
            233.694515520368,
            232.4851723897623,
            232.51907748779445,
            236.22709404142608,
            247.16462267513234,
            260.0016584489313,
            269.71895815043536,
            280.06792875556005,
        ]

        ref_upper = [
            539.5120234191062,
            533.7600988656649,
            499.7077983967436,
            501.83838947757715,
            495.42928607197587,
            463.86862510873084,
            506.72228058138205,
            528.1608151699602,
            509.4599294785196,
            523.6960606834363,
        ]

        ref_lower = [
            -53.37634906109531,
            -55.320389881180375,
            -32.31876735600764,
            -36.86804469805256,
            -30.39113109638697,
            8.585562974121359,
            -12.393035231117409,
            -8.157498272097541,
            29.977986822351113,
            36.439796827683836,
        ]

        ref_values = {
            "basis": ref_basis,
            "upper": ref_upper,
            "lower": ref_lower
        }

        ref_df = pd.DataFrame(ref_values, index=range(18, 28))

        test_df = bollinger_bands(
            self.source,
            self.short_length,
            self.stdev,
            ma_method="dema",
        ).iloc[18:28]

        pd.testing.assert_frame_equal(ref_df, test_df)

    def test_bollinger_bands_tema(self):
        ref_basis = [
            308.3061063292175,
            368.6533146577107,
            332.87025154352443,
            255.7309941597544,
            339.6853808636725,
            330.0673217639549,
            191.18167685702826,
            203.68236654130544,
            206.2695317937874,
            258.88733060418315,
        ]

        ref_upper = [
            542.7870934760081,
            587.7059212261888,
            518.0932402775246,
            473.16818815685633,
            566.3014536485578,
            557.6169744651718,
            470.5686247745476,
            471.255907834162,
            475.03502230343014,
            521.2682242175202,
        ]

        ref_lower = [
            73.82511918242704,
            149.60070808923265,
            147.64726280952425,
            38.293800162652474,
            113.06930807878717,
            102.51766906273798,
            -88.20527106049104,
            -63.89117475155109,
            -62.49595871585535,
            -3.493563009153945,
        ]

        ref_values = {
            "basis": ref_basis,
            "upper": ref_upper,
            "lower": ref_lower
        }

        ref_df = pd.DataFrame(ref_values, index=range(28, 38))

        test_df = bollinger_bands(
            self.source,
            self.short_length,
            self.stdev,
            ma_method="tema",
        ).iloc[28:38]

        pd.testing.assert_frame_equal(ref_df, test_df)

    def test_bollinger_bands_rma(self):
        ref_basis = [
            177.3,
            206.27000000000004,
            207.14300000000003,
            219.52870000000001,
            243.47583000000003,
            227.92824700000006,
            242.43542230000006,
            228.19188007000005,
            241.37269206300004,
            232.43542285670003,
        ]

        ref_upper = [
            443.8683985605029,
            522.5245388343594,
            482.61302256748985,
            490.62818973270555,
            561.5983780149993,
            548.9613709399864,
            563.7809269104323,
            560.175079109416,
            547.6982995100069,
            528.8796090968008,
        ]

        ref_lower = [
            -89.26839856050287,
            -109.98453883435934,
            -68.32702256748979,
            -51.57078973270552,
            -74.6467180149993,
            -93.10487693998635,
            -78.91008231043216,
            -103.79131896941587,
            -64.95291538400681,
            -64.00876338340072,
        ]

        ref_values = {
            "basis": ref_basis,
            "upper": ref_upper,
            "lower": ref_lower
        }

        ref_df = pd.DataFrame(ref_values, index=range(9, 19))

        test_df = bollinger_bands(
            self.source,
            self.short_length,
            self.stdev,
            ma_method="rma",
        ).iloc[9:19]

        pd.testing.assert_frame_equal(ref_df, test_df)

    def test_bollinger_bands_invalid_ma_method(self):
        with self.assertRaises(InvalidArgumentError) as context:
            bollinger_bands(
                self.source,
                self.short_length,
                self.stdev,
                ma_method="invalid_method",
            )

        self.assertEqual(
            str(context.exception),
            "ma_method must be 'sma', 'ema', 'dema', 'tema', or 'rma', got 'invalid_method'.",
        )
