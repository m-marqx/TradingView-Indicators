import unittest
import pandas as pd
import numpy as np
from src.tradingview_indicators.didi_index import didi_index as DidiIndex
from src.tradingview_indicators.errors_exceptions import InvalidArgumentError

class TestDidiIndex(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.source = pd.Series(np.random.randint(1, 500, 50))
        self.short_length = 10
        self.mid_length = 20
        self.long_length = 30

    def test_didi_index_sma(self):
        test_values = DidiIndex(
            self.source,
            self.short_length,
            self.mid_length,
            self.long_length,
            ma_method="sma",
        )

        ref_values = pd.Series(
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
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                -70.56666666666666,
                -77.46666666666664,
                -71.83333333333331,
                -88.33333333333331,
                -83.89999999999998,
                -39.73333333333329,
                -17.600000000000023,
                -10.233333333333348,
                4.666666666666686,
                17.16666666666663,
                25.83333333333337,
                34.03333333333333,
                31.633333333333326,
                34.866666666666646,
                38.43333333333334,
                6.7666666666666515,
                1.5666666666666629,
                9.166666666666686,
                1.9333333333333371,
                -9.899999999999977,
                3.5,
            ],
            index=range(9, 50),
        )

        pd.testing.assert_series_equal(test_values, ref_values)

    def test_didi_index_ema(self):
        test_values = DidiIndex(
            self.source,
            self.short_length,
            self.mid_length,
            self.long_length,
            ma_method="ema",
        )

        ref_values = pd.Series(
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
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                -70.21461720194196,
                -59.70834271556737,
                -37.359454117895666,
                -54.88878625729774,
                -50.53570867182782,
                -12.358887309672468,
                -10.090306885152756,
                -6.2414484735977,
                -16.008258328992838,
                11.425453287012544,
                -17.681593872432188,
                9.045054417522863,
                16.374893086168527,
                -14.101387499837756,
                -3.4794041653752856,
                -27.683934174301385,
                -23.245947651229528,
                -10.074984973438745,
                -29.905128085470494,
                -12.9433886130318,
                -31.59797055802113,
            ],
            index=range(9, 50),
        )

        pd.testing.assert_series_equal(test_values, ref_values)

    def test_didi_index_rma(self):
        test_values = DidiIndex(
            self.source,
            self.short_length,
            self.mid_length,
            self.long_length,
            ma_method="rma",
        )

        ref_values = pd.Series(
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
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                -40.102675375069,
                -37.376852282006496,
                -27.147463350102157,
                -38.09073676817843,
                -37.75108218601076,
                -17.86307909223825,
                -15.900972803682038,
                -13.007603756625855,
                -17.713680673164845,
                -2.0155886549765967,
                -16.75152997030284,
                -1.7826271480690252,
                3.846260397767935,
                -11.669428305346457,
                -5.9291793827048025,
                -19.315398888730897,
                -18.030358529344483,
                -11.452272221580472,
                -22.75782955975393,
                -14.451138345432298,
                -25.242813194487724,
            ],
            index=range(9, 50),
            name="RMA",
        )

        pd.testing.assert_series_equal(test_values, ref_values)

    def test_didi_index_dema(self):
        test_values = DidiIndex(
            self.source,
            self.short_length - 5,
            self.mid_length - 10,
            self.long_length - 20,
            ma_method="dema",
        )

        ref_values = pd.Series(
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
                -6.860266507219677,
                6.61702470736725,
                17.969899667745807,
                13.898053379543853,
                8.280525853113005,
                -3.8300330224965364,
                -27.91663490579495,
                -46.081295186117245,
                -48.79997666032841,
                -51.476004932103365,
                -37.74352499112672,
                -39.02479091689821,
                -32.2884691408438,
                -14.010364634607242,
                -17.4398829771485,
                -17.066308706948064,
                7.78372675197636,
                19.819980564785055,
                25.865542111305274,
                19.655859460500608,
                32.526296700114585,
                16.896516534238856,
                25.021998217885482,
                31.967340045049013,
                12.902076342752707,
                9.20924239672911,
                -9.568045607204795,
                -15.838554601966166,
                -9.62106432114075,
                -19.42771031750425,
                -12.36306598656438,
                -20.88702373046982,
            ],
            index=range(8, 50),
            name="sema",
        )

        pd.testing.assert_series_equal(test_values, ref_values)

    def test_didi_index_tema(self):
        test_values = DidiIndex(
            self.source,
            self.short_length - 5,
            self.mid_length - 10,
            self.long_length - 20,
            ma_method="tema",
        )

        ref_values = pd.Series(
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
                np.nan,
                np.nan,
                np.nan,
                9.39772330661981,
                59.60604592025817,
                -1.9108876382745166,
                24.495986229683695,
                56.52730795369479,
                -25.279031069543578,
                -7.034147254054176,
                73.53760326350269,
                16.577093027683873,
                -5.704907928905783,
                -43.445354110011806,
                27.65228335806998,
                -70.1725229470697,
                21.538627313538257,
                12.716442463321243,
                -73.92432290786701,
                -8.536747061030724,
                -55.79760596231978,
                -3.938504414320846,
                36.47047972859059,
                -23.456605800649243,
                35.845120444417546,
                -22.27323545340579,
            ],
            index=range(12, 50),
            name="sema",
        )

        pd.testing.assert_series_equal(test_values, ref_values)

    def test_didi_index_absolute(self):
        test_values = DidiIndex(
            self.source,
            self.short_length,
            self.mid_length,
            self.long_length,
            method="absolute",
        )

        ref_values = pd.Series(
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
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                -70.21461720194196,
                -59.70834271556737,
                -37.359454117895666,
                -54.88878625729774,
                -50.53570867182782,
                -12.358887309672468,
                -10.090306885152756,
                -6.2414484735977,
                -16.008258328992838,
                11.425453287012544,
                -17.681593872432188,
                9.045054417522863,
                16.374893086168527,
                -14.101387499837756,
                -3.4794041653752856,
                -27.683934174301385,
                -23.245947651229528,
                -10.074984973438745,
                -29.905128085470494,
                -12.9433886130318,
                -31.59797055802113,
            ],
            index=range(9, 50),
        )

        pd.testing.assert_series_equal(test_values, ref_values)

    def test_didi_index_ratio(self):
        test_values = DidiIndex(
            self.source,
            self.short_length,
            self.mid_length,
            self.long_length,
            method="ratio",
        )

        ref_values = pd.Series(
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
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                -0.23766079203006663,
                -0.20330795730146034,
                -0.13292925527786947,
                -0.18413449808371996,
                -0.16867394630253607,
                -0.045242933884033554,
                -0.03719976131942615,
                -0.023298572735442225,
                -0.05816244852767216,
                0.04503750714065702,
                -0.06435172105554199,
                0.03558021326420102,
                0.06651180230717224,
                -0.052601128872100666,
                -0.013358963224817999,
                -0.09892774960488471,
                -0.0833193974859715,
                -0.0372427529232342,
                -0.10411408749667517,
                -0.04680387395671615,
                -0.1079594144719489,
            ],
            index=range(9, 50),
        )

        pd.testing.assert_series_equal(test_values, ref_values)

    def test_didi_index_dtw_absolute(self):
        test_values = DidiIndex(
            self.source,
            self.short_length,
            self.mid_length,
            self.long_length,
            method="absolute",
            use_dtw=True,
        )

        ref_values = pd.Series(
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
                -44.42527287644759,
                -67.21537076874858,
                -53.39280298560624,
                -41.02545747918009,
                -53.3092692342982,
                -24.097051990099942,
                5.505482951036356,
                9.222018699150283,
                3.4780695771374894,
                0.6615775977817293,
                20.02512001292905,
                -3.978719709092786,
                0.04889892718273359,
                11.355418521763767,
                2.5071654879276934,
                5.023355722727501,
                -18.392252018223786,
                0.0008009306823169027,
                5.183243498634852,
                -18.07451885477792,
                -5.135807410497591,
            ],
            index=range(19, 50),
        )

        pd.testing.assert_series_equal(test_values, ref_values)

    def test_didi_index_dtw_ratio(self):
        test_values = DidiIndex(
            self.source,
            self.short_length,
            self.mid_length,
            self.long_length,
            method="ratio",
            use_dtw=True,
        )

        ref_values = pd.Series(
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
                -0.15036962328801762,
                -0.2282115357609592,
                -0.18615024552145487,
                -0.13945161637964543,
                -0.17831863509013635,
                -0.08173431871698655,
                0.020099316787192678,
                0.03405581768250332,
                0.013591430843266972,
                0.0022481270104341977,
                0.07893616672521742,
                -0.014971464088080477,
                0.005106881907246952,
                0.04735950676032796,
                0.010414767850309259,
                0.019356122485231664,
                -0.06798254083923627,
                0.005017243821823891,
                0.021376548886632873,
                -0.0634627559777865,
                -0.018780256715902555,
            ],
            index=range(19, 50),
        )

        pd.testing.assert_series_equal(test_values, ref_values)

    def test_didi_index_invalid_ma_method(self):
        with self.assertRaises(InvalidArgumentError) as context:
            DidiIndex(
                self.source,
                self.short_length,
                self.mid_length,
                self.long_length,
                method="invalid_method",
            )

        self.assertEqual(
            str(context.exception),
            "method must be 'absolute' or 'ratio'. got 'invalid_method'.",
        )

    def test_didi_index_invalid_method(self):
        with self.assertRaises(InvalidArgumentError) as context:
            DidiIndex(
                self.source,
                self.short_length,
                self.mid_length,
                self.long_length,
                ma_method="invalid_method",
            )

        self.assertEqual(
            str(context.exception),
            "ma_method must be 'sma', 'ema', 'dema', 'tema', or 'rma'. got 'invalid_method'.",
        )