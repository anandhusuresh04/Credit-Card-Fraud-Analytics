"""Run with: python -m unittest discover -s tests -p 'test_statistical_analysis.py'"""
import unittest
import numpy as np
import pandas as pd
from src.statistical_analysis import (
    wilson_interval, log_risk_ratio_interval, holm_adjust,
    segment_risk, duplicate_sensitivity, value_uncertainty,
)

class StatisticalAnalysisTests(unittest.TestCase):
    def test_wilson(self):
        lo, hi = wilson_interval(27, 1825)
        self.assertLess(lo, 27/1825)
        self.assertGreater(hi, 27/1825)
        self.assertTrue(0 < lo < hi < 1)

    def test_risk_ratio(self):
        rr, lo, hi = log_risk_ratio_interval(27, 1825, 465, 282982)
        self.assertAlmostEqual(rr, 9.0034, places=3)
        self.assertLess(lo, rr)
        self.assertGreater(hi, rr)

    def test_holm(self):
        actual = holm_adjust(np.array([0.01, 0.04, 0.03]))
        np.testing.assert_allclose(actual, [0.03, 0.06, 0.06])

    def test_duplicate_sensitivity_rejects_unrelated_file(self):
        original = pd.DataFrame({'Amount':[0,1,1,5,20,30,50,90,100,110],
                                 'Class':[0,0,0,0,0,1,0,1,0,1]})
        dedup = original.drop_duplicates().copy()
        result = duplicate_sensitivity(original, dedup)
        self.assertEqual(list(result['transactions']), [10,9])
        bad = dedup.copy()
        bad.loc[0,'Amount'] = 2
        with self.assertRaises(ValueError):
            duplicate_sensitivity(original,bad)

    def test_bootstrap_deterministic(self):
        a = np.tile([0.,1.,2.,5.,10.,20.,30.,70.,100.,150.], 10)
        y = np.tile([0,0,0,0,0,1,0,1,0,1], 10)
        frame = pd.DataFrame({'Amount':a, 'Class':y})
        x = value_uncertainty(frame,100,42)
        z = value_uncertainty(frame,100,42)
        pd.testing.assert_frame_equal(x,z)

if __name__ == '__main__':
    unittest.main()
