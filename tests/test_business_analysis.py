"""Run: python -m unittest discover -s tests -p test_business_analysis.py -v"""
import unittest
import numpy as np
import pandas as pd
from src.business_analysis import (
    _summary, amount_bands, amount_exposure, time_exposure, zero_amount_exposure,
    review_policy_table, assumption_scenarios, _check_reconciliation,
)


class BusinessAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'Time': [0, 400, 7200, 10000, 16000, 22000, 30000, 40000, 50000, 60000],
            'Amount': [0, 0, 2, 3, 4, 5, 6, 8, 10, 100],
            'Class': [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        })

    def test_reconcile_segments(self):
        a = amount_exposure(self.df, 5)
        t = time_exposure(self.df, 4)
        z = zero_amount_exposure(self.df)
        p = review_policy_table(self.df, amount_bands(self.df, 5))
        _check_reconciliation(self.df, a, t, z, p)
        self.assertAlmostEqual(a.fraud_amount.sum(), 109.0)
        self.assertEqual(int(z.fraud_cases.sum()), 4)

    def test_zero_amount_has_no_value(self):
        p = review_policy_table(self.df, amount_bands(self.df, 5))
        row = p.set_index('policy').loc['review_zero_amount']
        self.assertEqual(int(row.flagged_for_review), 2)
        self.assertEqual(int(row.labeled_fraud_in_flagged), 1)
        self.assertEqual(float(row.flagged_fraud_value), 0.0)

    def test_scenarios_charge_every_review(self):
        p = review_policy_table(self.df, amount_bands(self.df, 5))
        s = assumption_scenarios(p, (2,), (0.5,), 3)
        row = s.set_index('policy').loc['review_zero_amount']
        self.assertAlmostEqual(row.total_review_cost, 4)
        self.assertAlmostEqual(row.total_false_positive_friction_cost, 3)
        self.assertAlmostEqual(row.hypothetical_incremental_net_value, -7)

    def test_invalid_assumptions_fail(self):
        p = review_policy_table(self.df, amount_bands(self.df, 5))
        with self.assertRaises(ValueError):
            assumption_scenarios(p, (-1,), (0.5,))
        with self.assertRaises(ValueError):
            assumption_scenarios(p, (5,), (1.1,))

    def test_all_policy_matches_dataset(self):
        p = review_policy_table(self.df, amount_bands(self.df, 5))
        row = p.set_index('policy').loc['review_all']
        self.assertEqual(row.flagged_for_review, len(self.df))
        self.assertEqual(row.labeled_fraud_in_flagged, 4)
        self.assertAlmostEqual(row.value_coverage_pct, 100)

    def test_no_scaling_or_modification(self):
        original = self.df.copy(deep=True)
        amount_exposure(self.df, 5)
        time_exposure(self.df)
        review_policy_table(self.df, amount_bands(self.df, 5))
        pd.testing.assert_frame_equal(self.df, original)

    def test_quantile_failure_for_constant_amount(self):
        df = self.df.copy()
        df['Amount'] = 0
        with self.assertRaises(ValueError):
            amount_bands(df)


if __name__ == '__main__':
    unittest.main()
