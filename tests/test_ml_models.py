"""Run: python -m unittest discover -s tests -p test_ml_models.py -v"""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.ml_models import (REQUIRED, load_data, make_split, fold_indices, threshold_table,
                           choose_threshold, metric_row, out_of_fold, bootstrap_ci)


def dataset(n=240):
    rng = np.random.default_rng(7)
    d = pd.DataFrame({c: rng.normal(size=n) for c in REQUIRED if c.startswith('V')})
    d['Time'] = np.arange(n, dtype=float)
    d['Amount'] = rng.uniform(1, 500, size=n)
    d['Class'] = ((np.arange(n) % 12) == 0).astype(int)
    return d[REQUIRED]


class MlTests(unittest.TestCase):
    def test_load_rejects_invalid_data(self):
        df = dataset(120)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'data.csv'
            df.to_csv(path, index=False)
            loaded, info = load_data(path, keep_duplicates=False)
            self.assertEqual(len(loaded), 120)
            self.assertEqual(info['exact_duplicate_rows'], 0)
            df.loc[0, 'Amount'] = -5
            df.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_data(path, keep_duplicates=False)

    def test_chronological_validation_is_past_only(self):
        df = dataset().reset_index().rename(columns={'index': 'original_row'})
        sp = make_split(df, 'chronological', .25, True)
        self.assertLess(sp.train.Time.max(), sp.test.Time.min())
        self.assertNotIn('Time', sp.features)
        for a, b in fold_indices(sp, 3):
            self.assertLess(sp.train.iloc[a].Time.max(), sp.train.iloc[b].Time.min())

    def test_exact_threshold_and_capacity(self):
        y = np.array([0, 1, 0, 1])
        score = np.array([.1, .9, .2, .8])
        money = np.array([10., 100., 5., 200.])
        curve = threshold_table(y, score, money, 5, 1, 0)
        thr, row = choose_threshold(curve, 'f1', 500, len(y))
        self.assertAlmostEqual(row['f1'], 1)
        self.assertEqual(row['alerts'], 2)
        self.assertEqual(int((score >= thr).sum()), 2)

    def test_metric_calculation(self):
        result = metric_row(np.array([0, 0, 1, 1]), np.array([.1,.9,.8,.2]), .5,
                            np.array([1, 1, 100, 50]), 5, .5, 2)
        self.assertEqual((result['tn'], result['fp'], result['fn'], result['tp']), (1,1,1,1))
        self.assertEqual(result['alerts'], 2)
        self.assertAlmostEqual(result['fraud_value_recall'], 2/3)
        self.assertAlmostEqual(result['scenario_net_value'], 50 - 12)

    def test_oof_not_test_and_bootstrap_deterministic(self):
        df = dataset().reset_index().rename(columns={'index':'original_row'})
        sp = make_split(df, 'random', .25, False)
        oof, folds = out_of_fold(sp, 'LogisticRegression', 'baseline', 3)
        self.assertEqual(np.isfinite(oof).sum(), len(sp.train))
        self.assertEqual(len(folds), 3)
        y = np.array([0, 1, 0, 1, 1, 0])
        s = np.array([.1, .8, .2, .7, .4, .3])
        amt = np.array([1., 20., 2., 40., 30., 5.])
        self.assertEqual(bootstrap_ci(y,s,.5,amt,30), bootstrap_ci(y,s,.5,amt,30))

if __name__ == '__main__':
    unittest.main()
