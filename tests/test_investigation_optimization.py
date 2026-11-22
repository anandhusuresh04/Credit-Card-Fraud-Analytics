import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import investigation_optimization as io

class InvestigationOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.d=pd.DataFrame({'original_row':[9,2,4,1,7,5],
            'Class':[0,1,1,0,0,1],'Amount':[100.,2.,200.,50.,0.,10.],
            'score':[.9,.8,.7,.6,.5,.4]})

    def test_capacity_exact_and_never_exceeded(self):
        d=pd.concat([self.d]*1000,ignore_index=True)
        d['original_row']=np.arange(len(d))
        for cap in (5,10,20,50):
            f=io.allocations(d,'score',cap,'fraud_score')
            self.assertEqual(f.sum(),int(np.floor(len(d)*cap/10000)))

    def test_ties_deterministic_reorder(self):
        d=self.d.copy();d['score']=.5
        ids1=set(d.loc[io.allocations(d,'score',5000,'fraud_score'),'original_row'])
        d=d.sample(frac=1,random_state=42)
        ids2=set(d.loc[io.allocations(d,'score',5000,'fraud_score'),'original_row'])
        self.assertEqual(ids1,ids2)
        self.assertEqual(ids1,{1,2,4})

    def test_value_ranking_differs_from_probability_ranking(self):
        p=set(self.d.loc[io.allocations(self.d,'score',10000/6,'fraud_score'),'original_row'])
        v=set(self.d.loc[io.allocations(self.d,'score',10000/6,'score_times_amount'),'original_row'])
        self.assertEqual(p,{9})
        self.assertEqual(v,{4})

    def test_reconciliation_and_cost_includes_all_reviews(self):
        flag=io.allocations(self.d,'score',5000,'fraud_score')
        r=io.outcomes(self.d,flag,5000,'fraud_score',5,.5,2)
        self.assertEqual(r['alerts'],3)
        self.assertEqual(r['tp'],2)
        self.assertEqual(r['fp'],1)
        self.assertEqual(r['review_cost_total'],15)
        self.assertEqual(r['friction_cost_total'],2)
        self.assertAlmostEqual(r['fraud_value_flagged']+r['fraud_value_missed'],212)
        self.assertAlmostEqual(r['scenario_net_value'],(202*.5)-17)

    def test_choice_uses_only_training_grid(self):
        g=io.grid(self.d,'score',[10000/3],5,.5,0)
        w=io.training_choice(g,'value_recall')
        self.assertEqual(w.iloc[0].selected_policy,'score_times_amount')
        # Holdout not supplied to the selector; no possible holdout label access.
        self.assertEqual(len(w),1)

    def test_reject_bad_data(self):
        d=self.d.copy();d.loc[0,'score']=1.5
        with self.assertRaises(ValueError):io.validated_frame(d,'score','test')
        d=self.d.copy();d.loc[0,'original_row']=2
        with self.assertRaises(ValueError):io.validated_frame(d,'score','test')
        d=self.d.copy();d.loc[0,'Amount']=-1
        with self.assertRaises(ValueError):io.validated_frame(d,'score','test')

    def test_invalid_capacities(self):
        for s in ('0','-2','10001','nan','inf','5,5',''):
            with self.assertRaises(ValueError):io.parse_capacities(s)

    def test_zero_amount_ranks_at_zero_for_value_policy(self):
        d=self.d.copy();d.loc[d.original_row==7,'score']=1
        a=io.allocations(d,'score',10000/6,'score_times_amount')
        self.assertEqual(int(d.loc[a,'original_row'].iloc[0]),4)

if __name__=='__main__':unittest.main()
