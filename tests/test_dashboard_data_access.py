import sys
import unittest
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dashboard.data_access import *

class DashboardDataAccessTests(unittest.TestCase):
 def test_missing_reports_raise(self):
  with tempfile.TemporaryDirectory() as p:
   with self.assertRaises(DataIssue):read_csv(Path(p),'missing.csv')
   self.assertIsNone(read_csv(Path(p),'missing.csv',optional=True))
 def test_run_discovery_only_complete(self):
  with tempfile.TemporaryDirectory() as p:
   root=Path(p);a=root/'reports/tables/ml/valid';a.mkdir(parents=True)
   (a/'run_metadata.json').write_text('{}');(a/'holdout_metrics.csv').write_text('x\n1\n')
   (a.parent/'incomplete').mkdir()
   self.assertEqual(discover_runs(root),['valid'])
 def test_business_reconciliation(self):
  k=pd.Series({'transactions':3,'fraud_cases':1,'recorded_fraud_value':20,'fraud_rate_pct':33.333})
  frame=pd.DataFrame({'transactions':[1,2],'fraud_cases':[1,0],'fraud_amount':[20,0]})
  verify_business_consistency(k,frame)
  with self.assertRaises(DataIssue):verify_business_consistency(k,frame.assign(fraud_cases=[0,0]))
 def test_economics(self):
  row={'fraud_value_flagged':200.,'alerts':10,'fp':4}
  result=scenario_economics(row,review_cost=5,effectiveness=.5,friction_cost=3)
  self.assertAlmostEqual(result['gross'],100)
  self.assertAlmostEqual(result['cost'],62)
  self.assertAlmostEqual(result['net'],38)
  self.assertAlmostEqual(result['breakeven_effectiveness'],.31)
  with self.assertRaises(DataIssue):scenario_economics(row,review_cost=-1,effectiveness=.5,friction_cost=0)
  with self.assertRaises(DataIssue):scenario_economics(row,review_cost=1,effectiveness=1.5,friction_cost=0)
 def test_invalid_ml_confusion_rejected(self):
  with tempfile.TemporaryDirectory() as p:
   root=Path(p);base=root/'reports/tables/ml/run';base.mkdir(parents=True)
   (base/'run_metadata.json').write_text(json.dumps({'selected_model':'X','split':'chronological'}))
   pd.DataFrame([dict(selected_model='X',split='chronological',test_rows=20,test_fraud=2,
     tp=1,fp=1,fn=1,tn=17,pr_auc=.5,precision=.5,recall=.5,f1=.5,
     fraud_value_recall=.4,threshold=.5,alerts_per_1000=100)]).to_csv(base/'holdout_metrics.csv',index=False)
   get_run(root,'run')
   frame=pd.read_csv(base/'holdout_metrics.csv');frame.loc[0,'tn']=19;frame.to_csv(base/'holdout_metrics.csv',index=False)
   with self.assertRaises(DataIssue):get_run(root,'run')
 def test_optimization_selection_is_record_only(self):
  with tempfile.TemporaryDirectory() as p:
   root=Path(p);base=root/'reports/tables/investigation_optimization/run';base.mkdir(parents=True)
   rows=[]
   for policy in ('fraud_score','score_times_amount'):
    rows.append({'policy':policy,'capacity_per_10000':10.,'transactions':10000,'alerts':10,'tp':2,'fp':8,
      'case_recall_pct':20,'value_recall_pct':20,'fraud_value_flagged':100,
      'legitimate_value_flagged':30,'scenario_net_value':10,'fraud_value_total':500})
   pd.DataFrame(rows).to_csv(base/'holdout_policy_comparison.csv',index=False)
   pd.DataFrame([{'capacity_per_10000':10,'selected_policy':'score_times_amount','training_objective':'value_recall'}]).to_csv(base/'training_selected_policies.csv',index=False)
   df,sel=get_optimization(root,'run');self.assertEqual(sel.iloc[0].selected_policy,'score_times_amount')
   df.loc[df.policy=='fraud_score','alerts']=11;df.to_csv(base/'holdout_policy_comparison.csv',index=False)
   with self.assertRaises(DataIssue):get_optimization(root,'run')

if __name__=='__main__':unittest.main()
