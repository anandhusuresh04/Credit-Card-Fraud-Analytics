"""Focused unit tests for src.model_evaluation; run with unittest discovery."""
import copy
import unittest
import numpy as np
import pandas as pd
from src.model_evaluation import audit_run


def fixture():
    d=pd.DataFrame({'original_row':[10,11,12,13,14,15], 'Time':[1,2,3,4,5,6],
       'Amount':[2.,3.,11.,7.,8.,9.], 'Class':[0,1,1,0,0,1],
       'score':[.1,.9,.3,.8,.2,.95]})
    thr=.5
    d['predicted_fraud']=(d.score>=thr).astype(int)
    d['error_type']=['TN','TP','FN','FP','TN','TP']
    from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
    y=d.Class.to_numpy(); p=d.predicted_fraud.to_numpy();s=d.score.to_numpy()
    m={'tn':2,'fp':1,'fn':1,'tp':2,'precision':2/3,'recall':2/3,'f1':2/3,
       'pr_auc':average_precision_score(y,s), 'roc_auc':roc_auc_score(y,s),
       'brier_score':brier_score_loss(y,s),'alerts':3,'alerts_per_1000':500.,
       'flagged_fraud_amount':12.,'missed_fraud_amount':11.,'fraud_value_recall':12/23,
       'threshold':thr,'selected_model':'XGBoost__SMOTE','split':'chronological','test_rows':6,'test_fraud':3}
    sel=pd.DataFrame([{'key':'XGBoost__SMOTE','cv_oof_ap':.7,'threshold':thr},
                      {'key':'RandomForest__baseline','cv_oof_ap':.6,'threshold':.3}])
    folds=pd.DataFrame([{'key':k,'fold':f,'fold_ap':.6,'train_rows':10,'validation_rows':5}
                        for k in sel.key for f in [1,2]])
    meta={'split':'chronological','selected_model':'XGBoost__SMOTE',
          'selected_threshold':thr,'run_id':'fixture'}
    return d,pd.DataFrame([m]),sel,folds,meta


class ModelEvaluationTests(unittest.TestCase):
    def test_recompute_all_metrics(self):
        result=audit_run(*fixture())
        self.assertEqual((result['tp'],result['fn'],result['fp'],result['tn']),(2,1,1,2))
        self.assertEqual(result['false_positive_legitimate_value'],7.)
    def test_threshold_tampering_rejected(self):
        d,m,s,f,meta=fixture();d.loc[0,'predicted_fraud']=1
        with self.assertRaises(ValueError):audit_run(d,m,s,f,meta)
    def test_metric_tampering_rejected(self):
        d,m,s,f,meta=fixture();m.loc[0,'pr_auc']=.99
        with self.assertRaises(ValueError):audit_run(d,m,s,f,meta)
    def test_test_selected_model_rejected(self):
        d,m,s,f,meta=fixture();m.loc[0,'selected_model']='RandomForest__baseline'
        with self.assertRaises(ValueError):audit_run(d,m,s,f,meta)
    def test_duplicate_rows_rejected(self):
        d,m,s,f,meta=fixture();d.loc[1,'original_row']=10
        with self.assertRaises(ValueError):audit_run(d,m,s,f,meta)
    def test_error_label_rejected(self):
        d,m,s,f,meta=fixture();d.loc[1,'error_type']='FP'
        with self.assertRaises(ValueError):audit_run(d,m,s,f,meta)

if __name__ == '__main__': unittest.main()

class VisualizationRegressionTests(unittest.TestCase):
    def test_three_figures_create_png_and_svg(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from src.model_evaluation import confusion_figure, calibration_descriptive, errors_figure
        d,m,s,f,meta=fixture()
        audit=audit_run(d,m,s,f,meta)
        with TemporaryDirectory() as tmp:
            out=Path(tmp)
            confusion_figure(audit,out,['png','svg'])
            calibration_descriptive(d,out,['png','svg'])
            errors_figure(d,out,['png','svg'])
            for name in ['01_confusion_matrix','03_descriptive_reliability','04_error_characteristics']:
                for ext in ['png','svg']:
                    self.assertGreater((out / f'{name}.{ext}').stat().st_size,500)

    def test_chart_functions_handle_single_outcome_group(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from src.model_evaluation import calibration_descriptive, errors_figure
        d,*_=fixture()
        d=d.loc[d.Class.eq(0)].copy()
        d['score']=[0.,0.,1e-8]  # duplicate and zero scores
        with TemporaryDirectory() as tmp:
            out=Path(tmp)
            calibration_descriptive(d,out,['png'])
            errors_figure(d,out,['png'])
            self.assertTrue((out/'03_descriptive_reliability.png').exists())
            self.assertTrue((out/'04_error_characteristics.png').exists())
