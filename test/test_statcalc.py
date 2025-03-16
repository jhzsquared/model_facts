# Test stat calculation within the ModelFacts object
from modelfacts import ModelFacts
from sklearn.metrics import f1_score, r2_score
import pandas as pd
def test_class_accuracy():
        class_df  = pd.DataFrame({
                'DemoLabel': [1,0,1,0,1,0],
                'True': [1,1,1,0,0,0],
                'Pred': [0,1,1,0,1,1],
                'Baseline': [0,1,0,1,0,1]})
        mf_class = ModelFacts(class_df, 'True', 'Pred', 
                              'Baseline', f1_score,f1_score)
        results = mf_class.calc_accuracy(f1_score)
        assert results['Name'] == 'f1_score'
        assert round(results['Raw Score'],3) == .571 
        assert round(results['% Over Baseline'],3) == 71.429

def test_reg_accuracy():
        regress_df  = pd.DataFrame({
                'DemoLabel': [1,0,1,0,1,0],
                'True': [1,2,3,4,5,6],
                'Pred': [0,1,2,3,4,5],
                'Baseline': [2,2,2,2,2,2]})
        mf_reg = ModelFacts(regress_df, 'True', 'Pred', 
                              'Baseline', r2_score, r2_score,
                                classification = False)
        results = mf_reg.calc_accuracy(r2_score)
        assert results['Name'] == 'r2_score'
        assert round(results['Raw Score'],3) == .657
        assert round(results['% Over Baseline'],3) == -185.185

def test_class_demo():
        class_df  = pd.DataFrame({
                'DemoLabel': [1,0,1,0,1,0],
                'True': [1,1,1,0,0,0],
                'Pred': [0,1,1,0,1,1],
                'Baseline': [0,1,0,1,0,1]})
        mf_class = ModelFacts(class_df, 'True', 'Pred', 
                              'Baseline', f1_score,f1_score)
        results = mf_class.calc_demo('DemoLabel', age = False)
        assert results.columns[2] == '% Target'
        assert results.loc[0, 'Training Score'] == 2/3

def test_reg_demo():
        regress_df  = pd.DataFrame({
                'DemoLabel': [1,0,1,0,1,0],
                'True': [1,2,3,4,5,6],
                'Pred': [0,1,2,3,4,5],
                'Baseline': [2,2,2,2,2,2]})
        mf_reg = ModelFacts(regress_df, 'True', 'Pred', 
                              'Baseline', r2_score, r2_score,
                                classification = False)
        results = mf_reg.calc_demo('DemoLabel', age = False)
        assert results.columns[2] == 'Mean, Std'
        assert results.loc[0, 'Mean, Std'] == (4, 2)

def test_data_tables():
        class_df  = pd.DataFrame({
                'DemoLabel': [1,0,1,0,1,0],
                'True': [1,1,1,0,0,0],
                'Pred': [0,1,1,0,1,1],
                'Baseline': [0,1,0,1,0,1]})
        mf_class = ModelFacts(class_df, 'True', 'Pred', 
                              'Baseline', f1_score,f1_score)
        data = mf_class(['DemoLabel'])
        assert data[0].shape == (5,1)
        assert data[0].columns[0] == 0
        assert data[1].shape == (2,3)
        assert data[2].shape == (2,4)

if __name__ == "main":
        test_class_accuracy()
        test_reg_accuracy()
        test_class_demo()
        test_reg_demo()
        test_data_tables()