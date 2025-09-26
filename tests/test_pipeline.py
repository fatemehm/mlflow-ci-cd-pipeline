import pandas as pd
import os
def test_fix_predictions():
    df = pd.read_csv('prediction.csv')
    df['predicted_label'] = df['predicted_label'].replace({'POSITIVE': 1, 'positive': 1, 'NEGATIVE': 0, 'negative': 0})
    df['predicted_label'] = pd.to_numeric(df['predicted_label'], errors='coerce').fillna(0).astype(int)
    df = df.rename(columns={'true_label': 'target', 'predicted_label': 'prediction', 'prediction_score': 'confidence'})
    n = len(df) // 2
    df.iloc[:n].to_csv('data/reference_predictions.csv', index=False)
    df.iloc[n:].to_csv('data/predictions.csv', index=False)
    ref_df = pd.read_csv('data/reference_predictions.csv')
    cur_df = pd.read_csv('data/predictions.csv')
    assert 'prediction' in ref_df.columns, "Missing prediction column"
    assert ref_df['prediction'].isin([0, 1]).all(), "Non-numeric predictions in reference"
    assert cur_df['prediction'].isin([0, 1]).all(), "Non-numeric predictions in current"
def test_monitor_output():
    os.system('python src/monitor.py')
    assert os.path.exists('reports/sentiment_drift_report.html'), "Report not generated"