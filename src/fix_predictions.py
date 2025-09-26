import pandas as pd
import os

# Load prediction.csv
df = pd.read_csv('data/predictions.csv')

# Map prediction column (string labels) to numeric (0/1)
label_mapping = {
    'POSITIVE': 1,
    'positive': 1,
    'NEGATIVE': 0,
    'negative': 0
}
df['prediction'] = df['prediction'].map(label_mapping).fillna(df['prediction'])
df['prediction'] = pd.to_numeric(df['prediction'], errors='coerce').fillna(0).astype(int)

# No need to rename columns since they're already correct

# Split into reference and current
n = len(df) // 2
os.makedirs('data', exist_ok=True)

df.iloc[:n].to_csv('data/reference_predictions.csv', index=False)
df.iloc[n:].to_csv('data/predictions.csv', index=False)

# Print distributions
print('Updated data/reference_predictions.csv and data/predictions.csv')
print('Prediction distribution (reference):')
print(df.iloc[:n]['prediction'].value_counts())
print('Prediction distribution (current):')
print(df.iloc[n:]['prediction'].value_counts())
