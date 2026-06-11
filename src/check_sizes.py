import pandas as pd
import os

df = pd.read_csv('outputs/reports/results.csv')
top_10 = df.head(10)[['feature_set', 'model', 'f1_macro']]

print('Top 10 Models and their file sizes:')
for _, row in top_10.iterrows():
    fset = row['feature_set']
    model = row['model']
    fname = f"{fset}__{model}.joblib"
    path = os.path.join('outputs', 'models', fname)
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"- {fname}: {size_mb:.2f} MB (F1: {row['f1_macro']:.4f})")
    else:
        print(f"- {fname}: NOT FOUND")
