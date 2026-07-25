import pandas as pd
import numpy as np
import os

csv_path = r"c:\Users\dmishra\Desktop\sperm_project\sperm_results_saturnv2\single_measurements_v12.csv"
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    exit(1)

df = pd.read_csv(csv_path)

print("--- Statistical Summary ---")
print(f"Total Detections: {len(df)}")

metrics = {
    'Length (um)': 'length_um_geodesic',
    'Width (um)': 'width_um',
    'Tortuosity': 'tortuosity',
    'Area (px)': 'area_px'
}

for label, col in metrics.items():
    if col in df.columns:
        vals = pd.to_numeric(df[col], errors='coerce').dropna()
        print(f"\n{label}:")
        print(f"  Mean:   {vals.mean():.2f}")
        print(f"  Median: {vals.median():.2f}")
        print(f"  SD:     {vals.std():.2f}")
        print(f"  Max:    {vals.max():.2f}")
        print(f"  Min:    {vals.min():.2f}")

# Identify outliers (Potential Mergers)
if 'length_um_geodesic' in df.columns:
    q3 = df['length_um_geodesic'].quantile(0.75)
    iqr = q3 - df['length_um_geodesic'].quantile(0.25)
    threshold = q3 + 3.0 * iqr  # Strong outliers
    outliers = df[df['length_um_geodesic'] > threshold]
    print(f"\nPotential Mergers (Length > {threshold:.2f} um): {len(outliers)}")
    if len(outliers) > 0:
        print(outliers[['sperm_id', 'length_um_geodesic', 'tortuosity']].head(10).to_string(index=False))
