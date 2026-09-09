"""
check_mean_collapse.py

Checks whether the attention model's predictions on AVEC2014 Testing
data show signs of mean-collapse (predictions clustering near the
average regardless of the true label), by comparing the SPREAD (std)
of true labels vs predictions, and checking correlation at the
extremes (does the model predict higher for genuinely severe cases,
and lower for genuinely minimal cases?).
"""

import pandas as pd
import numpy as np

df = pd.read_csv(
    r"C:\Users\HP\Desktop\AVEC2014_processed\outputs\moe_fusion_branch\test_predictions_attention_final.csv"
)

print("=== Spread comparison ===")
print(f"True label  : mean={df['true_label'].mean():.2f}  std={df['true_label'].std():.2f}  "
      f"min={df['true_label'].min():.1f}  max={df['true_label'].max():.1f}")
print(f"Prediction  : mean={df['prediction'].mean():.2f}  std={df['prediction'].std():.2f}  "
      f"min={df['prediction'].min():.1f}  max={df['prediction'].max():.1f}")

ratio = df['prediction'].std() / df['true_label'].std()
print(f"\nPrediction std / True label std = {ratio:.2f}")
print("(Close to 1.0 = healthy spread. Much less than 1.0 = predictions are")
print(" compressed toward the mean, a sign of mean-collapse.)")

print("\n=== Behavior at the extremes ===")
low_true = df[df['true_label'] <= 5]
high_true = df[df['true_label'] >= 30]
print(f"For {len(low_true)} people with true_label <= 5 (very low severity): "
      f"average prediction = {low_true['prediction'].mean():.2f}")
print(f"For {len(high_true)} people with true_label >= 30 (severe): "
      f"average prediction = {high_true['prediction'].mean():.2f}")
print("(If the model is NOT collapsing to the mean, these two numbers should")
print(" be clearly different from each other, tracking the true severity.)")