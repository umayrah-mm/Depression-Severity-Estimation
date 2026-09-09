"""
diagnose_pattern.py

Follow-up to find_hard_subgroup.py. Tests the "regression to the mean /
range compression" hypothesis: does the model compress its predictions
toward the middle of the BDI-II scale, so that people with truly extreme
scores (very low or very high) get pulled toward the center and therefore
show the largest errors?
"""

import pandas as pd
import numpy as np

person = pd.read_csv("hard_subgroup_analysis.csv")

print("=== Overall prediction range vs true label range ===")
print(f"True label:  min={person['mean_true_label'].min():.1f}, "
      f"max={person['mean_true_label'].max():.1f}, "
      f"mean={person['mean_true_label'].mean():.2f}, "
      f"std={person['mean_true_label'].std():.2f}")
print(f"Prediction:  min={person['mean_prediction'].min():.1f}, "
      f"max={person['mean_prediction'].max():.1f}, "
      f"mean={person['mean_prediction'].mean():.2f}, "
      f"std={person['mean_prediction'].std():.2f}")

# distance of each person's TRUE label from the overall true-label mean
overall_mean_true = person["mean_true_label"].mean()
person["true_dist_from_mean"] = (person["mean_true_label"] - overall_mean_true).abs()

# correlation between how extreme someone's true score is, and their error
corr = person["true_dist_from_mean"].corr(person["mean_abs_error"])
print(f"\nCorrelation between |true_label - overall_mean| and abs_error: {corr:.3f}")
print("(closer to +1.0 = strong evidence the model struggles most on extreme scores)")

# split into thirds by true label to show the trend directly
person_sorted = person.sort_values("mean_true_label").reset_index(drop=True)
n = len(person_sorted)
low_third = person_sorted.iloc[: n // 3]
mid_third = person_sorted.iloc[n // 3 : 2 * n // 3]
high_third = person_sorted.iloc[2 * n // 3 :]

print("\n=== Mean abs error by true-label tertile ===")
for name, grp in [("Low true scores", low_third), ("Mid true scores", mid_third), ("High true scores", high_third)]:
    print(f"{name:18s} (n={len(grp):2d}, true label range {grp['mean_true_label'].min():.0f}-"
          f"{grp['mean_true_label'].max():.0f}): mean abs error = {grp['mean_abs_error'].mean():.2f}, "
          f"mean signed error = {grp['mean_signed_error'].mean():+.2f}")