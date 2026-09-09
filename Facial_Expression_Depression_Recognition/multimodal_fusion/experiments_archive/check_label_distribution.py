"""
check_label_distribution.py

Diagnostic: shows how BDI-II scores are distributed across the 4
clinical severity bands, both overall and per split (Training/
Development/Testing), plus the majority-class baseline per split
(the accuracy you'd get by always guessing the most common band -
useful context for judging whether the model is actually learning
something beyond that).

Uses the SAME alignment logic (align_all()) as the rest of the
pipeline, so counts reflect the actual 297 samples the model trains
and evaluates on - not the raw, unaligned 300-row labels.csv.

Run:
    python check_label_distribution.py
"""

import numpy as np
import pandas as pd

from dataset import align_all


def severity_band(score):
    if score <= 13:
        return "None/Minimal (0-13)"
    elif score <= 19:
        return "Mild (14-19)"
    elif score <= 28:
        return "Moderate (20-28)"
    else:
        return "Severe (29-63)"


def main():
    aligned = align_all()
    y = aligned["y"]
    split = aligned["split"]

    print(f"\nTotal samples (aligned): {len(y)}")
    print(f"Score range: {y.min():.0f} to {y.max():.0f}")
    print(f"Mean score: {y.mean():.2f}")

    df = pd.DataFrame({"score": y, "split": split})
    df["band"] = df["score"].apply(severity_band)

    band_order = ["None/Minimal (0-13)", "Mild (14-19)", "Moderate (20-28)", "Severe (29-63)"]

    print(f"\n=== Overall distribution across severity bands ===")
    for band in band_order:
        count = (df["band"] == band).sum()
        pct = 100 * count / len(df)
        print(f"  {band:<22} {count:>4} samples  ({pct:.1f}%)")

    print(f"\n=== Distribution by split ===")
    print(pd.crosstab(df["split"], df["band"])[band_order])

    print(f"\n=== Majority-class baseline per split ===")
    print("(accuracy if you always guessed the most common band in that split)")
    for split_name in df["split"].unique():
        split_df = df[df["split"] == split_name]
        majority_count = split_df["band"].value_counts().max()
        total = len(split_df)
        baseline = majority_count / total
        print(f"  {split_name}: {baseline:.2%}")


if __name__ == "__main__":
    main()