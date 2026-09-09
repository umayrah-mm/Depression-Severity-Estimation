"""
check_rppg_quality.py

For the 6 people who are hard across EVERY configuration (universal core),
check whether their rPPG signal quality (Q, from rppg_quality.csv) is
unusually low compared to the rest of the Testing set - i.e. whether their
physiological signal was borderline-noisy even though it technically passed
the paper's Q > 2.0 inclusion threshold.
"""

import pandas as pd
import config  # your project's config.py - defines DATA_ROOT etc. for YOUR machine

RPPG_QUALITY_CSV = config.DATA_ROOT / "rppg_quality.csv"

quality = pd.read_csv(RPPG_QUALITY_CSV)
quality = quality[quality["split"] == "Testing"].copy()

parts = quality["video_id"].str.replace("_video", "", regex=False).str.split("_", expand=True)
quality["subject"] = parts[0].astype(int)

person_quality = quality.groupby("subject")["rppg_quality"].mean().reset_index()

UNIVERSAL_HARD = [317, 325, 246, 206, 237, 359]
MODEL_SPECIFIC_HARD = [249, 328, 346, 211, 203, 220]

person_quality["group"] = "rest"
person_quality.loc[person_quality["subject"].isin(UNIVERSAL_HARD), "group"] = "universal_hard"
person_quality.loc[person_quality["subject"].isin(MODEL_SPECIFIC_HARD), "group"] = "model_specific_hard"

print("=== rPPG quality (Q) by group, Testing split ===")
print(person_quality.groupby("group")["rppg_quality"].agg(["count", "mean", "median", "min", "max"]).to_string())

print("\n=== Individual Q values ===")
print(person_quality.sort_values("group").to_string(index=False))

print(f"\nOverall Testing-set Q: mean={quality['rppg_quality'].mean():.2f}, "
      f"median={quality['rppg_quality'].median():.2f}, "
      f"paper's exclusion threshold = 2.0")

n_low = (person_quality["rppg_quality"] < 3.0).sum()
print(f"\nPeople with Q < 3.0 (well above exclusion threshold but still weak signal): {n_low} of {len(person_quality)}")