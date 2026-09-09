"""
cross_model_check.py

Tests the report's central claim: "across every configuration and
regularisation setting tested, one subgroup of ~12 people consistently
produced higher error." We now have real prediction files from 7 different
Testing-set model configurations, all evaluated on the SAME 100 rows
(50 people x 2 tasks: Freeform + Northwind). For each configuration we
compute per-row absolute error, join everything on the exact sample_id
string (so we never risk mismatching one model's row with another
person's), then aggregate to the person level and rank.
"""

import pandas as pd
import config  # your project's config.py - defines DATA_ROOT / OUTPUTS_ROOT etc. for YOUR machine

# ---- Step 1: load every Testing-set configuration and standardize columns ----
configs = {}

df = pd.read_csv("test_predictions_attention_final.csv")
configs["attention_only"] = df.rename(columns={"sample_id": "id"})[["id", "true_label", "prediction"]]

df = pd.read_csv(config.MOE_FUSION_BRANCH_DIR / "attention_moe_test_predictions.csv")
configs["attention_moe"] = df.rename(columns={"video_id": "id"})[["id", "true_label", "prediction"]]

df = pd.read_csv(config.MOE_FUSION_BRANCH_DIR / "test_predictions.csv")
configs["moe_only"] = df.rename(columns={"sample_id": "id"})[["id", "true_label", "prediction"]]

df = pd.read_csv(config.MOE_FUSION_BRANCH_DIR / "test_predictions_corrected_v2.csv")
configs["moe_svr_corrected"] = df.rename(columns={"sample_id": "id"})[["id", "true_label", "corrected_prediction"]].rename(
    columns={"corrected_prediction": "prediction"}
)

df = pd.read_csv(config.FUSION_BRANCH_DIR / "fusion_testing_predictions.csv")
configs["baseline_single_split"] = df.rename(
    columns={"video_id": "id", "y_true": "true_label", "fusion_pred_98_2": "prediction"}
)[["id", "true_label", "prediction"]]

df = pd.read_csv(config.BASELINE_CV_DIR / "baseline_cv_predictions.csv")
configs["baseline_5fold_cv"] = df.rename(
    columns={"video_id": "id", "BDI_II": "true_label", "fusion_98_2": "prediction"}
)[["id", "true_label", "prediction"]]

# ---- Step 2: compute abs error per row, per config ----
for name, d in configs.items():
    d["abs_error"] = (d["true_label"] - d["prediction"]).abs()

# ---- Step 3: alignment check across configs - true_label must match for the
# same id everywhere. If it doesn't, something is badly wrong (e.g. two
# different models' rows got mismatched to different people).
base_labels = configs["attention_only"].set_index("id")["true_label"]
for name, d in configs.items():
    merged = d.set_index("id")["true_label"]
    diff = (merged - base_labels).abs()
    bad = diff[diff > 0.01]
    if len(bad) > 0:
        print(f"WARNING: {name} has {len(bad)} ids with a DIFFERENT true_label than attention_only!")
        print(bad)
print("Cross-config true_label alignment check: OK if no WARNING printed above.\n")

# ---- Step 4: build one wide table: id -> abs_error in each config ----
wide = None
for name, d in configs.items():
    col = d.set_index("id")["abs_error"].rename(name)
    wide = col.to_frame() if wide is None else wide.join(col, how="outer")

wide = wide.reset_index()
parts = wide["id"].str.replace("_video", "", regex=False).str.split("_", expand=True)
wide["subject"] = parts[0].astype(int)

# ---- Step 5: aggregate to person level (average across sessions/tasks) for each config ----
person_wide = wide.groupby("subject")[list(configs.keys())].mean()

# ---- Step 6: our previously identified hard 12 (from the attention-only model) ----
HARD_12 = [317, 328, 249, 325, 206, 211, 246, 346, 237, 220, 203, 359]

# for each config, rank people worst-to-best, and see what rank our hard-12 get
print("=== Mean abs error per config, HARD-12 vs REST ===")
for name in configs.keys():
    hard_mean = person_wide.loc[HARD_12, name].mean()
    rest_mean = person_wide.loc[~person_wide.index.isin(HARD_12), name].mean()
    print(f"{name:24s}  hard-12 mean = {hard_mean:6.2f}   rest mean = {rest_mean:5.2f}   ratio = {hard_mean/rest_mean:.2f}x")

print("\n=== For each config, what RANK (1=worst) does each of the hard-12 get out of 42 people? ===")
rank_table = person_wide.rank(ascending=False, method="min").astype(int)
rank_table_hard = rank_table.loc[HARD_12]
rank_table_hard["mean_rank"] = rank_table_hard.mean(axis=1).round(1)
rank_table_hard = rank_table_hard.sort_values("mean_rank")
pd.set_option("display.width", 160)
print(rank_table_hard.to_string())

n_people = len(person_wide)
print(f"\n(out of {n_people} total people per config; rank 1 = highest error in that config)")

person_wide.to_csv("cross_model_person_errors.csv")
print("\nSaved full per-person, per-config error table to cross_model_person_errors.csv")