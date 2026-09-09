"""
find_hard_subgroup.py

Purpose: identify the "persistently hard subgroup" of ~12 people mentioned
in the report (section 10.4), using the attention model's held-out Testing
predictions (test_predictions_attention_final.csv).

Input:  test_predictions_attention_final.csv
        columns: sample_id (e.g. "220_1_Freeform_video"), true_label, prediction
Output: hard_subgroup_analysis.csv (per-person breakdown, sorted worst-first)
        printed summary comparing the hard subgroup to everyone else
"""

import pandas as pd

# ---- Step 1: load the raw per-row predictions ----
df = pd.read_csv("test_predictions_attention_final.csv")
print(f"Loaded {len(df)} rows (should be 100 = 50 people x 2 tasks)\n")

# ---- Step 2: parse sample_id into subject / session / task ----
# sample_id format: "{subject}_{session}_{task}_video"
# e.g. "220_1_Freeform_video" -> subject=220, session=1, task=Freeform
parts = df["sample_id"].str.replace("_video", "", regex=False).str.split("_", expand=True)
df["subject"] = parts[0].astype(int)
df["session"] = parts[1].astype(int)
df["task"] = parts[2]

# ---- Step 3: per-row absolute error ----
df["abs_error"] = (df["true_label"] - df["prediction"]).abs()
df["signed_error"] = df["prediction"] - df["true_label"]  # positive = over-predicted

# ---- Step 4: sanity check - same subject+session should have the SAME true_label
# across both tasks (Freeform/Northwind), since the label is per recording session,
# not per task. If it doesn't match, that's an alignment bug worth flagging.
check = df.groupby(["subject", "session"])["true_label"].nunique()
mismatches = check[check > 1]
if len(mismatches) > 0:
    print("WARNING: inconsistent true_label within a subject+session:")
    print(mismatches)
else:
    print("Alignment check passed: true_label is consistent across Freeform/Northwind "
          "for every subject+session (no label mismatch).\n")

# ---- Step 5: aggregate up to the PERSON level ----
# Average across session(s) and task(s) so someone with 2 sessions and 2 tasks
# (4 rows) is still counted as ONE person, matching the report's "12 people" framing.
person = df.groupby("subject").agg(
    n_rows=("abs_error", "count"),
    mean_true_label=("true_label", "mean"),
    mean_prediction=("prediction", "mean"),
    mean_abs_error=("abs_error", "mean"),
    mean_signed_error=("signed_error", "mean"),
).reset_index()

person = person.sort_values("mean_abs_error", ascending=False).reset_index(drop=True)

# ---- Step 6: define the hard subgroup as the worst 12 people by mean_abs_error ----
N_HARD = 12
hard = person.iloc[:N_HARD].copy()
rest = person.iloc[N_HARD:].copy()

hard["group"] = "hard_subgroup"
rest["group"] = "rest"
person_labeled = pd.concat([hard, rest], ignore_index=True)
person_labeled.to_csv("hard_subgroup_analysis.csv", index=False)

# ---- Step 7: print a readable summary ----
pd.set_option("display.width", 120)
pd.set_option("display.float_format", lambda x: f"{x:.2f}")

print(f"=== Top {N_HARD} hardest people (by mean absolute error) ===")
print(hard[["subject", "n_rows", "mean_true_label", "mean_prediction",
            "mean_abs_error", "mean_signed_error"]].to_string(index=False))

print(f"\n=== Summary comparison ===")
print(f"Hard subgroup (n={len(hard)}): mean abs error = {hard['mean_abs_error'].mean():.2f}, "
      f"mean true label = {hard['mean_true_label'].mean():.2f}, "
      f"mean signed error = {hard['mean_signed_error'].mean():.2f}")
print(f"Rest (n={len(rest)}):          mean abs error = {rest['mean_abs_error'].mean():.2f}, "
      f"mean true label = {rest['mean_true_label'].mean():.2f}, "
      f"mean signed error = {rest['mean_signed_error'].mean():.2f}")

print(f"\nOverall test-set mean abs error (all {len(person)} people): "
      f"{person['mean_abs_error'].mean():.2f}")

# ---- Step 8: over- vs under-prediction split within the hard subgroup ----
over = (hard["mean_signed_error"] > 0).sum()
under = (hard["mean_signed_error"] < 0).sum()
print(f"\nWithin hard subgroup: {over} over-predicted, {under} under-predicted")

# ---- Step 9: where do hard subgroup true labels sit on the BDI-II scale? ----
# 0-13 None/Minimal, 14-19 Mild, 20-28 Moderate, 29-63 Severe
def severity_band(score):
    if score <= 13:
        return "None/Minimal (0-13)"
    elif score <= 19:
        return "Mild (14-19)"
    elif score <= 28:
        return "Moderate (20-28)"
    else:
        return "Severe (29-63)"

hard["severity_band"] = hard["mean_true_label"].round().apply(severity_band)
rest["severity_band"] = rest["mean_true_label"].round().apply(severity_band)

print("\nHard subgroup severity band distribution:")
print(hard["severity_band"].value_counts().to_string())
print("\nRest-of-testset severity band distribution:")
print(rest["severity_band"].value_counts().to_string())

print("\nSaved full per-person breakdown to hard_subgroup_analysis.csv")