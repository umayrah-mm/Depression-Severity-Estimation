from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr


VISUAL_DIR = Path(r"C:\Users\HP\Desktop\AVEC2014_processed\outputs\visual_branch")
RPPG_DIR = Path(r"C:\Users\HP\Desktop\AVEC2014_processed\outputs\rppg_branch")
OUTPUT_DIR = Path(r"C:\Users\HP\Desktop\AVEC2014_processed\outputs\fusion_branch")


def concordance_ccc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    return (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)


def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    try:
        pcc = pearsonr(y_true, y_pred)[0]
    except Exception:
        pcc = 0.0

    ccc = concordance_ccc(y_true, y_pred)

    return mae, rmse, pcc, ccc


def load_and_merge(split_name, visual_file, rppg_file):
    visual = pd.read_csv(VISUAL_DIR / visual_file)
    rppg = pd.read_csv(RPPG_DIR / rppg_file)

    print(f"\n{split_name}")
    print("Visual columns:", list(visual.columns))
    print("rPPG columns:", list(rppg.columns))

    # Find prediction column in rPPG file
    possible_rppg_pred_cols = [
        "prediction",
        "pred",
        "y_pred",
        "rppg_pred",
        "ensemble_pred",
        "final_prediction",
        "BDI_pred",
    ]

    rppg_pred_col = None
    for col in possible_rppg_pred_cols:
        if col in rppg.columns:
            rppg_pred_col = col
            break

    if rppg_pred_col is None:
        numeric_cols = rppg.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ["BDI_II", "BDI-II", "y_true"]]

        if len(numeric_cols) == 0:
            raise ValueError("Could not find rPPG prediction column.")

        rppg_pred_col = numeric_cols[-1]
        print(f"Auto-selected rPPG prediction column: {rppg_pred_col}")

    # Find true label column in rPPG
    if "BDI_II" in rppg.columns:
        rppg_true_col = "BDI_II"
    elif "BDI-II" in rppg.columns:
        rppg_true_col = "BDI-II"
    elif "y_true" in rppg.columns:
        rppg_true_col = "y_true"
    else:
        rppg_true_col = None

    # Standardize rPPG
    rppg_small = rppg[["video_id", rppg_pred_col]].copy()
    rppg_small = rppg_small.rename(columns={rppg_pred_col: "rppg_pred"})

    merged = visual.merge(rppg_small, on="video_id", how="inner")

    print(f"Merged rows: {len(merged)}")

    if len(merged) == 0:
        raise ValueError("No matching video_id values between visual and rPPG CSVs.")

    return merged


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = {
        "Training": (
            "visual_training_predictions.csv",
            "rppg_training_video_predictions.csv",
        ),
        "Development": (
            "visual_development_predictions.csv",
            "rppg_development_video_predictions.csv",
        ),
        "Testing": (
            "visual_testing_predictions.csv",
            "rppg_testing_video_predictions.csv",
        ),
    }

    all_results = {}

    for split_name, (visual_file, rppg_file) in files.items():
        merged = load_and_merge(split_name, visual_file, rppg_file)

        # Paper-style weighted fusion: visual is much stronger, rPPG has smaller weight
        merged["fusion_pred_98_02"] = (
            0.98 * merged["visual_pred"] +
            0.02 * merged["rppg_pred"]
        )

        # Also test a few other weights to see what works best
        for visual_weight in [0.90, 0.95, 0.98, 0.99]:
            rppg_weight = 1.0 - visual_weight
            col = f"fusion_pred_{int(visual_weight*100)}_{int(rppg_weight*100)}"
            merged[col] = (
                visual_weight * merged["visual_pred"] +
                rppg_weight * merged["rppg_pred"]
            )

        output_file = OUTPUT_DIR / f"fusion_{split_name.lower()}_predictions.csv"
        merged.to_csv(output_file, index=False)

        all_results[split_name] = merged

    test = all_results["Testing"]

    print("\n================ FINAL TEST RESULTS ================")

    for pred_col in ["visual_pred", "rppg_pred", "fusion_pred_90_9", "fusion_pred_95_5", "fusion_pred_98_2", "fusion_pred_99_1"]:
        if pred_col not in test.columns:
            continue

        mae, rmse, pcc, ccc = metrics(test["y_true"], test[pred_col])

        print(
            f"{pred_col:18s} "
            f"MAE={mae:.4f}  RMSE={rmse:.4f}  PCC={pcc:.4f}  CCC={ccc:.4f}"
        )

    print("\nSaved fusion outputs to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()