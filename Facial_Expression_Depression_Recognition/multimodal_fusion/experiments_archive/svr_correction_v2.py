"""
svr_correction_v2.py

Module B4: same idea as svr_correction.py, but using the RICHER features
from Module B3 (individual per-modality predictions), not just the gate
weights. Also adds a few "interaction features" - differences between
modality predictions - similar in spirit to what the original
LightFusionNet paper's fusion stage used.

Training rules (same as before, to avoid cheating):
    - SVR trained ONLY on train_predictions_with_heads.csv
    - Hyperparameters chosen using dev_predictions_with_heads.csv
    - Isotonic calibration fit on train_predictions_with_heads.csv
    - test_predictions_with_heads.csv used ONLY ONCE, at the end

Run:
    python svr_correction_v2.py
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from scipy.stats import pearsonr

import config

MODALITY_PRED_COLUMNS = ["visual_pred", "rppg_pred", "clip_pred", "smile_pred"]
GATE_COLUMNS = ["visual_gate", "rppg_gate", "clip_gate", "smile_gate"]
BASE_FEATURE_COLUMNS = ["prediction"] + GATE_COLUMNS + MODALITY_PRED_COLUMNS


def concordance_ccc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    try:
        pcc = pearsonr(y_true, y_pred)[0]
    except Exception:
        pcc = 0.0
    ccc = concordance_ccc(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "PCC": pcc, "CCC": ccc}


def add_interaction_features(df):
    """
    Adds pairwise absolute differences between modality predictions -
    e.g. |visual_pred - rppg_pred| - so the correction model can see
    where modalities disagree, similar to the original paper's approach.
    """
    df = df.copy()
    pairs = [
        ("visual_pred", "rppg_pred"),
        ("visual_pred", "clip_pred"),
        ("visual_pred", "smile_pred"),
        ("clip_pred", "smile_pred"),
    ]
    interaction_columns = []
    for a, b in pairs:
        col_name = f"diff_{a}_{b}"
        df[col_name] = (df[a] - df[b]).abs()
        interaction_columns.append(col_name)
    return df, interaction_columns


def load_split(filename):
    path = config.MOE_FUSION_BRANCH_DIR / filename
    df = pd.read_csv(path)
    df, interaction_columns = add_interaction_features(df)
    feature_columns = BASE_FEATURE_COLUMNS + interaction_columns
    X = df[feature_columns].values
    y = df["true_label"].values
    return df, X, y, feature_columns


def main():
    print("Loading CSVs...")
    train_df, X_train, y_train, feature_columns = load_split("train_predictions_with_heads.csv")
    dev_df, X_dev, y_dev, _ = load_split("dev_predictions_with_heads.csv")
    test_df, X_test, y_test, _ = load_split("test_predictions_with_heads.csv")

    print(f"Features used ({len(feature_columns)}): {feature_columns}")
    print(f"Train: {X_train.shape}, Dev: {X_dev.shape}, Test: {X_test.shape}")

    # ---- Step 1: scale features (fit ONLY on train) ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)
    X_test_scaled = scaler.transform(X_test)

    # ---- Step 2: try a few SVR settings, pick the best using Dev ----
    candidate_settings = [
        {"C": 1, "epsilon": 0.1},
        {"C": 5, "epsilon": 0.1},
        {"C": 10, "epsilon": 0.1},
        {"C": 25, "epsilon": 0.01},
        {"C": 50, "epsilon": 0.01},
    ]

    print("\nSearching for the best SVR settings using the Dev split...")
    best_mae = float("inf")
    best_settings = None
    best_model = None

    for settings in candidate_settings:
        svr = SVR(kernel="rbf", C=settings["C"], epsilon=settings["epsilon"])
        svr.fit(X_train_scaled, y_train)
        dev_pred = svr.predict(X_dev_scaled)
        dev_mae = np.mean(np.abs(y_dev - dev_pred))
        print(f"  C={settings['C']}, epsilon={settings['epsilon']}  ->  Dev MAE: {dev_mae:.4f}")

        if dev_mae < best_mae:
            best_mae = dev_mae
            best_settings = settings
            best_model = svr

    print(f"\nBest settings: {best_settings} (Dev MAE: {best_mae:.4f})")

    # ---- Step 3: fit Isotonic calibration on TRAIN predictions only ----
    train_svr_pred = best_model.predict(X_train_scaled)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(train_svr_pred, y_train)

    # ---- Step 4: final, one-time evaluation on TEST ----
    test_svr_pred = best_model.predict(X_test_scaled)
    test_final_pred = calibrator.predict(test_svr_pred)

    original_metrics = compute_metrics(y_test, test_df["prediction"].values)
    corrected_metrics = compute_metrics(y_test, test_final_pred)

    print("\n================ COMPARISON ON TEST SET ================")
    print(f"{'Metric':<8}{'Original (MoE only)':<22}{'Corrected (v2, richer features)':<22}")
    for key in ["MAE", "RMSE", "PCC", "CCC"]:
        print(f"{key:<8}{original_metrics[key]:<22.4f}{corrected_metrics[key]:<22.4f}")

    # ---- Save results for inspection ----
    output_df = test_df.copy()
    output_df["svr_prediction"] = test_svr_pred
    output_df["corrected_prediction"] = test_final_pred
    output_path = config.MOE_FUSION_BRANCH_DIR / "test_predictions_corrected_v2.csv"
    output_df.to_csv(output_path, index=False)
    print(f"\nSaved detailed results to: {output_path}")


if __name__ == "__main__":
    main()