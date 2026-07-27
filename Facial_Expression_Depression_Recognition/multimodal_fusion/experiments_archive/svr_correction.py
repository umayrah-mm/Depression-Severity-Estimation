"""
svr_correction.py

Module A2: trains a small SVR "correction" model on top of the MoE
fusion model's outputs, to see if a calibrated second-stage regressor
(like the one used in the original LightFusionNet paper) improves MAE.

Input features per sample (5 numbers):
    prediction, visual_gate, rppg_gate, clip_gate, smile_gate

Output: 1 corrected BDI-II severity score per sample.

Training rules (to avoid cheating / data leakage):
    - The SVR is trained ONLY on train_predictions.csv
    - Hyperparameters (C, epsilon) are chosen using dev_predictions.csv
    - Isotonic calibration is fit on train_predictions.csv
    - test_predictions.csv is used ONLY ONCE, at the very end, to report
      the final honest result.

Run:
    python svr_correction.py
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from scipy.stats import pearsonr

import config


FEATURE_COLUMNS = ["prediction", "visual_gate", "rppg_gate", "clip_gate", "smile_gate"]


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


def load_split(filename):
    path = config.MOE_FUSION_BRANCH_DIR / filename
    df = pd.read_csv(path)
    X = df[FEATURE_COLUMNS].values          # shape: (num_samples, 5)
    y = df["true_label"].values             # shape: (num_samples,)
    return df, X, y


def main():
    print("Loading CSVs...")
    train_df, X_train, y_train = load_split("train_predictions.csv")
    dev_df, X_dev, y_dev = load_split("dev_predictions.csv")
    test_df, X_test, y_test = load_split("test_predictions.csv")

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
    print(f"{'Metric':<8}{'Original (MoE only)':<22}{'Corrected (SVR+Calib)':<22}")
    for key in ["MAE", "RMSE", "PCC", "CCC"]:
        print(f"{key:<8}{original_metrics[key]:<22.4f}{corrected_metrics[key]:<22.4f}")

    # ---- Save results for inspection ----
    output_df = test_df.copy()
    output_df["svr_prediction"] = test_svr_pred
    output_df["corrected_prediction"] = test_final_pred
    output_path = config.MOE_FUSION_BRANCH_DIR / "test_predictions_corrected.csv"
    output_df.to_csv(output_path, index=False)
    print(f"\nSaved detailed results to: {output_path}")


if __name__ == "__main__":
    main()