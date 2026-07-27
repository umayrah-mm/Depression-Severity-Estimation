"""
evaluate.py

Loads the best trained MoE fusion model and evaluates it on the held-out
TESTING split - data that was never used for training or for any of our
tuning decisions (LOAD_BALANCE_WEIGHT, EPOCHS, etc.). This is the only
honest, unbiased estimate of real-world performance.

Computes: MAE, RMSE, PCC, CCC
Saves: per-sample predictions + gate weights to CSV

Run:
    python evaluate.py
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

import config
from dataset import load_split_dataset
from models import DepressionPredictionModel, MODALITY_ORDER


def concordance_ccc(y_true, y_pred):
    """
    Concordance Correlation Coefficient - same formula used in your
    existing run_fusion.py, reused here for direct comparability.
    """
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    test_ds = load_split_dataset("Testing")
    print(f"Testing samples: {len(test_ds)}")
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    model = DepressionPredictionModel().to(device)
    model_path = config.MOE_FUSION_BRANCH_DIR / "best_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from: {model_path}")

    rows = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = {name: batch[name].to(device) for name in MODALITY_ORDER}
            y_true = batch["y"]
            y_pred, gate_weights = model(inputs)

            for i in range(len(y_true)):
                row = {
                    "sample_id": batch["video_id"][i],
                    "true_label": y_true[i].item(),
                    "prediction": y_pred[i].item(),
                }
                for j, name in enumerate(MODALITY_ORDER):
                    row[f"{name}_gate"] = gate_weights[i, j].item()
                rows.append(row)

    pred_df = pd.DataFrame(rows)
    output_path = config.MOE_FUSION_BRANCH_DIR / "test_predictions.csv"
    pred_df.to_csv(output_path, index=False)

    metrics = compute_metrics(pred_df["true_label"], pred_df["prediction"])

    print("\n================ TEST SET RESULTS ================")
    print(f"MAE:  {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"PCC:  {metrics['PCC']:.4f}")
    print(f"CCC:  {metrics['CCC']:.4f}")

    print(f"\nAverage gate weights on test set:")
    for name in MODALITY_ORDER:
        avg = pred_df[f"{name}_gate"].mean()
        std = pred_df[f"{name}_gate"].std()
        print(f"  {name}: mean={avg:.4f}, std={std:.4f}")

    print(f"\nSaved predictions to: {output_path}")

    # ---- Honest comparison note ----
    print("\n================ CONTEXT ================")
    print("Your original run_fusion.py (visual + rPPG, hand-tuned late fusion)")
    print("reported test metrics using a 0.98/0.02 visual/rPPG weighting.")
    print("Compare THOSE printed numbers against the MAE/RMSE/PCC/CCC above")
    print("for an honest, apples-to-apples comparison on the same Testing split.")


if __name__ == "__main__":
    main()
    