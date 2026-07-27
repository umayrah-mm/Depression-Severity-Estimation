"""
ensemble_predict.py

Loads the 5 models trained during cross-validation (cv_model_fold1.pt
through cv_model_fold5.pt) and averages their 5 predictions together for
every person in the Testing split. Compares that averaged result against
your original single-model result (MAE 8.15).

Run:
    python ensemble_predict.py
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

import config
from dataset import load_split_dataset
from models import DepressionPredictionModel, MODALITY_ORDER


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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    test_ds = load_split_dataset("Testing")
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"Testing samples: {len(test_ds)}")

    fold_paths = [config.MOE_FUSION_BRANCH_DIR / f"cv_model_fold{i}.pt" for i in range(1, 6)]

    all_fold_preds = []  # will hold 5 lists, one per model
    y_true_final = None

    for fold_path in fold_paths:
        model = DepressionPredictionModel().to(device)
        model.load_state_dict(torch.load(fold_path, map_location=device))
        model.eval()

        preds, trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                inputs = {name: batch[name].to(device) for name in MODALITY_ORDER}
                y_pred, _ = model(inputs)
                preds.extend(y_pred.cpu().numpy().tolist())
                trues.extend(batch["y"].numpy().tolist())

        all_fold_preds.append(preds)
        y_true_final = trues
        print(f"Loaded and ran: {fold_path.name}")

    # ---- Average the 5 models' predictions together, per person ----
    all_fold_preds = np.array(all_fold_preds)  # shape: (5 models, 100 people)
    ensemble_preds = all_fold_preds.mean(axis=0)  # shape: (100 people,)

    ensemble_metrics = compute_metrics(y_true_final, ensemble_preds)
    original = {"MAE": 8.1516, "RMSE": 9.9038, "PCC": 0.5395, "CCC": 0.4944}

    print("\n================ COMPARISON ON TEST SET ================")
    print(f"{'Metric':<8}{'Original (1 model)':<22}{'Ensemble (5 models averaged)':<30}")
    for key in ["MAE", "RMSE", "PCC", "CCC"]:
        print(f"{key:<8}{original[key]:<22.4f}{ensemble_metrics[key]:<30.4f}")


if __name__ == "__main__":
    main()