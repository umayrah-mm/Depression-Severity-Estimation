"""
regenerate_attention_cv_results.py

The 5 fold checkpoints from train_cv_safe_attention.py already exist on
disk (training already happened successfully), but results_attention_cv.txt
was never written - the run was almost certainly interrupted or crashed
during the final Testing-evaluation step, AFTER the expensive part
(training) already finished.

This script skips training entirely and just redoes the fast part:
load the 5 existing fold checkpoints, evaluate them on the untouched
Testing split, ensemble their predictions, and save results in the
exact same format train_cv_safe_attention.py would have produced.

Run:
    python regenerate_attention_cv_results.py
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

import config
from dataset import load_split_dataset
from models_attention import DepressionPredictionModelAttention, MODALITY_ORDER

N_FOLDS = 5
WEIGHT_DECAY = 1e-2  # informational only, matches the original training settings


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

    fold_paths = [
        config.MOE_FUSION_BRANCH_DIR / f"cv_safe_attention_final_fold{i}.pt"
        for i in range(1, N_FOLDS + 1)
    ]

    print("Checking that all 5 fold checkpoints exist...")
    for p in fold_paths:
        status = "FOUND" if p.exists() else "MISSING <-- problem, stop and investigate"
        print(f"  {p.name}: {status}")
    if not all(p.exists() for p in fold_paths):
        raise FileNotFoundError("Not all 5 fold checkpoints were found - cannot proceed.")

    test_ds = load_split_dataset("Testing")
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"\nTesting samples: {len(test_ds)}")

    all_fold_preds = []
    y_true_final = None
    per_fold_metrics = []

    for fold_path in fold_paths:
        model = DepressionPredictionModelAttention().to(device)
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

        single_fold_metrics = compute_metrics(trues, preds)
        per_fold_metrics.append(single_fold_metrics)
        print(f"  {fold_path.name} alone -> MAE={single_fold_metrics['MAE']:.4f}")

    fold_maes = np.array([m["MAE"] for m in per_fold_metrics])
    print(f"\nPer-fold MAE mean={fold_maes.mean():.4f}  std={fold_maes.std():.4f}")

    all_fold_preds = np.array(all_fold_preds)
    ensemble_preds = all_fold_preds.mean(axis=0)
    ensemble_metrics = compute_metrics(y_true_final, ensemble_preds)

    print("\n================ Regenerated attention ensemble results ================")
    for key in ["MAE", "RMSE", "PCC", "CCC"]:
        print(f"{key:<8}{ensemble_metrics[key]:.4f}")

    results_path = config.MOE_FUSION_BRANCH_DIR / "results_attention_cv.txt"
    with open(results_path, "w") as f:
        f.write("Self-attention fusion, 5-fold cross-validated ensemble\n")
        f.write("(regenerated from existing fold checkpoints - training not re-run)\n")
        f.write(f"Settings: dropout=0.6, weight_decay={WEIGHT_DECAY}\n")
        f.write(f"Testing samples: {len(test_ds)}\n")
        f.write(f"Per-fold MAE: mean={fold_maes.mean():.4f}  std={fold_maes.std():.4f}\n\n")
        f.write(f"{'Metric':<8}{'Value':<12}\n")
        for key in ["MAE", "RMSE", "PCC", "CCC"]:
            f.write(f"{key:<8}{ensemble_metrics[key]:<12.4f}\n")
    print(f"\nSaved: {results_path}")
    print("\nSanity check: your project notes say this model previously reached")
    print("approximately MAE 7.33. If the number above is close to that, these")
    print("checkpoints are the same trained models and this result is trustworthy.")
    print("If it's very different (much worse), the checkpoints may be stale or")
    print("incomplete, and we should retrain with train_cv_safe_attention.py instead.")


if __name__ == "__main__":
    main()