"""
combine_ensembles.py

Loads both sets of already-trained models:
    - MoE ensemble: cv_safe_fold1.pt ... cv_safe_fold5.pt
    - Attention ensemble: cv_safe_attention_final_fold1.pt ... fold5.pt

Runs all 10 models on the real Testing split, gets each ensemble's
averaged prediction per person, checks how much the two ensembles
actually disagree, and tests whether averaging THEM together helps,
hurts, or makes no real difference versus using the attention ensemble
alone (currently the best single result, MAE 7.33).

No new training happens here - this only reuses models you already
have saved, so it's quick (a couple of minutes, not 15-20).

Run:
    python combine_ensembles.py
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

import config
from dataset import load_split_dataset
import models as moe_models
import models_attention as att_models


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


def run_ensemble(model_class, modality_order, fold_paths, test_loader, device):
    """Runs all fold models in one ensemble, returns (avg_preds, y_true)."""
    all_fold_preds = []
    y_true_final = None

    for fold_path in fold_paths:
        model = model_class().to(device)
        model.load_state_dict(torch.load(fold_path, map_location=device))
        model.eval()

        preds, trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                inputs = {name: batch[name].to(device) for name in modality_order}
                y_pred, _ = model(inputs)
                preds.extend(y_pred.cpu().numpy().tolist())
                trues.extend(batch["y"].numpy().tolist())

        all_fold_preds.append(preds)
        y_true_final = trues

    all_fold_preds = np.array(all_fold_preds)  # (5, num_people)
    avg_preds = all_fold_preds.mean(axis=0)
    return avg_preds, y_true_final


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    test_ds = load_split_dataset("Testing")
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"Testing samples: {len(test_ds)}")

    # ---- MoE ensemble ----
    moe_fold_paths = [config.MOE_FUSION_BRANCH_DIR / f"cv_safe_fold{i}.pt" for i in range(1, 6)]
    moe_preds, y_true = run_ensemble(
        moe_models.DepressionPredictionModel, moe_models.MODALITY_ORDER, moe_fold_paths, test_loader, device
    )
    moe_metrics = compute_metrics(y_true, moe_preds)
    print(f"\nMoE ensemble alone       -> MAE={moe_metrics['MAE']:.4f}")

    # ---- Attention ensemble ----
    att_fold_paths = [config.MOE_FUSION_BRANCH_DIR / f"cv_safe_attention_final_fold{i}.pt" for i in range(1, 6)]
    att_preds, _ = run_ensemble(
        att_models.DepressionPredictionModelAttention, att_models.MODALITY_ORDER, att_fold_paths, test_loader, device
    )
    att_metrics = compute_metrics(y_true, att_preds)
    print(f"Attention ensemble alone -> MAE={att_metrics['MAE']:.4f}")

    # ---- How much do the two ensembles actually disagree? ----
    disagreement = np.abs(moe_preds - att_preds)
    agreement_corr = pearsonr(moe_preds, att_preds)[0]
    print(f"\nAgreement between the two ensembles' predictions:")
    print(f"  Correlation between their predictions: {agreement_corr:.4f}  "
          f"(closer to 1.0 = very similar guesses, less room for combining to help)")
    print(f"  Average |difference| between their guesses per person: {disagreement.mean():.4f}")
    print(f"  Max difference on any single person: {disagreement.max():.4f}")

    # ---- Combined average of both ensembles ----
    combined_preds = (moe_preds + att_preds) / 2.0
    combined_metrics = compute_metrics(y_true, combined_preds)

    print("\n================ FINAL, HONEST COMPARISON ================")
    print(f"{'Metric':<8}{'MoE ensemble':<16}{'Attention ensemble':<20}{'Combined (both)':<18}")
    for key in ["MAE", "RMSE", "PCC", "CCC"]:
        print(f"{key:<8}{moe_metrics[key]:<16.4f}{att_metrics[key]:<20.4f}{combined_metrics[key]:<18.4f}")


if __name__ == "__main__":
    main()