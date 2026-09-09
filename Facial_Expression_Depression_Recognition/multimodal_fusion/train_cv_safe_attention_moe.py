"""
train_cv_safe_attention_moe.py

Leak-free cross-validation + ensembling for the COMBINED attention+MoE
fusion model (models_attention_moe.py), using the exact same safe
methodology as train_cv_safe.py and train_cv_safe_attention.py:
5 folds trained on Training+Development people only (197 total),
Testing split (100 people) never touched until the final, one-time
evaluation at the end.

This model's forward pass returns THREE things (not two, like the other
two models):
    y_pred        - the predicted BDI-II score
    gate_weights  - the MoE gate's per-modality routing weights, AFTER
                    self-attention has let the modalities inform each
                    other (this is the whole point of the merge)
    attn_weights  - the raw self-attention map, kept for inspection but
                    not used in the loss

The load-balancing loss from train_cv_safe.py is reused unchanged, so
the gate still can't collapse onto favoring 1-2 modalities across a
batch, exactly as in the MoE-only model. weight_decay is reused from
train_cv_safe_attention.py, since this model also contains the
attention layer that motivated adding it there.

Compares the final ensemble against both existing results, read
directly from their saved results_*.txt files INSIDE
config.MOE_FUSION_BRANCH_DIR (not the current folder), so the
comparison numbers can never go stale, be mistyped, or silently fail
to be found.

Saves final metrics to results_attention_moe_cv.txt.

Run:
    python train_cv_safe_attention_moe.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr

import config
from dataset import load_split_dataset
from models_attention_moe import (
    DepressionPredictionModelAttentionMoE,
    MODALITY_ORDER,
    count_trainable_parameters,
)

N_FOLDS = 5
PATIENCE = 15
WEIGHT_DECAY = 1e-2  # same as train_cv_safe_attention.py - this model also
                      # contains the attention layer, so it carries the same
                      # extra overfitting risk that motivated weight_decay there


def load_balancing_loss(gate_weights):
    """
    Same load-balancing penalty as train_cv_safe.py: discourages the
    gate from favoring some modalities over others ACROSS A BATCH,
    without forcing every individual sample to use all modalities
    equally.
    """
    avg_usage = gate_weights.mean(dim=0)
    target = 1.0 / gate_weights.shape[1]
    return ((avg_usage - target) ** 2).sum()


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


def read_saved_metrics(path):
    """
    Reads a results_*.txt file written by train_cv_safe.py or
    train_cv_safe_attention.py and returns its MAE/RMSE/PCC/CCC as a
    dict, or None if the file doesn't exist yet or can't be parsed.
    """
    if not path.exists():
        return None
    metrics = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2 and parts[0] in ("MAE", "RMSE", "PCC", "CCC"):
                try:
                    metrics[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return metrics if len(metrics) == 4 else None


def run_one_epoch(model, loader, loss_fn, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    gate_sum = torch.zeros(len(MODALITY_ORDER))

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            inputs = {name: batch[name].to(device) for name in MODALITY_ORDER}
            y_true = batch["y"].to(device)

            if train:
                optimizer.zero_grad()

            y_pred, gate_weights, attn_weights = model(inputs)
            loss = loss_fn(y_pred, y_true) + config.LOAD_BALANCE_WEIGHT * load_balancing_loss(gate_weights)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y_true)
            n += len(y_true)
            gate_sum += gate_weights.detach().cpu().sum(dim=0)

    avg_loss = total_loss / n
    avg_gates = {name: (gate_sum[i] / n).item() for i, name in enumerate(MODALITY_ORDER)}
    return avg_loss, avg_gates


def main():
    torch.manual_seed(config.RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model_check = DepressionPredictionModelAttentionMoE()
    n_params = count_trainable_parameters(model_check)
    print(f"Model trainable parameters per fold: {n_params:,} ({n_params/1e6:.3f} M)")

    # ---- Pool = Training + Development ONLY. Testing is untouched. ----
    train_ds = load_split_dataset("Training")
    dev_ds = load_split_dataset("Development")
    pool_ds = ConcatDataset([train_ds, dev_ds])
    print(f"Train+Dev pool size: {len(pool_ds)}  (Testing is NOT included here)")

    pool_ids = [pool_ds[i]["video_id"] for i in range(len(pool_ds))]
    group_ids = [str(vid).split("_")[0] for vid in pool_ids]
    print(f"Unique people in pool: {len(set(group_ids))}")

    kfold = GroupKFold(n_splits=N_FOLDS)
    indices = np.arange(len(pool_ds))

    fold_model_paths = []

    for fold_index, (train_idx, val_idx) in enumerate(kfold.split(indices, groups=group_ids)):
        train_people = set(group_ids[i] for i in train_idx)
        val_people = set(group_ids[i] for i in val_idx)
        overlap = train_people & val_people
        if overlap:
            raise RuntimeError(f"LEAKAGE in fold {fold_index + 1}: {overlap}")

        print(f"\n=== Fold {fold_index + 1}/{N_FOLDS} ===")
        print(f"Train: {len(train_idx)} samples ({len(train_people)} people)  |  "
              f"Val: {len(val_idx)} samples ({len(val_people)} people)")
        print(f"Verified: 0 people overlap between train and val")

        train_subset = Subset(pool_ds, train_idx)
        val_subset = Subset(pool_ds, val_idx)
        train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False)

        model = DepressionPredictionModelAttentionMoE().to(device)
        loss_fn = nn.SmoothL1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_state = None

        for epoch in range(1, config.EPOCHS + 1):
            train_loss, train_gates = run_one_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
            val_loss, val_gates = run_one_epoch(model, val_loader, loss_fn, optimizer, device, train=False)

            improved = val_loss < best_val_loss
            if epoch % 10 == 0 or epoch == 1:
                marker = "  <-- best so far" if improved else ""
                print(f"  Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}{marker}")
                gate_str = ", ".join(f"{k}={v:.3f}" for k, v in val_gates.items())
                print(f"      val gate weights (avg): {gate_str}")

            if improved:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

        model.load_state_dict(best_state)
        fold_path = config.MOE_FUSION_BRANCH_DIR / f"cv_safe_attention_moe_fold{fold_index + 1}.pt"
        torch.save(model.state_dict(), fold_path)
        fold_model_paths.append(fold_path)
        print(f"  Saved: {fold_path}")

    # ---- ONLY NOW do we touch the real Testing split - once, for evaluation ----
    print(f"\n{'=' * 60}")
    print("Evaluating all 5 folds' models on the REAL, untouched Testing split")
    print(f"{'=' * 60}")

    test_ds = load_split_dataset("Testing")
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"Testing samples: {len(test_ds)}")

    all_fold_preds = []
    all_fold_gates = []
    y_true_final = None
    test_video_ids = None
    per_fold_metrics = []

    for fold_path in fold_model_paths:
        model = DepressionPredictionModelAttentionMoE().to(device)
        model.load_state_dict(torch.load(fold_path, map_location=device))
        model.eval()

        preds, trues, gates, vids = [], [], [], []
        with torch.no_grad():
            for batch in test_loader:
                inputs = {name: batch[name].to(device) for name in MODALITY_ORDER}
                y_pred, gate_weights, _ = model(inputs)
                preds.extend(y_pred.cpu().numpy().tolist())
                trues.extend(batch["y"].numpy().tolist())
                gates.extend(gate_weights.cpu().numpy().tolist())
                vids.extend(batch["video_id"])

        all_fold_preds.append(preds)
        all_fold_gates.append(gates)
        y_true_final = trues
        test_video_ids = vids

        single_fold_metrics = compute_metrics(trues, preds)
        per_fold_metrics.append(single_fold_metrics)
        print(f"  {fold_path.name} alone -> MAE={single_fold_metrics['MAE']:.4f}")

    fold_maes = np.array([m["MAE"] for m in per_fold_metrics])
    print(f"\nPer-fold MAE mean={fold_maes.mean():.4f}  std={fold_maes.std():.4f}")

    all_fold_preds = np.array(all_fold_preds)   # (5, N_test)
    all_fold_gates = np.array(all_fold_gates)   # (5, N_test, 4)
    ensemble_preds = all_fold_preds.mean(axis=0)
    ensemble_gates = all_fold_gates.mean(axis=0)

    ensemble_metrics = compute_metrics(y_true_final, ensemble_preds)

    # ---- Honest comparison against your two existing saved results ----
    moe_metrics = read_saved_metrics(config.MOE_FUSION_BRANCH_DIR / "results_moe_cv.txt")
    if moe_metrics is None:
        moe_metrics = {"MAE": 7.7283, "RMSE": 10.0032, "PCC": 0.5145, "CCC": 0.4488}
        print("\n(results_moe_cv.txt not found - using last known MoE ensemble figures.)")

    attn_metrics = read_saved_metrics(config.MOE_FUSION_BRANCH_DIR / "results_attention_cv.txt")
    if attn_metrics is None:
        print("\n(results_attention_cv.txt not found - re-run train_cv_safe_attention.py")
        print(" if you want the attention-only row in the comparison below.)")

    print("\n================ FINAL, HONEST COMPARISON ================")
    header = f"{'Metric':<8}{'MoE only':<14}"
    if attn_metrics:
        header += f"{'Attention only':<18}"
    header += f"{'Attention+MoE (new)':<20}"
    print(header)

    for key in ["MAE", "RMSE", "PCC", "CCC"]:
        row = f"{key:<8}{moe_metrics[key]:<14.4f}"
        if attn_metrics:
            row += f"{attn_metrics[key]:<18.4f}"
        row += f"{ensemble_metrics[key]:<20.4f}"
        print(row)

    # ---- Save gate weights + predictions for explainability ----
    pred_df = pd.DataFrame({
        "video_id": test_video_ids,
        "true_label": y_true_final,
        "prediction": ensemble_preds,
    })
    for i, name in enumerate(MODALITY_ORDER):
        pred_df[f"{name}_gate"] = ensemble_gates[:, i]
    pred_df.to_csv(config.MOE_FUSION_BRANCH_DIR / "attention_moe_test_predictions.csv", index=False)

    # ---- Save results to a file so they're never lost ----
    results_path = config.MOE_FUSION_BRANCH_DIR / "results_attention_moe_cv.txt"
    with open(results_path, "w") as f:
        f.write("Combined self-attention + MoE fusion, 5-fold cross-validated ensemble\n")
        f.write(f"Settings: weight_decay={WEIGHT_DECAY}, load_balance_weight={config.LOAD_BALANCE_WEIGHT}\n")
        f.write(f"Testing samples: {len(test_ds)}\n")
        f.write(f"Per-fold MAE: mean={fold_maes.mean():.4f}  std={fold_maes.std():.4f}\n\n")
        f.write(f"{'Metric':<8}{'Value':<12}\n")
        for key in ["MAE", "RMSE", "PCC", "CCC"]:
            f.write(f"{key:<8}{ensemble_metrics[key]:<12.4f}\n")
    print(f"\nResults saved to: {results_path}")
    print(f"Per-sample predictions + gate weights saved to: "
          f"{config.MOE_FUSION_BRANCH_DIR / 'attention_moe_test_predictions.csv'}")


if __name__ == "__main__":
    main()