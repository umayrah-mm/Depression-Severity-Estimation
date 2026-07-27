"""
train_cv.py

Module C1: 5-fold cross-validation for the MoE fusion model.

Why: a single train/dev/test split (your current best_model.pt result)
gives ONE number per metric, based on how the model did on ONE specific
group of 100 people. With cross-validation, we combine ALL your data
(Training + Development + Testing = 297 people), split it into 5 groups,
and train/test 5 separate times so every person gets tested on exactly
once. Averaging the 5 results gives a far more trustworthy estimate of
your model's real performance, and the spread across folds tells us how
much that estimate could vary by chance.

This does NOT touch or overwrite best_model.pt, evaluate.py, or any of
your existing CSVs. It trains 5 NEW models (saved separately) and writes
NEW result files only.

Run:
    python train_cv.py
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
from models import DepressionPredictionModel, MODALITY_ORDER as MODEL_MODALITY_ORDER, count_trainable_parameters


N_FOLDS = 5
PATIENCE = 15
LOG_GATE_EVERY = 5


def load_balancing_loss(gate_weights):
    avg_usage_per_expert = gate_weights.mean(dim=0)
    num_modalities = gate_weights.shape[1]
    target_usage = 1.0 / num_modalities
    return ((avg_usage_per_expert - target_usage) ** 2).sum()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


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


def run_one_epoch(model, loader, loss_fn, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    n_samples = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            inputs = {name: batch[name].to(device) for name in MODEL_MODALITY_ORDER}
            y_true = batch["y"].to(device)

            if train:
                optimizer.zero_grad()

            y_pred, gate_weights = model(inputs)
            prediction_loss = loss_fn(y_pred, y_true)
            balance_loss = load_balancing_loss(gate_weights)
            loss = prediction_loss + config.LOAD_BALANCE_WEIGHT * balance_loss

            if train:
                loss.backward()
                optimizer.step()

            batch_size = y_true.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

    return total_loss / n_samples


def train_one_fold(fold_index, train_subset, val_subset, device):
    print(f"\n{'=' * 60}")
    print(f"FOLD {fold_index + 1}/{N_FOLDS}")
    print(f"{'=' * 60}")
    print(f"Train samples: {len(train_subset)}  |  Val samples: {len(val_subset)}")

    train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = DepressionPredictionModel().to(device)
    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = run_one_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
        val_loss = run_one_epoch(model, val_loader, loss_fn, optimizer, device, train=False)

        improved = val_loss < best_val_loss
        if epoch % LOG_GATE_EVERY == 0 or epoch == 1:
            marker = "  <-- best so far" if improved else ""
            print(f"  Epoch {epoch:3d}/{config.EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}{marker}")

        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs).")
            break

    model.load_state_dict(best_state)

    # Save this fold's model separately - does not touch best_model.pt
    fold_model_path = config.MOE_FUSION_BRANCH_DIR / f"cv_model_fold{fold_index + 1}.pt"
    torch.save(model.state_dict(), fold_model_path)
    print(f"  Saved: {fold_model_path}")

    # ---- Collect out-of-fold predictions for this fold's val set ----
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in val_loader:
            inputs = {name: batch[name].to(device) for name in MODEL_MODALITY_ORDER}
            y_true = batch["y"]
            y_pred, gate_weights = model(inputs)
            for i in range(len(y_true)):
                row = {
                    "sample_id": batch["video_id"][i],
                    "true_label": y_true[i].item(),
                    "prediction": y_pred[i].item(),
                    "fold": fold_index + 1,
                }
                for j, name in enumerate(MODEL_MODALITY_ORDER):
                    row[f"{name}_gate"] = gate_weights[i, j].item()
                rows.append(row)

    fold_df = pd.DataFrame(rows)
    metrics = compute_metrics(fold_df["true_label"], fold_df["prediction"])
    print(f"  Fold {fold_index + 1} results: MAE={metrics['MAE']:.4f} RMSE={metrics['RMSE']:.4f} "
          f"PCC={metrics['PCC']:.4f} CCC={metrics['CCC']:.4f}")

    return metrics, fold_df


def main():
    set_seed(config.RANDOM_SEED)
    config.ensure_output_dirs()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Combine ALL data (Training + Development + Testing) into one pool ----
    print("\nLoading and combining all splits into one pool...")
    train_ds = load_split_dataset("Training")
    dev_ds = load_split_dataset("Development")
    test_ds = load_split_dataset("Testing")
    full_ds = ConcatDataset([train_ds, dev_ds, test_ds])
    print(f"Total combined samples: {len(full_ds)}  (expected: {len(train_ds) + len(dev_ds) + len(test_ds)})")

    model_check = DepressionPredictionModel()
    n_params = count_trainable_parameters(model_check)
    print(f"Model trainable parameters per fold: {n_params:,} ({n_params/1e6:.3f} M)")

    # ---- Build a "group ID" for every sample, so people stay together ----
    # sample_id looks like "203_1_Freeform_video" or "215_2_Northwind_video".
    # We use the number BEFORE the first underscore (e.g. "203", "215") as
    # the person's group ID, so both their videos always end up in the
    # SAME fold - never split between train and val for the same person.
    all_sample_ids = []
    for i in range(len(full_ds)):
        all_sample_ids.append(full_ds[i]["video_id"])
    group_ids = [str(sid).split("_")[0] for sid in all_sample_ids]

    unique_people = sorted(set(group_ids))
    print(f"\nTotal video rows: {len(group_ids)}  |  Unique people: {len(unique_people)}")

    # ---- Create 5 GROUPED folds - keeps each person entirely in one fold ----
    from sklearn.model_selection import GroupKFold
    kfold = GroupKFold(n_splits=N_FOLDS)
    indices = np.arange(len(full_ds))

    all_metrics = []
    all_oof_dfs = []

    for fold_index, (train_idx, val_idx) in enumerate(kfold.split(indices, groups=group_ids)):
        train_subset = Subset(full_ds, train_idx)
        val_subset = Subset(full_ds, val_idx)

        # Safety check: confirm no person appears in both train and val
        train_people = set(group_ids[i] for i in train_idx)
        val_people = set(group_ids[i] for i in val_idx)
        overlap = train_people & val_people
        if overlap:
            raise RuntimeError(f"LEAKAGE DETECTED in fold {fold_index + 1}: "
                                f"these people are in both train and val: {overlap}")
        print(f"  Verified: 0 people overlap between train ({len(train_people)} people) "
              f"and val ({len(val_people)} people)")

        metrics, fold_df = train_one_fold(fold_index, train_subset, val_subset, device)
        all_metrics.append(metrics)
        all_oof_dfs.append(fold_df)

    # ---- Combine out-of-fold predictions from all 5 folds ----
    oof_df = pd.concat(all_oof_dfs, ignore_index=True)
    oof_path = config.MOE_FUSION_BRANCH_DIR / "cv_out_of_fold_predictions.csv"
    oof_df.to_csv(oof_path, index=False)

    # ---- Summarize results across folds ----
    metrics_df = pd.DataFrame(all_metrics)
    summary_path = config.MOE_FUSION_BRANCH_DIR / "cv_fold_results.csv"
    metrics_df.to_csv(summary_path, index=False)

    print(f"\n{'=' * 60}")
    print("CROSS-VALIDATION SUMMARY (5 folds)")
    print(f"{'=' * 60}")
    for metric_name in ["MAE", "RMSE", "PCC", "CCC"]:
        values = metrics_df[metric_name]
        print(f"{metric_name}: mean={values.mean():.4f}  std={values.std():.4f}  "
              f"(min={values.min():.4f}, max={values.max():.4f})")

    # ---- Also compute metrics on ALL out-of-fold predictions pooled together ----
    # (a second, slightly different way to look at the same result - using
    # every single prediction at once rather than averaging 5 fold-level scores)
    pooled_metrics = compute_metrics(oof_df["true_label"], oof_df["prediction"])
    print(f"\nPooled (all 297 out-of-fold predictions treated as one group):")
    print(f"  MAE={pooled_metrics['MAE']:.4f} RMSE={pooled_metrics['RMSE']:.4f} "
          f"PCC={pooled_metrics['PCC']:.4f} CCC={pooled_metrics['CCC']:.4f}")

    print(f"\nSaved:")
    print(f"  {oof_path}")
    print(f"  {summary_path}")
    print(f"  5 fold models: cv_model_fold1.pt ... cv_model_fold5.pt")


if __name__ == "__main__":
    main()