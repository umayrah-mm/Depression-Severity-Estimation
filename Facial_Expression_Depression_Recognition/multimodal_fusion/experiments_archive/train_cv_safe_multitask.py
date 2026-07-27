"""
train_cv_safe_multitask.py

Leak-free cross-validation + ensembling for the MULTI-TASK attention
model (models_attention_multitask.py): trains on BOTH the regression
loss (score) and classification loss (severity band) together, using
the exact same safe methodology as train_cv_safe_attention.py.

Combined loss = regression_loss + TASK_WEIGHT * classification_loss

Uses the same proven Round 3 settings (dropout=0.6, weight_decay=1e-2)
as the current best single-task attention model, so any difference we
see is attributable to the multi-task addition itself, not a different
regularization setup.

Train+Dev pool (197 people) used for the 5 folds. Testing split (100
people) untouched until final evaluation.

Run:
    python train_cv_safe_multitask.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr

import config
from dataset import load_split_dataset
from models_attention import MODALITY_ORDER
from models_attention_multitask import (
    DepressionPredictionModelMultitask, score_to_band, BAND_NAMES, count_trainable_parameters
)

N_FOLDS = 5
PATIENCE = 15
WEIGHT_DECAY = 1e-2   # proven Round 3 settings
TASK_WEIGHT = 0.5     # how much the classification loss counts, relative to regression


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


def run_one_epoch(model, loader, regression_loss_fn, classification_loss_fn, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            inputs = {name: batch[name].to(device) for name in MODALITY_ORDER}
            y_true = batch["y"].to(device)
            band_true = torch.tensor(
                [score_to_band(s.item()) for s in y_true], dtype=torch.long, device=device
            )

            if train:
                optimizer.zero_grad()

            score_pred, band_logits, _ = model(inputs)

            reg_loss = regression_loss_fn(score_pred, y_true)
            cls_loss = classification_loss_fn(band_logits, band_true)
            loss = reg_loss + TASK_WEIGHT * cls_loss

            if train:
                loss.backward()
                optimizer.step()

            batch_size = y_true.size(0)
            total_loss += loss.item() * batch_size
            n += batch_size

    return total_loss / n


def main():
    torch.manual_seed(config.RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Settings: {N_FOLDS} folds, weight_decay={WEIGHT_DECAY}, task_weight={TASK_WEIGHT}")

    model_check = DepressionPredictionModelMultitask()
    n_params = count_trainable_parameters(model_check)
    print(f"Model trainable parameters per fold: {n_params:,} ({n_params/1e6:.3f} M)")

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

        model = DepressionPredictionModelMultitask().to(device)
        regression_loss_fn = nn.SmoothL1Loss()
        classification_loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_state = None

        for epoch in range(1, config.EPOCHS + 1):
            train_loss = run_one_epoch(
                model, train_loader, regression_loss_fn, classification_loss_fn, optimizer, device, train=True
            )
            val_loss = run_one_epoch(
                model, val_loader, regression_loss_fn, classification_loss_fn, optimizer, device, train=False
            )

            improved = val_loss < best_val_loss
            if epoch % 10 == 0 or epoch == 1:
                marker = "  <-- best so far" if improved else ""
                print(f"  Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}{marker}")

            if improved:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

        final_train_loss = run_one_epoch(
            model, train_loader, regression_loss_fn, classification_loss_fn, optimizer, device, train=False
        )
        gap = best_val_loss - final_train_loss
        print(f"  Final check -> train_loss={final_train_loss:.4f}  val_loss={best_val_loss:.4f}  (gap={gap:.4f})")

        model.load_state_dict(best_state)
        fold_path = config.MOE_FUSION_BRANCH_DIR / f"cv_safe_multitask_fold{fold_index + 1}.pt"
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

    all_fold_score_preds = []
    all_fold_band_correct = []
    y_true_final = None

    for fold_path in fold_model_paths:
        model = DepressionPredictionModelMultitask().to(device)
        model.load_state_dict(torch.load(fold_path, map_location=device))
        model.eval()

        score_preds, trues, band_preds, band_trues = [], [], [], []
        with torch.no_grad():
            for batch in test_loader:
                inputs = {name: batch[name].to(device) for name in MODALITY_ORDER}
                y_true = batch["y"]
                score_pred, band_logits, _ = model(inputs)

                score_preds.extend(score_pred.cpu().numpy().tolist())
                trues.extend(y_true.numpy().tolist())
                band_preds.extend(band_logits.argmax(dim=1).cpu().numpy().tolist())
                band_trues.extend([score_to_band(s.item()) for s in y_true])

        all_fold_score_preds.append(score_preds)
        y_true_final = trues

        band_accuracy = np.mean(np.array(band_preds) == np.array(band_trues))
        all_fold_band_correct.append(band_accuracy)

        single_fold_metrics = compute_metrics(trues, score_preds)
        print(f"  {fold_path.name} alone -> MAE={single_fold_metrics['MAE']:.4f}  "
              f"band_accuracy={band_accuracy:.4f}")

    all_fold_score_preds = np.array(all_fold_score_preds)
    ensemble_preds = all_fold_score_preds.mean(axis=0)
    ensemble_metrics = compute_metrics(y_true_final, ensemble_preds)

    single_task_attention = {"MAE": 7.3274, "RMSE": 9.5848, "PCC": 0.5831, "CCC": 0.5490}

    print(f"\nAverage severity-band classification accuracy across folds: {np.mean(all_fold_band_correct):.4f}  "
          f"(4 bands, so 0.25 = random guessing)")

    print("\n================ FINAL, HONEST COMPARISON ================")
    print(f"{'Metric':<8}{'Single-task attention (best)':<30}{'Multi-task attention (new)':<26}")
    for key in ["MAE", "RMSE", "PCC", "CCC"]:
        print(f"{key:<8}{single_task_attention[key]:<30.4f}{ensemble_metrics[key]:<26.4f}")


if __name__ == "__main__":
    main()