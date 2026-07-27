"""
train_cv_safe_weighted.py

Same leak-free cross-validation + ensembling as train_cv_safe.py, but
with a WEIGHTED loss - mistakes on rarer severity bands (Mild, Moderate,
Severe) count more than mistakes on the common None/Minimal band, so
the model can't get a good average error just by being lazy on the
less common, more clinically important cases.

Weights (inverse-frequency, balanced across 4 bands, based on the real
297-sample distribution):
    None/Minimal (0-13):  0.48x
    Mild (14-19):         1.69x
    Moderate (20-28):     1.46x
    Severe (29-63):       1.55x

Train+Dev pool (197 people) used for the 5 folds, grouped by person (no
leakage). Testing split (100 people) untouched until final evaluation.
Compares against the unweighted MoE ensemble (MAE 7.7283).

Run:
    python train_cv_safe_weighted.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr

import config
from dataset import load_split_dataset
from models import DepressionPredictionModel, MODALITY_ORDER, count_trainable_parameters

N_FOLDS = 5
PATIENCE = 15


def get_sample_weight(score):
    """Returns the loss multiplier for one person, based on their severity band."""
    if score <= 13:
        return 0.48
    elif score <= 19:
        return 1.69
    elif score <= 28:
        return 1.46
    else:
        return 1.55


def load_balancing_loss(gate_weights):
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


def run_one_epoch(model, loader, base_loss_fn, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            inputs = {name: batch[name].to(device) for name in MODALITY_ORDER}
            y_true = batch["y"].to(device)

            if train:
                optimizer.zero_grad()

            y_pred, gate_weights = model(inputs)

            # ---- Per-sample weighting: compute loss per person, then weight it ----
            per_sample_loss = base_loss_fn(y_pred, y_true)  # shape: (batch,)
            weights = torch.tensor(
                [get_sample_weight(s.item()) for s in y_true], dtype=torch.float32, device=device
            )
            weighted_loss = (per_sample_loss * weights).mean()

            balance_loss = load_balancing_loss(gate_weights)
            loss = weighted_loss + config.LOAD_BALANCE_WEIGHT * balance_loss

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

    # SmoothL1Loss with reduction="none" gives us PER-SAMPLE loss values,
    # which we need so we can multiply each one by its own weight.
    base_loss_fn = nn.SmoothL1Loss(reduction="none")

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

        model = DepressionPredictionModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_state = None

        for epoch in range(1, config.EPOCHS + 1):
            train_loss = run_one_epoch(model, train_loader, base_loss_fn, optimizer, device, train=True)
            val_loss = run_one_epoch(model, val_loader, base_loss_fn, optimizer, device, train=False)

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

        final_train_loss = run_one_epoch(model, train_loader, base_loss_fn, optimizer, device, train=False)
        print(f"  Final check -> train_loss={final_train_loss:.4f}  val_loss={best_val_loss:.4f}  "
              f"(gap={best_val_loss - final_train_loss:.4f})")

        model.load_state_dict(best_state)
        fold_path = config.MOE_FUSION_BRANCH_DIR / f"cv_safe_weighted_fold{fold_index + 1}.pt"
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
    y_true_final = None

    for fold_path in fold_model_paths:
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

        single_fold_metrics = compute_metrics(trues, preds)
        print(f"  {fold_path.name} alone -> MAE={single_fold_metrics['MAE']:.4f}")

    all_fold_preds = np.array(all_fold_preds)
    ensemble_preds = all_fold_preds.mean(axis=0)
    ensemble_metrics = compute_metrics(y_true_final, ensemble_preds)

    unweighted_moe = {"MAE": 7.7283, "RMSE": 10.0032, "PCC": 0.5145, "CCC": 0.4488}
    attention_best = {"MAE": 7.3274, "RMSE": 9.5848, "PCC": 0.5831, "CCC": 0.5490}

    print("\n================ FINAL, HONEST COMPARISON ================")
    print(f"{'Metric':<8}{'MoE unweighted':<18}{'Attention (best)':<20}{'MoE weighted (new)':<20}")
    for key in ["MAE", "RMSE", "PCC", "CCC"]:
        print(f"{key:<8}{unweighted_moe[key]:<18.4f}{attention_best[key]:<20.4f}{ensemble_metrics[key]:<20.4f}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()