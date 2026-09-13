"""
train_cv_safe_evidence.py

Leak-free 5-fold CV training for EvidenceFusionModel.
Same safe methodology as train_cv_safe_attention.py:
    - 5 folds trained on Training+Development people only
    - Testing split never touched until final, one-time evaluation
    - GroupKFold by subject ID, leakage checked explicitly every fold

Differences vs. the attention script:
    - modality dropout applied ONLY during training batches (evidence_training_utils.py)
    - combined_loss (SmoothL1 main + heteroscedastic aux) instead of plain SmoothL1Loss
    - validation/testing always use mask=all-present (real, undegraded data) -
      the missing-modality robustness check is a SEPARATE script, run after this one

VERIFY BEFORE RUNNING: batch key names below must match your dataset.py.

Run:
    python train_cv_safe_evidence.py
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr

import config
from dataset import load_split_dataset
from models_evidence import EvidenceFusionModel
from evidence_training_utils import apply_modality_dropout, combined_loss

MODALITY_ORDER = ["visual", "clip", "rppg", "smile"]  # <-- verify against dataset.py

N_FOLDS = 5
PATIENCE = 15
WEIGHT_DECAY = 1e-2
DROPOUT_PROB = 0.15
AUX_LOSS_WEIGHT = 0.3


def concordance_ccc(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)


def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    try:
        pcc = pearsonr(y_true, y_pred)[0]
    except Exception:
        pcc = 0.0
    return {"MAE": mae, "RMSE": rmse, "PCC": pcc, "CCC": concordance_ccc(y_true, y_pred)}


def run_one_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_main, total_aux, n = 0.0, 0.0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            visual = batch["visual"].to(device)
            clip = batch["clip"].to(device)
            rppg = batch["rppg"].to(device)
            smile = batch["smile"].to(device)
            targets = batch["y"].to(device)

            if train:
                visual, clip, rppg, smile, mask = apply_modality_dropout(
                    visual, clip, rppg, smile, dropout_prob=DROPOUT_PROB
                )
                optimizer.zero_grad()
            else:
                mask = torch.ones(visual.shape[0], 4, device=device)

            outputs = model(visual, clip, rppg, smile, mask)
            total_loss, main_loss, aux_loss = combined_loss(
                outputs, targets, mask, aux_loss_weight=AUX_LOSS_WEIGHT
            )

            if train:
                total_loss.backward()
                optimizer.step()

            batch_n = len(targets)
            total_main += main_loss.item() * batch_n
            total_aux += aux_loss.item() * batch_n
            n += batch_n

    return total_main / n, total_aux / n


def main():
    torch.manual_seed(config.RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    config.EVIDENCE_BRANCH_DIR.mkdir(parents=True, exist_ok=True)

    model_check = EvidenceFusionModel()
    n_params = sum(p.numel() for p in model_check.parameters())
    print(f"Model trainable parameters per fold: {n_params:,} ({n_params/1e6:.3f} M)")

    train_ds = load_split_dataset("Training")
    dev_ds = load_split_dataset("Development")
    pool_ds = ConcatDataset([train_ds, dev_ds])
    print(f"Train+Dev pool size: {len(pool_ds)}")

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
        print("Verified: 0 people overlap between train and val")

        train_subset = Subset(pool_ds, train_idx)
        val_subset = Subset(pool_ds, val_idx)
        train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False)

        model = EvidenceFusionModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        best_val_main = float("inf")
        epochs_without_improvement = 0
        best_state = None

        for epoch in range(1, config.EPOCHS + 1):
            train_main, train_aux = run_one_epoch(model, train_loader, optimizer, device, train=True)
            val_main, val_aux = run_one_epoch(model, val_loader, optimizer, device, train=False)

            improved = val_main < best_val_main
            if epoch % 10 == 0 or epoch == 1:
                marker = "  <-- best so far" if improved else ""
                print(f"  Epoch {epoch:3d} | train_main={train_main:.4f} train_aux={train_aux:.4f} "
                      f"| val_main={val_main:.4f} val_aux={val_aux:.4f}{marker}")

            if improved:
                best_val_main = val_main
                epochs_without_improvement = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

        model.load_state_dict(best_state)
        fold_path = config.EVIDENCE_BRANCH_DIR / f"cv_safe_evidence_fold{fold_index + 1}.pt"
        torch.save(model.state_dict(), fold_path)
        fold_model_paths.append(fold_path)
        print(f"  Saved: {fold_path}")

    print(f"\n{'=' * 60}")
    print("Evaluating all 5 folds on the REAL Testing split (all modalities present)")
    print(f"{'=' * 60}")

    test_ds = load_split_dataset("Testing")
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"Testing samples: {len(test_ds)}")

    all_fold_preds = []
    y_true_final = None
    per_fold_metrics = []

    for fold_path in fold_model_paths:
        model = EvidenceFusionModel().to(device)
        model.load_state_dict(torch.load(fold_path, map_location=device))
        model.eval()

        preds, trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                visual = batch["visual"].to(device)
                clip = batch["clip"].to(device)
                rppg = batch["rppg"].to(device)
                smile = batch["smile"].to(device)
                mask = torch.ones(visual.shape[0], 4, device=device)

                outputs = model(visual, clip, rppg, smile, mask)
                preds.extend(outputs["prediction"].squeeze(-1).cpu().numpy().tolist())
                trues.extend(batch["y"].numpy().tolist())

        all_fold_preds.append(preds)
        y_true_final = trues
        m = compute_metrics(trues, preds)
        per_fold_metrics.append(m)
        print(f"  {fold_path.name} alone -> MAE={m['MAE']:.4f}")

    fold_maes = np.array([m["MAE"] for m in per_fold_metrics])
    print(f"\nPer-fold MAE mean={fold_maes.mean():.4f}  std={fold_maes.std():.4f}")

    all_fold_preds = np.array(all_fold_preds)
    ensemble_preds = all_fold_preds.mean(axis=0)
    ensemble_metrics = compute_metrics(y_true_final, ensemble_preds)

    attention_ensemble = {"MAE": 7.3274, "RMSE": 9.5848, "PCC": 0.5831, "CCC": 0.5490}

    print("\n================ FINAL, HONEST COMPARISON ================")
    print(f"{'Metric':<8}{'Attention (current best)':<28}{'Evidence (new)':<20}")
    for key in ["MAE", "RMSE", "PCC", "CCC"]:
        print(f"{key:<8}{attention_ensemble[key]:<28.4f}{ensemble_metrics[key]:<20.4f}")

    results_path = config.EVIDENCE_BRANCH_DIR / "results_evidence_cv.txt"
    with open(results_path, "w") as f:
        f.write("Evidence-based (uncertainty-weighted) fusion, 5-fold cross-validated ensemble\n")
        f.write(f"Settings: weight_decay={WEIGHT_DECAY}, dropout_prob={DROPOUT_PROB}, "
                f"aux_loss_weight={AUX_LOSS_WEIGHT}\n")
        f.write(f"Testing samples: {len(test_ds)}\n")
        f.write(f"Per-fold MAE: mean={fold_maes.mean():.4f}  std={fold_maes.std():.4f}\n\n")
        f.write(f"{'Metric':<8}{'Value':<12}\n")
        for key in ["MAE", "RMSE", "PCC", "CCC"]:
            f.write(f"{key:<8}{ensemble_metrics[key]:<12.4f}\n")
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()