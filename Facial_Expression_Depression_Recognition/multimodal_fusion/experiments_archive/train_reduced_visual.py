"""
train_reduced_visual.py

Shrinks the 2304-number visual feature down to the 500 most useful
numbers (chosen using ONLY the Training split, so no cheating), then
trains and tests the model with that smaller input - same official
Training/Development/Testing split as evaluate.py, for a fair,
direct comparison against your current MAE 8.15 result.

Run:
    python train_reduced_visual.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.stats import pearsonr

import config
import models
from dataset import align_all, split_by_name, normalize_with_train_stats, MultimodalDataset


K = 500          # how many visual numbers to keep, out of 2304
PATIENCE = 15


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


def load_balancing_loss(gate_weights):
    avg_usage = gate_weights.mean(dim=0)
    target = 1.0 / gate_weights.shape[1]
    return ((avg_usage - target) ** 2).sum()


def run_one_epoch(model, loader, loss_fn, optimizer, device, modality_order, train=True):
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            inputs = {name: batch[name].to(device) for name in modality_order}
            y_true = batch["y"].to(device)
            if train:
                optimizer.zero_grad()
            y_pred, gate_weights = model(inputs)
            loss = loss_fn(y_pred, y_true) + config.LOAD_BALANCE_WEIGHT * load_balancing_loss(gate_weights)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y_true)
            n += len(y_true)
    return total_loss / n


def main():
    torch.manual_seed(config.RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Load raw (unaligned-only, not-yet-normalized) data ----
    print("\nLoading and aligning data...")
    aligned = align_all()
    train_mask = split_by_name(aligned, "Training")
    dev_mask = split_by_name(aligned, "Development")
    test_mask = split_by_name(aligned, "Testing")

    # ---- Step 1: pick the 500 best visual numbers, using TRAINING data only ----
    print(f"\nSelecting the {K} most useful visual features (fit on Training only)...")
    selector = SelectKBest(score_func=f_regression, k=K)
    selector.fit(aligned["visual_X"][train_mask], aligned["y"][train_mask])
    selected_idx = selector.get_support(indices=True)
    print(f"Kept {len(selected_idx)} out of {aligned['visual_X'].shape[1]} visual features.")

    visual_reduced = aligned["visual_X"][:, selected_idx]

    # ---- Step 2: normalize every modality using TRAINING stats only ----
    def norm(modality_X):
        return normalize_with_train_stats(
            modality_X[train_mask], modality_X[dev_mask], modality_X[test_mask]
        )

    visual_train, visual_dev, visual_test, _, _ = normalize_with_train_stats(
        visual_reduced[train_mask], visual_reduced[dev_mask], visual_reduced[test_mask]
    )
    rppg_train, rppg_dev, rppg_test, _, _ = norm(aligned["rppg_X"])
    clip_train, clip_dev, clip_test, _, _ = norm(aligned["clip_X"])
    smile_train, smile_dev, smile_test, _, _ = norm(aligned["smile_X"])

    train_ds = MultimodalDataset(visual_train, rppg_train, clip_train, smile_train,
                                  aligned["y"][train_mask], aligned["video_ids"][train_mask])
    dev_ds = MultimodalDataset(visual_dev, rppg_dev, clip_dev, smile_dev,
                                aligned["y"][dev_mask], aligned["video_ids"][dev_mask])
    test_ds = MultimodalDataset(visual_test, rppg_test, clip_test, smile_test,
                                 aligned["y"][test_mask], aligned["video_ids"][test_mask])

    print(f"\nTraining samples: {len(train_ds)}  Development: {len(dev_ds)}  Testing: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    # ---- Step 3: tell the model the visual input is now size K, not 2304 ----
    models.MODALITY_INPUT_DIMS["visual"] = K
    modality_order = models.MODALITY_ORDER

    model = models.DepressionPredictionModel().to(device)
    n_params = models.count_trainable_parameters(model)
    print(f"Trainable parameters: {n_params:,} ({n_params/1e6:.3f} M)")

    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # ---- Step 4: train, same early-stopping pattern as train_moe.py ----
    best_dev_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    print(f"\nTraining for up to {config.EPOCHS} epochs (patience={PATIENCE})...\n")
    for epoch in range(1, config.EPOCHS + 1):
        train_loss = run_one_epoch(model, train_loader, loss_fn, optimizer, device, modality_order, train=True)
        dev_loss = run_one_epoch(model, dev_loader, loss_fn, optimizer, device, modality_order, train=False)

        improved = dev_loss < best_dev_loss
        if epoch % 5 == 0 or epoch == 1:
            marker = "  <-- best so far" if improved else ""
            print(f"Epoch {epoch:3d}/{config.EPOCHS} | train_loss={train_loss:.4f} | dev_loss={dev_loss:.4f}{marker}")

        if improved:
            best_dev_loss = dev_loss
            epochs_without_improvement = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    model.eval()

    # ---- Step 5: evaluate on the SAME Testing split as evaluate.py ----
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs = {name: batch[name].to(device) for name in modality_order}
            y_pred, _ = model(inputs)
            all_preds.extend(y_pred.cpu().numpy().tolist())
            all_true.extend(batch["y"].numpy().tolist())

    new_metrics = compute_metrics(all_true, all_preds)

    print("\n================ COMPARISON ON TEST SET ================")
    print(f"{'Metric':<8}{'Original (2304 visual feats)':<30}{'Reduced (500 visual feats)':<30}")
    original = {"MAE": 8.1516, "RMSE": 9.9038, "PCC": 0.5395, "CCC": 0.4944}
    for key in ["MAE", "RMSE", "PCC", "CCC"]:
        print(f"{key:<8}{original[key]:<30.4f}{new_metrics[key]:<30.4f}")

    output_path = config.MOE_FUSION_BRANCH_DIR / "best_model_reduced_visual.pt"
    torch.save(model.state_dict(), output_path)
    print(f"\nSaved model to: {output_path}")


if __name__ == "__main__":
    main()