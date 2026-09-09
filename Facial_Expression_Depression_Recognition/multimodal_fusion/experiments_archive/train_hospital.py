"""
train_hospital.py

Trains the self-attention fusion model from scratch on Hospital_Data,
using its own train/dev/test splits directly (no cross-validation
needed - real splits already exist, unlike the small AVEC2014 setup).

Requires hospital_features_combined.npz from extract_hospital_features.py
to already exist.

Saves:
    <root>\\hospital_model.pt
    <root>\\hospital_normalization_stats.npz
    <root>\\hospital_results.txt

Run:
    python train_hospital.py --root "E:\\Data\\Hospital_Data"
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr

from models_attention import DepressionPredictionModelAttention, MODALITY_ORDER

PATIENCE = 15
MAX_EPOCHS = 120
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
BATCH_SIZE = 16


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


def normalize_with_train_stats(X_train, X_dev, X_test):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1e-8
    return (X_train - mean) / std, (X_dev - mean) / std, (X_test - mean) / std, mean, std


def make_batches(tensors_dict, y, batch_size, shuffle):
    n = len(y)
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, batch_size):
        batch_idx = idx[i:i + batch_size]
        yield {k: v[batch_idx] for k, v in tensors_dict.items()}, y[batch_idx]


def run_epoch(model, tensors_dict, y, loss_fn, optimizer, device, train):
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_x, batch_y in make_batches(tensors_dict, y, BATCH_SIZE, shuffle=train):
            inputs = {k: v.to(device) for k, v in batch_x.items()}
            targets = batch_y.to(device)
            if train:
                optimizer.zero_grad()
            pred, _ = model(inputs)
            loss = loss_fn(pred, targets)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(targets)
            n += len(targets)
    return total_loss / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data = np.load(root / "hospital_features_combined.npz", allow_pickle=True)
    split = data["split"]

    train_mask = split == "train"
    dev_mask = split == "dev"
    test_mask = split == "test"

    print(f"Train: {train_mask.sum()}  Dev: {dev_mask.sum()}  Test: {test_mask.sum()}")
    if train_mask.sum() == 0 or dev_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("One of train/dev/test has 0 samples - check hospital_features_combined.npz")

    norm_stats = {}
    tensors = {"train": {}, "dev": {}, "test": {}}
    for modality, key in [("visual", "visual_X"), ("rppg", "rppg_X"), ("clip", "clip_X"), ("smile", "smile_X")]:
        X = data[key]
        X_train, X_dev, X_test, mean, std = normalize_with_train_stats(X[train_mask], X[dev_mask], X[test_mask])
        tensors["train"][modality] = torch.tensor(X_train, dtype=torch.float32)
        tensors["dev"][modality] = torch.tensor(X_dev, dtype=torch.float32)
        tensors["test"][modality] = torch.tensor(X_test, dtype=torch.float32)
        norm_stats[f"{modality}_X_mean"] = mean
        norm_stats[f"{modality}_X_std"] = std

    y_train = torch.tensor(data["y"][train_mask], dtype=torch.float32)
    y_dev = torch.tensor(data["y"][dev_mask], dtype=torch.float32)
    y_test = torch.tensor(data["y"][test_mask], dtype=torch.float32)

    np.savez(root / "hospital_normalization_stats.npz", **norm_stats)
    print(f"Saved normalization stats to: {root / 'hospital_normalization_stats.npz'}")

    model = DepressionPredictionModelAttention().to(device)
    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_dev_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    print(f"\nTraining for up to {MAX_EPOCHS} epochs (patience={PATIENCE})...\n")
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = run_epoch(model, tensors["train"], y_train, loss_fn, optimizer, device, train=True)
        dev_loss = run_epoch(model, tensors["dev"], y_dev, loss_fn, optimizer, device, train=False)

        improved = dev_loss < best_dev_loss
        if epoch % 10 == 0 or epoch == 1:
            marker = "  <-- best so far" if improved else ""
            print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | dev_loss={dev_loss:.4f}{marker}")

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

    all_preds = []
    with torch.no_grad():
        for i in range(0, len(y_test), BATCH_SIZE):
            batch_x = {k: v[i:i + BATCH_SIZE].to(device) for k, v in tensors["test"].items()}
            pred, _ = model(batch_x)
            all_preds.extend(pred.cpu().numpy().tolist())

    metrics = compute_metrics(y_test.numpy(), all_preds)

    print("\n================ TEST SET RESULTS (HAMD scale, trained from scratch) ================")
    for key in ["MAE", "RMSE", "PCC", "CCC"]:
        print(f"{key}: {metrics[key]:.4f}")

    model_path = root / "hospital_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved model to: {model_path}")

    results_path = root / "hospital_results.txt"
    with open(results_path, "w") as f:
        f.write("Self-attention fusion, trained from scratch on Hospital_Data\n")
        f.write(f"Train: {train_mask.sum()}  Dev: {dev_mask.sum()}  Test: {test_mask.sum()}\n\n")
        for key in ["MAE", "RMSE", "PCC", "CCC"]:
            f.write(f"{key}: {metrics[key]:.4f}\n")
    print(f"Saved results to: {results_path}")


if __name__ == "__main__":
    main()