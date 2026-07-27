"""
train_no_rppg.py

Ablation: trains and evaluates the MoE fusion model WITHOUT the rPPG
modality (visual + clip + smile/audio only), using the SAME official
Training/Development/Testing split as evaluate.py, so the result is
directly comparable to your current MAE 8.15 (4-modality) result and
your MAE 9.02 (no-CLIP) result.

This does NOT modify models.py, evaluate.py, or train_moe.py. It only
temporarily tells the model to build itself with 3 modalities instead
of 4, for this one run.

Run:
    python train_no_rppg.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

import config
import models
from dataset import load_split_dataset

PATIENCE = 15

# ---- The only real change: drop "rppg" from the modality list ----
NO_RPPG_MODALITIES = ["visual", "clip", "smile"]
models.MODALITY_ORDER = NO_RPPG_MODALITIES


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


def run_one_epoch(model, loader, loss_fn, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            inputs = {name: batch[name].to(device) for name in NO_RPPG_MODALITIES}
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
    print(f"Modalities used: {NO_RPPG_MODALITIES}  (rPPG removed)")

    train_ds = load_split_dataset("Training")
    dev_ds = load_split_dataset("Development")
    test_ds = load_split_dataset("Testing")
    print(f"\nTraining: {len(train_ds)}  Development: {len(dev_ds)}  Testing: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    model = models.DepressionPredictionModel().to(device)
    n_params = models.count_trainable_parameters(model)
    print(f"Trainable parameters: {n_params:,} ({n_params/1e6:.3f} M)  "
          f"(expect slightly fewer than 547,909 - rppg's projector/expert are small, so the drop will be small)")

    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_dev_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    print(f"\nTraining for up to {config.EPOCHS} epochs (patience={PATIENCE})...\n")
    for epoch in range(1, config.EPOCHS + 1):
        train_loss = run_one_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
        dev_loss = run_one_epoch(model, dev_loader, loss_fn, optimizer, device, train=False)

        improved = dev_loss < best_dev_loss
        if epoch % 10 == 0 or epoch == 1:
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

    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs = {name: batch[name].to(device) for name in NO_RPPG_MODALITIES}
            y_pred, _ = model(inputs)
            all_preds.extend(y_pred.cpu().numpy().tolist())
            all_true.extend(batch["y"].numpy().tolist())

    new_metrics = compute_metrics(all_true, all_preds)
    original = {"MAE": 8.1516, "RMSE": 9.9038, "PCC": 0.5395, "CCC": 0.4944}
    no_clip = {"MAE": 9.0150, "RMSE": 11.4146, "PCC": 0.3907, "CCC": 0.3750}

    print("\n================ COMPARISON ON TEST SET ================")
    print(f"{'Metric':<8}{'Original (4 mod)':<20}{'No CLIP (3 mod)':<20}{'No rPPG (3 mod)':<20}")
    for key in ["MAE", "RMSE", "PCC", "CCC"]:
        print(f"{key:<8}{original[key]:<20.4f}{no_clip[key]:<20.4f}{new_metrics[key]:<20.4f}")

    output_path = config.MOE_FUSION_BRANCH_DIR / "best_model_no_rppg.pt"
    torch.save(model.state_dict(), output_path)
    print(f"\nSaved model to: {output_path}")


if __name__ == "__main__":
    main()