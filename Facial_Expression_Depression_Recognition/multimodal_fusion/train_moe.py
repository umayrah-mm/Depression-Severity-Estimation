"""
train_moe.py

Trains the small, trainable components of the MoE fusion model (projectors,
experts, gate, prediction head) on the aligned, already-normalized
train/dev splits. CLIP and MobileNetV3 remain frozen and were never
touched - only the ~548K parameters defined in models.py get updated here.

Run:
    python train_moe.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import load_split_dataset
from models import DepressionPredictionModel, MODALITY_ORDER as MODEL_MODALITY_ORDER, count_trainable_parameters


PATIENCE = 15  # early stopping: stop if dev loss doesn't improve for this many epochs
LOG_GATE_EVERY = 5  # print average gate weights every N epochs

def load_balancing_loss(gate_weights):
    """
    Penalizes the gate for consistently favoring some experts over others
    across a batch. Does NOT force every individual sample to use all
    experts equally - it only discourages an expert from being starved
    across the whole batch, while still allowing genuine per-sample
    adaptivity (as we saw with visual/rppg).

    gate_weights: (batch, num_modalities), each row already sums to 1.

    Returns a scalar: 0 when all experts get equal total usage across
    the batch, larger when usage is unbalanced.
    """
    avg_usage_per_expert = gate_weights.mean(dim=0)  # (num_modalities,)
    num_modalities = gate_weights.shape[1]
    target_usage = 1.0 / num_modalities  # e.g. 0.25 for 4 experts
    return ((avg_usage_per_expert - target_usage) ** 2).sum()



def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_one_epoch(model, loader, loss_fn, optimizer, device, train=True):
    """
    Runs one full pass over `loader`. If train=True, updates model weights.
    Returns: (average_loss, average_gate_weights as a dict)
    """
    model.train() if train else model.eval()

    total_loss = 0.0
    n_samples = 0
    gate_sum = torch.zeros(len(MODEL_MODALITY_ORDER))

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
            gate_sum += gate_weights.detach().cpu().sum(dim=0)

    avg_loss = total_loss / n_samples
    avg_gates = {name: (gate_sum[i] / n_samples).item() for i, name in enumerate(MODEL_MODALITY_ORDER)}
    return avg_loss, avg_gates


def main():
    set_seed(config.RANDOM_SEED)
    config.ensure_output_dirs()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Data ----
    train_ds = load_split_dataset("Training")
    dev_ds = load_split_dataset("Development")
    print(f"Training samples: {len(train_ds)}")
    print(f"Development samples: {len(dev_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    # ---- Model ----
    model = DepressionPredictionModel().to(device)
    n_params = count_trainable_parameters(model)
    print(f"Trainable parameters: {n_params:,} ({n_params/1e6:.3f} M)")

    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # ---- Training loop with early stopping ----
    best_dev_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history = []

    print(f"\nTraining for up to {config.EPOCHS} epochs (patience={PATIENCE})...\n")

    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_gates = run_one_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
        dev_loss, dev_gates = run_one_epoch(model, dev_loader, loss_fn, optimizer, device, train=False)

        history.append({"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss})

        improved = dev_loss < best_dev_loss
        marker = "  <-- best so far" if improved else ""
        print(f"Epoch {epoch:3d}/{config.EPOCHS} | train_loss={train_loss:.4f} | dev_loss={dev_loss:.4f}{marker}")

        if epoch % LOG_GATE_EVERY == 0 or epoch == 1:
            gate_str = ", ".join(f"{k}={v:.3f}" for k, v in dev_gates.items())
            print(f"    dev gate weights (avg): {gate_str}")

        if improved:
            best_dev_loss = dev_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), config.MOE_FUSION_BRANCH_DIR / "best_model.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping: no dev improvement for {PATIENCE} epochs.")
            break

    print(f"\nBest dev loss: {best_dev_loss:.4f} at epoch {best_epoch}")

    # ---- Save training history ----
    pd.DataFrame(history).to_csv(config.MOE_FUSION_BRANCH_DIR / "training_history.csv", index=False)

    # ---- Reload best model and save dev predictions ----
    model.load_state_dict(torch.load(config.MOE_FUSION_BRANCH_DIR / "best_model.pt"))
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in dev_loader:
            inputs = {name: batch[name].to(device) for name in MODEL_MODALITY_ORDER}
            y_true = batch["y"]
            y_pred, gate_weights = model(inputs)

            for i in range(len(y_true)):
                row = {
                    "video_id": batch["video_id"][i],
                    "true_label": y_true[i].item(),
                    "prediction": y_pred[i].item(),
                }
                for j, name in enumerate(MODEL_MODALITY_ORDER):
                    row[f"{name}_gate"] = gate_weights[i, j].item()
                rows.append(row)

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(config.MOE_FUSION_BRANCH_DIR / "dev_predictions.csv", index=False)

    mae = (pred_df["prediction"] - pred_df["true_label"]).abs().mean()
    print(f"\nDev set MAE (best model): {mae:.4f}")
    print(f"\nSaved:")
    print(f"  {config.MOE_FUSION_BRANCH_DIR / 'best_model.pt'}")
    print(f"  {config.MOE_FUSION_BRANCH_DIR / 'training_history.csv'}")
    print(f"  {config.MOE_FUSION_BRANCH_DIR / 'dev_predictions.csv'}")


if __name__ == "__main__":
    main()