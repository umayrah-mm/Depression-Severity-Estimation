"""
train_modality_heads.py

Module B2: trains ONLY the 4 new per-modality prediction heads added in
Module B1 (models.py). Everything else in the model (projectors, experts,
gate, main fused prediction head) is FROZEN - their weights are loaded
from your already-trained best_model.pt and never change here.

Why: we want each modality to produce its own honest, independent BDI-II
guess, so the upcoming correction step (Module B4) has real per-modality
disagreement to learn from - not just repackaged versions of one number.

Training rules:
    - Only modality_heads parameters are updated.
    - Loss = L1 loss (MAE) between each modality's own guess and the true
      label, averaged across the 4 modalities.
    - Early stopping based on Development split loss, same pattern as
      your original Module 7 training script.

Output:
    best_model_with_heads.pt - a full copy of your model, with the same
    frozen weights as best_model.pt, PLUS trained modality heads. Your
    original best_model.pt is never modified.

Run:
    python train_modality_heads.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import load_split_dataset
from models import DepressionPredictionModel, MODALITY_ORDER

# --- Hyperparameters for this training run only ---
MAX_EPOCHS = 150
LEARNING_RATE = 1e-3
PATIENCE = 15          # stop if no improvement for this many epochs
SEED = 42


def compute_modality_loss(modality_predictions, y_true, loss_fn):
    """
    Averages the L1 loss across all 4 modality heads.
    modality_predictions: dict of {name: (batch,) tensor}
    y_true: (batch,) tensor
    """
    losses = [loss_fn(modality_predictions[name], y_true) for name in MODALITY_ORDER]
    return torch.stack(losses).mean()


def evaluate_modality_loss(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            inputs = {name: batch[name].to(device) for name in MODALITY_ORDER}
            y_true = batch["y"].to(device)
            _, _, modality_predictions = model(inputs)
            loss = compute_modality_loss(modality_predictions, y_true, loss_fn)
            total_loss += loss.item() * len(y_true)
            total_samples += len(y_true)
    return total_loss / total_samples


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Load data ----
    train_ds = load_split_dataset("Training")
    dev_ds = load_split_dataset("Development")
    print(f"Training samples: {len(train_ds)}")
    print(f"Development samples: {len(dev_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    # ---- Load the already-trained model ----
    model = DepressionPredictionModel().to(device)
    model_path = config.MOE_FUSION_BRANCH_DIR / "best_model.pt"
    model.load_state_dict(model.state_dict() | torch.load(model_path, map_location=device))
    print(f"Loaded base weights from: {model_path}")

    # ---- Freeze everything except the new modality heads ----
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        if "modality_heads" in name:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    print(f"\nFrozen parameters:    {frozen_count:,}")
    print(f"Trainable parameters: {trainable_count:,}  (should be small - just the 4 new heads)")

    # ---- Optimizer only sees the unfrozen (trainable) parameters ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE)
    loss_fn = nn.L1Loss()

    best_dev_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    print(f"\nTraining for up to {MAX_EPOCHS} epochs (early stopping patience: {PATIENCE})...\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.eval()  # keep frozen parts in eval mode (no dropout randomness in the frozen backbone)
        total_train_loss = 0.0
        total_train_samples = 0

        for batch in train_loader:
            inputs = {name: batch[name].to(device) for name in MODALITY_ORDER}
            y_true = batch["y"].to(device)

            optimizer.zero_grad()
            _, _, modality_predictions = model(inputs)
            loss = compute_modality_loss(modality_predictions, y_true, loss_fn)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * len(y_true)
            total_train_samples += len(y_true)

        train_loss = total_train_loss / total_train_samples
        dev_loss = evaluate_modality_loss(model, dev_loader, device, loss_fn)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}  |  Train loss: {train_loss:.4f}  |  Dev loss: {dev_loss:.4f}")

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            epochs_without_improvement = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no Dev improvement for {PATIENCE} epochs).")
            break

    print(f"\nBest Development loss (avg per-modality MAE): {best_dev_loss:.4f}")

    # ---- Save the full model (frozen parts unchanged + trained heads) ----
    model.load_state_dict(best_state)
    output_path = config.MOE_FUSION_BRANCH_DIR / "best_model_with_heads.pt"
    torch.save(model.state_dict(), output_path)
    print(f"Saved model with trained modality heads to: {output_path}")

    # ---- Sanity check: report per-modality MAE on Dev, for context ----
    model.eval()
    per_modality_totals = {name: 0.0 for name in MODALITY_ORDER}
    total_samples = 0
    with torch.no_grad():
        for batch in dev_loader:
            inputs = {name: batch[name].to(device) for name in MODALITY_ORDER}
            y_true = batch["y"].to(device)
            _, _, modality_predictions = model(inputs)
            for name in MODALITY_ORDER:
                per_modality_totals[name] += torch.sum(torch.abs(modality_predictions[name] - y_true)).item()
            total_samples += len(y_true)

    print(f"\nPer-modality MAE on Development split (each modality guessing ALONE):")
    for name in MODALITY_ORDER:
        print(f"  {name}: {per_modality_totals[name] / total_samples:.4f}")
    print("\n(These are expected to be worse than your main fused prediction's MAE -")
    print(" that's normal. We only need them to be honest, distinct opinions.)")


if __name__ == "__main__":
    main()