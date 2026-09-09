"""
train_attention_single_split.py

Single train/dev/test split version of the self-attention fusion model
(models_attention.py) - the missing parallel to train_moe.py, but for
attention instead of the MoE gate.

WHY THIS SCRIPT EXISTS
------------------------
Every other model in this project has an "apples to apples" pair of
results:
  - Baseline (visual+rPPG):  single-split 8.10   AND  5-fold CV 8.49
  - MoE gate:                single-split 8.15   AND  5-fold ensemble 7.73
  - Self-attention:          never run on a single split -  only 5-fold ensemble 7.33

This script fills that gap. It trains the exact same attention
architecture and the exact same "Round 3, final" regularization settings
(dropout=0.6, weight_decay=0.01) that train_cv_safe_attention.py uses
per fold, but on ONE split only: train on Training, early-stop using
Development as the validation set, then evaluate ONCE on the untouched
Testing split - exactly how train_moe.py produced the 8.15 single-split
MoE number.

INPUTS: same aligned/normalized dataset every other training script here
    uses (config.ALIGNED_DATA_DIR, via dataset.load_split_dataset). No
    new data extraction needed.

OUTPUTS:
    - config.MOE_FUSION_BRANCH_DIR/best_model_attention_single.pt
    - config.MOE_FUSION_BRANCH_DIR/attention_single_split_training_history.csv
    - config.MOE_FUSION_BRANCH_DIR/attention_single_split_test_predictions.csv
    - config.MOE_FUSION_BRANCH_DIR/results_attention_single_split.txt

Run:
    python train_attention_single_split.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

import config
from dataset import load_split_dataset
from models_attention import DepressionPredictionModelAttention, MODALITY_ORDER, count_trainable_parameters


PATIENCE = 15
WEIGHT_DECAY = 1e-2  # same "Round 3, final settings" as train_cv_safe_attention.py, for a fair comparison


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
    total_loss, n = 0.0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            inputs = {name: batch[name].to(device) for name in MODALITY_ORDER}
            y_true = batch["y"].to(device)
            if train:
                optimizer.zero_grad()
            y_pred, _ = model(inputs)
            loss = loss_fn(y_pred, y_true)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y_true)
            n += len(y_true)
    return total_loss / n


def main():
    set_seed(config.RANDOM_SEED)
    config.ensure_output_dirs()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = DepressionPredictionModelAttention().to(device)
    n_params = count_trainable_parameters(model)
    print(f"Trainable parameters: {n_params:,} ({n_params/1e6:.3f} M)")
    print(f"Regularization: dropout=0.6 (fixed in models_attention.py), weight_decay={WEIGHT_DECAY}")

    # ---- Data: single split, exactly like train_moe.py ----
    train_ds = load_split_dataset("Training")
    dev_ds = load_split_dataset("Development")
    print(f"Training samples: {len(train_ds)}")
    print(f"Development samples: {len(dev_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # ---- Training loop with early stopping (Development = validation set) ----
    best_dev_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    best_state = None
    history = []

    print(f"\nTraining for up to {config.EPOCHS} epochs (patience={PATIENCE})...\n")

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = run_one_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
        dev_loss = run_one_epoch(model, dev_loader, loss_fn, optimizer, device, train=False)

        history.append({"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss})

        improved = dev_loss < best_dev_loss
        if epoch % 10 == 0 or epoch == 1:
            marker = "  <-- best so far" if improved else ""
            print(f"Epoch {epoch:3d}/{config.EPOCHS} | train_loss={train_loss:.4f} | dev_loss={dev_loss:.4f}{marker}")

        if improved:
            best_dev_loss = dev_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping: no dev improvement for {PATIENCE} epochs (best was epoch {best_epoch}).")
            break

    print(f"\nBest dev loss: {best_dev_loss:.4f} at epoch {best_epoch}")
    model.load_state_dict(best_state)

    model_path = config.MOE_FUSION_BRANCH_DIR / "best_model_attention_single.pt"
    torch.save(model.state_dict(), model_path)

    pd.DataFrame(history).to_csv(
        config.MOE_FUSION_BRANCH_DIR / "attention_single_split_training_history.csv", index=False
    )

    # ---- Overfitting check: train loss vs best dev loss (same diagnostic used for the 5-fold version) ----
    final_train_loss = run_one_epoch(model, train_loader, loss_fn, optimizer, device, train=False)
    print(f"Final check -> train_loss={final_train_loss:.4f}  dev_loss={best_dev_loss:.4f}  "
          f"(gap={best_dev_loss - final_train_loss:.4f})")
    print("(A large, growing gap would signal overfitting - same check used in train_cv_safe_attention.py.)")

    # ---- ONLY NOW, ONE TIME: evaluate on the untouched Testing split ----
    print(f"\n{'=' * 60}")
    print("Evaluating on the REAL, untouched Testing split (one time only)")
    print(f"{'=' * 60}")

    test_ds = load_split_dataset("Testing")
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"Testing samples: {len(test_ds)}")

    model.eval()
    rows = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = {name: batch[name].to(device) for name in MODALITY_ORDER}
            y_true = batch["y"]
            y_pred, _ = model(inputs)
            for i in range(len(y_true)):
                rows.append({
                    "video_id": batch["video_id"][i],
                    "true_label": y_true[i].item(),
                    "prediction": y_pred[i].item(),
                })

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(
        config.MOE_FUSION_BRANCH_DIR / "attention_single_split_test_predictions.csv", index=False
    )

    metrics = compute_metrics(pred_df["true_label"], pred_df["prediction"])

    print("\n================ TEST SET RESULTS (single split, no CV) ================")
    print(f"MAE:  {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"PCC:  {metrics['PCC']:.4f}")
    print(f"CCC:  {metrics['CCC']:.4f}")

    print("\n================ CONTEXT: full comparison table ================")
    print(f"{'Model':<45}{'MAE':<10}")
    print(f"{'Baseline, single split':<45}{'8.10':<10}")
    print(f"{'Baseline, 5-fold CV':<45}{'8.49':<10}")
    print(f"{'MoE gate, single split':<45}{'8.15':<10}")
    print(f"{'MoE gate, 5-fold ensemble':<45}{'7.73':<10}")
    print(f"{'Self-attention, single split (THIS RUN)':<45}{metrics['MAE']:<10.4f}")
    print(f"{'Self-attention, 5-fold ensemble':<45}{'7.33':<10}")

    results_path = config.MOE_FUSION_BRANCH_DIR / "results_attention_single_split.txt"
    with open(results_path, "w") as f:
        f.write("Self-attention fusion, SINGLE train/dev/test split (no cross-validation)\n")
        f.write(f"Settings: dropout=0.6, weight_decay={WEIGHT_DECAY} (same as the 5-fold version, for fair comparison)\n")
        f.write(f"Training samples: {len(train_ds)}  Development samples: {len(dev_ds)}  Testing samples: {len(test_ds)}\n")
        f.write(f"Best epoch: {best_epoch}  Final train/dev gap: {best_dev_loss - final_train_loss:.4f}\n\n")
        f.write(f"{'Metric':<8}{'Value':<12}\n")
        for key in ["MAE", "RMSE", "PCC", "CCC"]:
            f.write(f"{key:<8}{metrics[key]:<12.4f}\n")
        f.write("\nCompare against: 5-fold ensemble self-attention MAE 7.33, MoE single-split 8.15, MoE 5-fold 7.73\n")

    print(f"\nSaved model to:       {model_path}")
    print(f"Saved test preds to:  {config.MOE_FUSION_BRANCH_DIR / 'attention_single_split_test_predictions.csv'}")
    print(f"Saved summary to:     {results_path}")


if __name__ == "__main__":
    main()
