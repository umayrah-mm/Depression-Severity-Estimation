"""
export_predictions_with_heads.py

Module B3: same purpose as export_predictions.py, but uses
best_model_with_heads.pt (the version with trained per-modality heads
from Module B2) instead of best_model.pt, and additionally saves each
modality's OWN individual prediction per sample.

This does not modify export_predictions.py, best_model.pt, or any
earlier CSVs - it writes three NEW files with "_with_heads" in the name.

Run:
    python export_predictions_with_heads.py
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from dataset import load_split_dataset
from models import DepressionPredictionModel, MODALITY_ORDER


def export_split(split_name, output_filename, model, device):
    print(f"\nLoading {split_name} split...")
    ds = load_split_dataset(split_name)
    print(f"{split_name} samples: {len(ds)}")
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False)

    rows = []
    with torch.no_grad():
        for batch in loader:
            inputs = {name: batch[name].to(device) for name in MODALITY_ORDER}
            y_true = batch["y"]
            y_pred, gate_weights, modality_predictions = model(inputs)

            for i in range(len(y_true)):
                row = {
                    "sample_id": batch["video_id"][i],
                    "true_label": y_true[i].item(),
                    "prediction": y_pred[i].item(),
                }
                for j, name in enumerate(MODALITY_ORDER):
                    row[f"{name}_gate"] = gate_weights[i, j].item()
                for name in MODALITY_ORDER:
                    row[f"{name}_pred"] = modality_predictions[name][i].item()
                rows.append(row)

    df = pd.DataFrame(rows)
    output_path = config.MOE_FUSION_BRANCH_DIR / output_filename
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}  ({len(df)} rows)")
    return df


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = DepressionPredictionModel().to(device)
    model_path = config.MOE_FUSION_BRANCH_DIR / "best_model_with_heads.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from: {model_path}")

    export_split("Training", "train_predictions_with_heads.csv", model, device)
    export_split("Development", "dev_predictions_with_heads.csv", model, device)
    export_split("Testing", "test_predictions_with_heads.csv", model, device)

    print("\nDone. You should now have three new files in your output folder:")
    print("  train_predictions_with_heads.csv")
    print("  dev_predictions_with_heads.csv")
    print("  test_predictions_with_heads.csv")


if __name__ == "__main__":
    main()