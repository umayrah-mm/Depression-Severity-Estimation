"""
export_predictions.py

Generates the same kind of per-sample CSV that evaluate.py makes for the
Testing split, but for ALL THREE splits: Training, Development, Testing.

Why we need this: the upcoming SVR "cleanup" model must be TRAINED on the
Training split and only ever CHECKED on the Testing split (never trained
or tuned on it). evaluate.py currently only handles Testing, so we need
Training and Development versions too.

This script does NOT change your trained model, your training process,
or evaluate.py in any way. It only re-uses the already-trained model to
write three CSV files.

Run:
    python export_predictions.py
"""

import numpy as np
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
            y_pred, gate_weights = model(inputs)

            for i in range(len(y_true)):
                row = {
                    "sample_id": batch["video_id"][i],
                    "true_label": y_true[i].item(),
                    "prediction": y_pred[i].item(),
                }
                for j, name in enumerate(MODALITY_ORDER):
                    row[f"{name}_gate"] = gate_weights[i, j].item()
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
    model_path = config.MOE_FUSION_BRANCH_DIR / "best_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from: {model_path}")

    export_split("Training", "train_predictions.csv", model, device)
    export_split("Development", "dev_predictions.csv", model, device)
    export_split("Testing", "test_predictions.csv", model, device)

    print("\nDone. You should now have three files in your multimodal_fusion folder:")
    print("  train_predictions.csv")
    print("  dev_predictions.csv")
    print("  test_predictions.csv")


if __name__ == "__main__":
    main()