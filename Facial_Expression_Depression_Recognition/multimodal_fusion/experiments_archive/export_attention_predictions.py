"""
export_attention_predictions.py

Generates a per-sample predictions CSV for the FINAL, BEST model
(self-attention fusion, 5-fold ensemble, MAE 7.33) on the real Testing
split - the file that was never saved during training itself.

Output: test_predictions_attention_final.csv
    (sample_id, true_label, prediction)
"""

import numpy as np
import pandas as pd
import torch

import config
from dataset import load_split_dataset
from models_attention import DepressionPredictionModelAttention, MODALITY_ORDER


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    test_ds = load_split_dataset("Testing")
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"Testing samples: {len(test_ds)}")

    fold_paths = [config.MOE_FUSION_BRANCH_DIR / f"cv_safe_attention_final_fold{i}.pt" for i in range(1, 6)]

    all_fold_preds = []
    y_true_final = None
    video_ids_final = None

    for fold_path in fold_paths:
        model = DepressionPredictionModelAttention().to(device)
        model.load_state_dict(torch.load(fold_path, map_location=device))
        model.eval()

        preds, trues, ids = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                inputs = {name: batch[name].to(device) for name in MODALITY_ORDER}
                y_pred, _ = model(inputs)
                preds.extend(y_pred.cpu().numpy().tolist())
                trues.extend(batch["y"].numpy().tolist())
                ids.extend(batch["video_id"])

        all_fold_preds.append(preds)
        y_true_final = trues
        video_ids_final = ids
        print(f"Loaded: {fold_path.name}")

    all_fold_preds = np.array(all_fold_preds)
    ensemble_preds = all_fold_preds.mean(axis=0)

    df = pd.DataFrame({
        "sample_id": video_ids_final,
        "true_label": y_true_final,
        "prediction": ensemble_preds,
    })

    out_path = config.MOE_FUSION_BRANCH_DIR / "test_predictions_attention_final.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    mae = (df["prediction"] - df["true_label"]).abs().mean()
    print(f"MAE check: {mae:.4f}  (should be close to 7.33)")


if __name__ == "__main__":
    main()