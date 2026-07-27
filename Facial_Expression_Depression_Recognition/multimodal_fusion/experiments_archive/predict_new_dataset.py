"""
predict_new_dataset.py

Final step: loads the 4 extracted feature sets for the 2 new videos
(0001, 0006), normalizes them using your ORIGINAL training statistics
(never recomputed from these 2 new samples), and runs them through your
best trained model to get predicted BDI-II severity scores.

CRITICAL: normalization uses config.ALIGNED_DATA_DIR/normalization_stats.npz
- the exact same mean/std computed from your original ~240 AVEC2014
training samples. This is what makes the new videos' features
comparable to what the model actually learned from.

Run:
    python predict_new_dataset.py
"""

from pathlib import Path

import numpy as np
import torch

import config
from models_attention import DepressionPredictionModelAttention, MODALITY_ORDER

NEW_DATA_DIR = Path(r"C:\Users\HP\Desktop\NewDataset_processed")


def load_new_features():
    """
    Loads and aligns the 4 modality feature sets for the 2 new videos,
    by video_id string - same explicit alignment discipline as dataset.py,
    never assuming row order matches across files.
    """
    visual_X = np.load(NEW_DATA_DIR / "visual_embeddings_new.npy")
    visual_ids = np.load(NEW_DATA_DIR / "visual_sample_ids_new.npy", allow_pickle=True)

    clip_X = np.load(NEW_DATA_DIR / "clip_features_new.npy")
    clip_ids = np.load(NEW_DATA_DIR / "clip_sample_ids_new.npy", allow_pickle=True)

    smile_X = np.load(NEW_DATA_DIR / "opensmile_features_new.npy")
    smile_ids = np.load(NEW_DATA_DIR / "opensmile_sample_ids_new.npy", allow_pickle=True)

    import pandas as pd
    rppg_df = pd.read_csv(NEW_DATA_DIR / "rppg_features_new.csv", dtype={"video_id": str})
    rppg_df = rppg_df.set_index("video_id", drop=False)

    rppg_columns = ["HR", "RR", "SDNN", "RMSSD", "LF", "HF", "LF_HF", "sample_entropy", "dfa_alpha"]

    # Common video_ids across all 4 sources - should be ['0001', '0006']
    id_sets = [set(visual_ids), set(clip_ids), set(smile_ids), set(rppg_df.index)]
    common_ids = sorted(set.intersection(*id_sets))
    print(f"Common video_ids across all 4 modalities: {common_ids}")

    if len(common_ids) != 2:
        raise ValueError(f"Expected 2 common video_ids, found {len(common_ids)}. Check extraction outputs.")

    visual_map = {vid: i for i, vid in enumerate(visual_ids)}
    clip_map = {vid: i for i, vid in enumerate(clip_ids)}
    smile_map = {vid: i for i, vid in enumerate(smile_ids)}

    visual_aligned = np.stack([visual_X[visual_map[v]] for v in common_ids])
    clip_aligned = np.stack([clip_X[clip_map[v]] for v in common_ids])
    smile_aligned = np.stack([smile_X[smile_map[v]] for v in common_ids])
    rppg_aligned = rppg_df.loc[common_ids, rppg_columns].to_numpy(dtype=np.float32)

    return {
        "visual": visual_aligned,
        "rppg": rppg_aligned,
        "clip": clip_aligned,
        "smile": smile_aligned,
    }, common_ids


def normalize_with_original_stats(features_dict):
    """
    Applies your ORIGINAL training mean/std to the new videos' features.
    NEVER computes new statistics from these 2 samples.
    """
    stats_path = config.ALIGNED_DATA_DIR / "normalization_stats.npz"
    stats = np.load(stats_path)
    print(f"\nLoaded original training normalization stats from: {stats_path}")

    normalized = {}
    for modality in ["visual", "rppg", "clip", "smile"]:
        mean = stats[f"{modality}_X_mean"]
        std = stats[f"{modality}_X_std"]
        X = features_dict[modality]

        X_norm = (X - mean) / std
        normalized[modality] = X_norm

        print(f"  {modality}: raw range [{X.min():.2f}, {X.max():.2f}]  ->  "
              f"normalized range [{X_norm.min():.2f}, {X_norm.max():.2f}]")

    return normalized


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("=== Loading and aligning new dataset features ===")
    features_dict, video_ids = load_new_features()

    print("\n=== Normalizing with ORIGINAL training statistics ===")
    normalized = normalize_with_original_stats(features_dict)

    # ---- Load your best trained model ----
    model_path = config.MOE_FUSION_BRANCH_DIR / "cv_safe_attention_final_fold1.pt"
    print(f"\n=== Loading model: {model_path} ===")

    model = DepressionPredictionModelAttention().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ---- Build input batch ----
    batch = {
        name: torch.tensor(normalized[name], dtype=torch.float32).to(device)
        for name in MODALITY_ORDER
    }

    with torch.no_grad():
        predictions, attn_weights = model(batch)

    print("\n================ PREDICTIONS ================")
    for i, video_id in enumerate(video_ids):
        print(f"Video {video_id}: predicted BDI-II score = {predictions[i].item():.2f}")

    print("\nNOTE: these predictions are on a DIFFERENT depression rating scale")
    print("(your supervisor's dataset uses HAMD, your model predicts BDI-II).")
    print("Treat this as an exploratory pipeline check, not a validated accuracy result.")


if __name__ == "__main__":
    main()