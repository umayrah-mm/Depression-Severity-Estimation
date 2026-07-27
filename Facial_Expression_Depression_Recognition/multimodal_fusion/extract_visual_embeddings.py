"""
extract_visual_embeddings.py

Re-runs the FROZEN MobileNetV3-Small visual feature pipeline from your
existing LightFusionNet code, but saves the raw 2304-dim pooled embedding
per video BEFORE it gets passed into SelectKBest + SVR/RandomForest.

This does NOT retrain or modify your existing visual model in any way -
it reuses the same frozen extractor and the same region/pooling logic
from LightFusionNet/utils.py, imported directly (not duplicated), so the
underlying numbers are guaranteed identical to what your original
pipeline computed internally.

Shape reasoning:
    MobileNetV3-Small (truncated after avgpool) outputs 576 features.
    4 facial regions (eyes, mouth, left_cheek, right_cheek) are processed
    separately: 4 x 576 = 2304.
    enhanced_weighted_pooling() concatenates all 4 region vectors into
    one 2304-dim descriptor per video.

Input:
    - config.LABELS_PATH
    - config.FRAMES_ROOT / <split> / <video_id> / frame_*.jpg

Output:
    - config.VISUAL_RAW_BRANCH_DIR / visual_embeddings.npy   shape (N, 2304)
    - config.VISUAL_RAW_BRANCH_DIR / visual_sample_ids.npy   shape (N,)
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm

import config

# Make LightFusionNet/dataset.py and LightFusionNet/utils.py importable.
sys.path.insert(0, str(config.LIGHTFUSIONNET_CODE_DIR))

from dataset import AVEC2014Dataset, IMAGE_TRANSFORM  # noqa: E402
from utils import (  # noqa: E402
    select_expressive_frames,
    extract_multi_region_features,
    enhanced_weighted_pooling,
)


def build_feature_extractor():
    """
    Identical to video_mobilenet.py's build_feature_extractor():
    frozen MobileNetV3-Small truncated after global average pooling.
    Output feature dimension: 576 per region.
    """
    full_model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )
    extractor = nn.Sequential(full_model.features, full_model.avgpool)
    for param in extractor.parameters():
        param.requires_grad = False
    extractor.eval()
    return extractor


def extract_split(dataset, extractor, device, max_expressive=100):
    """
    Same logic as video_mobilenet.py's precompute_features(), except we
    return the raw pooled embeddings instead of feeding them onward.
    """
    features, sample_ids = [], []
    errors = []

    for i in range(len(dataset)):
        frames, label, video_id = dataset[i]
        print(f"  Video {i + 1}/{len(dataset)}  ({video_id})", end="\r")

        try:
            if len(frames) > max_expressive:
                frames, _ = select_expressive_frames(frames, extractor, device, max_expressive)
            region_feats = extract_multi_region_features(frames, extractor, device)
            pooled = enhanced_weighted_pooling(region_feats)
            features.append(pooled.numpy())
            sample_ids.append(video_id)
        except Exception as e:
            errors.append((video_id, str(e)))

    print()
    return features, sample_ids, errors


def main():
    config.ensure_output_dirs()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    extractor = build_feature_extractor().to(device)
    total_params = sum(p.numel() for p in extractor.parameters())
    print(f"Feature extractor parameters: {total_params / 1e6:.2f} M (frozen)")

    all_features, all_sample_ids, all_errors = [], [], []

    for split in config.SPLIT_NAMES:
        print(f"\n=== Extracting raw visual embeddings: {split} ===")
        ds = AVEC2014Dataset(
            str(config.FRAMES_ROOT), str(config.LABELS_PATH), split, IMAGE_TRANSFORM, 500
        )
        feats, ids, errs = extract_split(ds, extractor, device)
        all_features.extend(feats)
        all_sample_ids.extend(ids)
        all_errors.extend(errs)

    features_array = np.stack(all_features, axis=0)  # shape (N, 2304)
    sample_ids_array = np.array(all_sample_ids)

    np.save(config.VISUAL_RAW_BRANCH_DIR / "visual_embeddings.npy", features_array)
    np.save(config.VISUAL_RAW_BRANCH_DIR / "visual_sample_ids.npy", sample_ids_array)

    print("\n================ SUMMARY ================")
    print(f"Successfully extracted : {len(all_sample_ids)}")
    print(f"Errors                 : {len(all_errors)}")
    print(f"Feature vector shape   : {features_array.shape}  (samples x 2304)")

    if all_errors:
        print("\nFirst 10 errors:")
        for vid, msg in all_errors[:10]:
            print(f"  {vid}: {msg}")

    nan_count = np.isnan(features_array).sum()
    inf_count = np.isinf(features_array).sum()
    print(f"\nNaN values in features : {nan_count}")
    print(f"Inf values in features  : {inf_count}")

    print(f"\nSaved to: {config.VISUAL_RAW_BRANCH_DIR}")


if __name__ == "__main__":
    main()