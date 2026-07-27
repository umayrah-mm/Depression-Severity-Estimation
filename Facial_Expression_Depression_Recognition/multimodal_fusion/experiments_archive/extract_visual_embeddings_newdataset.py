"""
extract_visual_embeddings_newdataset.py

Adapted from extract_visual_embeddings.py for the 2 new videos from your
supervisor's dataset (0001 = train, 0006 = test). Uses the EXACT SAME
frozen MobileNetV3-Small extractor and region/pooling logic from
LightFusionNet/features.py, imported directly - not reimplemented - so
the numbers this produces are computed the identical way your original
training data was.

Input:
    C:\\Users\\HP\\Desktop\\NewDataset_processed\\processed_frames\\Train\\0001\\*.jpg
    C:\\Users\\HP\\Desktop\\NewDataset_processed\\processed_frames\\Test\\0006\\*.jpg

Output:
    C:\\Users\\HP\\Desktop\\NewDataset_processed\\visual_embeddings_new.npy   shape (2, 2304)
    C:\\Users\\HP\\Desktop\\NewDataset_processed\\visual_sample_ids_new.npy   shape (2,)

Run:
    python extract_visual_embeddings_newdataset.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image

import config

sys.path.insert(0, str(config.LIGHTFUSIONNET_CODE_DIR))

from dataset import IMAGE_TRANSFORM  # noqa: E402
from features import (  # noqa: E402
    select_expressive_frames,
    extract_multi_region_features,
    enhanced_weighted_pooling,
)

FRAMES_ROOT = Path(r"C:\Users\HP\Desktop\NewDataset_processed\processed_frames")
OUTPUT_DIR = Path(r"C:\Users\HP\Desktop\NewDataset_processed")

# video_id -> split folder name (matches prepare_new_dataset_frames.py's output)
VIDEOS = {
    "0001": "Train",
    "0006": "Test",
}

MAX_EXPRESSIVE_FRAMES = 100


def build_feature_extractor():
    full_model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )
    extractor = nn.Sequential(full_model.features, full_model.avgpool)
    for param in extractor.parameters():
        param.requires_grad = False
    extractor.eval()
    return extractor


def load_frames_as_tensor(video_dir: Path, max_frames=500):
    """
    Loads all frame images from a folder, applies the SAME IMAGE_TRANSFORM
    used during original training (resize 224x224, ImageNet normalize),
    and stacks them into a single (N, C, H, W) tensor.
    """
    frame_files = sorted(
        f for f in video_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )[:max_frames]

    frames = []
    for f in frame_files:
        img = Image.open(f).convert("RGB")
        img_tensor = IMAGE_TRANSFORM(img)
        frames.append(img_tensor)

    if not frames:
        raise ValueError(f"No usable frames found in {video_dir}")

    return torch.stack(frames)  # (N, 3, 224, 224)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    extractor = build_feature_extractor().to(device)
    print("Loaded frozen MobileNetV3-Small extractor.")

    all_features = []
    all_ids = []

    for video_id, split_name in VIDEOS.items():
        video_dir = FRAMES_ROOT / split_name / video_id
        print(f"\nProcessing {video_id} ({split_name}) from {video_dir}")

        frames = load_frames_as_tensor(video_dir)
        print(f"  Loaded {frames.shape[0]} frames, shape {tuple(frames.shape)}")

        if frames.shape[0] > MAX_EXPRESSIVE_FRAMES:
            frames, _ = select_expressive_frames(frames, extractor, device, MAX_EXPRESSIVE_FRAMES)
            print(f"  Selected {frames.shape[0]} most expressive frames")

        region_feats = extract_multi_region_features(frames, extractor, device)
        pooled = enhanced_weighted_pooling(region_feats)  # (2304,)

        print(f"  Pooled descriptor shape: {tuple(pooled.shape)}  (expect (2304,))")

        all_features.append(pooled.numpy())
        all_ids.append(video_id)

    features_array = np.stack(all_features, axis=0)  # (2, 2304)
    ids_array = np.array(all_ids)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "visual_embeddings_new.npy", features_array)
    np.save(OUTPUT_DIR / "visual_sample_ids_new.npy", ids_array)

    print("\n================ SUMMARY ================")
    print(f"Feature array shape: {features_array.shape}  (expect (2, 2304))")
    print(f"Sample IDs: {ids_array}")

    nan_count = np.isnan(features_array).sum()
    inf_count = np.isinf(features_array).sum()
    print(f"NaN values: {nan_count}")
    print(f"Inf values: {inf_count}")

    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()