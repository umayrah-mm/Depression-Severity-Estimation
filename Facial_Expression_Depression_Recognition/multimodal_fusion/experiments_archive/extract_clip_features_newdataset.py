"""
extract_clip_features_newdataset.py

Adapted from extract_clip_features.py for the 2 new videos (0001, 0006).
Same frozen CLIP ViT-B-32 model, same evenly-spaced frame sampling, same
average pooling across frames - identical logic to your original
pipeline, just pointed at the new dataset's frame folders.

Input:
    C:\\Users\\HP\\Desktop\\NewDataset_processed\\processed_frames\\Train\\0001\\*.jpg
    C:\\Users\\HP\\Desktop\\NewDataset_processed\\processed_frames\\Test\\0006\\*.jpg

Output:
    C:\\Users\\HP\\Desktop\\NewDataset_processed\\clip_features_new.npy    shape (2, 512)
    C:\\Users\\HP\\Desktop\\NewDataset_processed\\clip_sample_ids_new.npy  shape (2,)

Run:
    python extract_clip_features_newdataset.py
"""

from pathlib import Path

import numpy as np
import torch
import open_clip
from PIL import Image

import config

FRAMES_ROOT = Path(r"C:\Users\HP\Desktop\NewDataset_processed\processed_frames")
OUTPUT_DIR = Path(r"C:\Users\HP\Desktop\NewDataset_processed")

VIDEOS = {
    "0001": "Train",
    "0006": "Test",
}


def load_frozen_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model, preprocess, device


def pick_evenly_spaced_frames(frame_files, max_frames):
    n = len(frame_files)
    if n <= max_frames:
        return frame_files
    indices = np.linspace(0, n - 1, max_frames).astype(int)
    return [frame_files[i] for i in indices]


def extract_one_video(video_dir, model, preprocess, device, max_frames):
    frame_files = sorted(
        f for f in video_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not frame_files:
        raise ValueError(f"No frames found in {video_dir}")

    selected = pick_evenly_spaced_frames(frame_files, max_frames)

    frame_embeddings = []
    with torch.no_grad():
        for frame_path in selected:
            img = Image.open(frame_path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            embedding = model.encode_image(img_tensor)
            frame_embeddings.append(embedding.squeeze(0).cpu().numpy())

    video_embedding = np.mean(np.stack(frame_embeddings, axis=0), axis=0)
    return video_embedding.astype(np.float32)


def main():
    model, preprocess, device = load_frozen_clip()

    all_features = []
    all_ids = []

    for video_id, split_name in VIDEOS.items():
        video_dir = FRAMES_ROOT / split_name / video_id
        print(f"\nProcessing {video_id} ({split_name}) from {video_dir}")

        embedding = extract_one_video(video_dir, model, preprocess, device, config.MAX_FRAMES_FOR_CLIP)
        print(f"  Embedding shape: {embedding.shape}  (expect (512,))")

        all_features.append(embedding)
        all_ids.append(video_id)

    features_array = np.stack(all_features, axis=0)
    ids_array = np.array(all_ids)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "clip_features_new.npy", features_array)
    np.save(OUTPUT_DIR / "clip_sample_ids_new.npy", ids_array)

    print("\n================ SUMMARY ================")
    print(f"Feature array shape: {features_array.shape}  (expect (2, 512))")
    print(f"Sample IDs: {ids_array}")

    nan_count = np.isnan(features_array).sum()
    inf_count = np.isinf(features_array).sum()
    print(f"NaN values: {nan_count}")
    print(f"Inf values: {inf_count}")

    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()