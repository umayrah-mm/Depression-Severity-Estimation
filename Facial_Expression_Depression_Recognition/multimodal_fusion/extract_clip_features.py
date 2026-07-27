"""
extract_clip_features.py

Extracts visual semantic features from face frames using a FROZEN,
pretrained CLIP image encoder (ViT-B-32).

"Frozen" means CLIP's weights are never updated. We only ever run images
THROUGH the network (forward pass, under torch.no_grad()) to get a
representation of what the network already knows how to see - we never
train it. This is Green-AI friendly: we reuse a large model's existing
knowledge instead of paying the enormous cost of training or fine-tuning
it ourselves.

Process per video:
    1. Look inside FRAMES_ROOT/<split>/<video_id>/
    2. Pick MAX_FRAMES_FOR_CLIP evenly-spaced frames
    3. Run each frame through frozen CLIP -> 512-dim vector per frame
    4. Average the frame vectors -> one 512-dim vector per video

Input:
    - config.LABELS_PATH                (video, split, BDI-II)
    - config.FRAMES_ROOT / <split> / <video_id> / frame_*.jpg

Output:
    - config.CLIP_BRANCH_DIR / clip_features.npy    shape (N, 512)
    - config.CLIP_BRANCH_DIR / clip_sample_ids.npy  shape (N,)
"""

import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image
from tqdm import tqdm

import config


def load_frozen_clip():
    """
    Loads the pretrained ViT-B-32 CLIP model and freezes it.
    Returns: (model, preprocess_transform, device)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = model.to(device)
    model.eval()  # inference mode (disables dropout etc.)

    # Freeze: make sure no gradients are ever computed for CLIP's weights.
    for param in model.parameters():
        param.requires_grad = False

    return model, preprocess, device


def pick_evenly_spaced_frames(frame_files, max_frames):
    """
    Given a sorted list of frame filenames, pick up to max_frames of them,
    evenly spaced across the whole video. If the video has fewer frames
    than max_frames, just use all of them.
    """
    n = len(frame_files)
    if n <= max_frames:
        return frame_files

    indices = np.linspace(0, n - 1, max_frames).astype(int)
    return [frame_files[i] for i in indices]


def extract_one_video(video_dir, model, preprocess, device, max_frames):
    """
    Returns a single 512-dim numpy vector for one video, or None if
    no usable frames were found.
    """
    if not video_dir.exists():
        return None

    frame_files = sorted(
        f for f in video_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not frame_files:
        return None

    selected = pick_evenly_spaced_frames(frame_files, max_frames)

    frame_embeddings = []
    with torch.no_grad():  # CRITICAL: no gradients, no training, pure inference
        for frame_path in selected:
            try:
                img = Image.open(frame_path).convert("RGB")
                img_tensor = preprocess(img).unsqueeze(0).to(device)  # shape (1, 3, 224, 224)
                embedding = model.encode_image(img_tensor)            # shape (1, 512)
                frame_embeddings.append(embedding.squeeze(0).cpu().numpy())
            except Exception:
                continue

    if not frame_embeddings:
        return None

    # Average pool across all frame embeddings -> one 512-dim vector.
    video_embedding = np.mean(np.stack(frame_embeddings, axis=0), axis=0)
    return video_embedding.astype(np.float32)


def main():
    config.ensure_output_dirs()

    df = pd.read_csv(config.LABELS_PATH)
    print(f"Loaded {len(df)} rows from labels.csv")

    model, preprocess, device = load_frozen_clip()

    all_features = []
    all_sample_ids = []
    missing_frames = []
    errors = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting CLIP features"):
        video_id = row["video"]
        split = row["split"]
        video_dir = config.FRAMES_ROOT / split / video_id

        try:
            embedding = extract_one_video(
                video_dir, model, preprocess, device, config.MAX_FRAMES_FOR_CLIP
            )
        except Exception as e:
            errors.append((video_id, str(e)))
            continue

        if embedding is None:
            missing_frames.append(video_id)
            continue

        all_features.append(embedding)
        all_sample_ids.append(video_id)

    features_array = np.stack(all_features, axis=0)  # shape (N, 512)
    sample_ids_array = np.array(all_sample_ids)

    np.save(config.CLIP_BRANCH_DIR / "clip_features.npy", features_array)
    np.save(config.CLIP_BRANCH_DIR / "clip_sample_ids.npy", sample_ids_array)

    print("\n================ SUMMARY ================")
    print(f"Successfully extracted : {len(all_sample_ids)}")
    print(f"Missing/empty frames   : {len(missing_frames)}")
    print(f"Errors                 : {len(errors)}")
    print(f"Feature vector shape   : {features_array.shape}  (samples x 512)")

    if missing_frames:
        print(f"\nFirst 10 missing-frame video_ids:")
        for vid in missing_frames[:10]:
            print(f"  {vid}")

    if errors:
        print(f"\nFirst 10 errors:")
        for vid, msg in errors[:10]:
            print(f"  {vid}: {msg}")

    nan_count = np.isnan(features_array).sum()
    inf_count = np.isinf(features_array).sum()
    print(f"\nNaN values in features : {nan_count}")
    print(f"Inf values in features  : {inf_count}")

    print(f"\nSaved to: {config.CLIP_BRANCH_DIR}")


if __name__ == "__main__":
    main()