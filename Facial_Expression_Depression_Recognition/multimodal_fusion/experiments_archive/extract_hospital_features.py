"""
extract_hospital_features.py

Extracts and caches all 4 modality features for every video in a
Hospital_Data-shaped dataset (train/dev/test folders + label.csv).

Caches one .npz file PER VIDEO under <root>\\feature_cache\\, so if this
crashes or is interrupted partway through a large dataset, re-running
picks up where it left off instead of starting over - already-cached
videos are skipped.

After processing all videos, combines everything into one file:
    <root>\\hospital_features_combined.npz
containing visual_X, clip_X, rppg_X, smile_X, y (HAMD scores), split,
video_ids - ready for train_hospital.py to use directly.

Run:
    python extract_hospital_features.py --root "E:\\Data\\Hospital_Data"
    python extract_hospital_features.py --root "E:\\Data\\Hospital_Data" --limit 5
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import open_clip
import opensmile

import config
from hospital_data_loader import scan_hospital_data
from predict_hospital_dataset import (
    extract_frames_from_video,
    build_mobilenet_extractor,
    extract_visual,
    extract_clip,
    extract_rppg,
)

try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip


def extract_one_video(record, mobilenet, clip_model, clip_preprocess, smile, device, tmp_root):
    tmp_dir = tmp_root / record.video_id
    frame_paths = extract_frames_from_video(record.path, tmp_dir)

    visual = extract_visual(frame_paths, mobilenet, device)
    clip_feat = extract_clip(frame_paths, clip_model, clip_preprocess, device, config.MAX_FRAMES_FOR_CLIP)
    rppg_feat = extract_rppg(frame_paths)

    audio_path = tmp_dir / "audio.wav"
    clip_obj = VideoFileClip(str(record.path))
    if clip_obj.audio is None:
        clip_obj.close()
        return None
    clip_obj.audio.write_audiofile(str(audio_path), fps=config.AUDIO_SAMPLE_RATE, nbytes=2, codec="pcm_s16le", logger=None)
    smile_feat = smile.process_file(str(audio_path)).iloc[0].to_numpy(dtype=np.float32)
    clip_obj.close()

    return visual, rppg_feat, clip_feat, smile_feat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N videos (for testing)")
    args = parser.parse_args()

    root = Path(args.root)
    cache_dir = root / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = root / "_tmp_frames"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"\nScanning {root} ...")
    records = scan_hospital_data(root)
    if args.limit:
        records = records[: args.limit]
    print(f"Found {len(records)} videos with valid labels.")

    mobilenet = build_mobilenet_extractor().to(device)
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_model = clip_model.to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)

    n_skipped, n_extracted, n_failed = 0, 0, 0

    for i, record in enumerate(records):
        cache_path = cache_dir / f"{record.video_id}.npz"

        if cache_path.exists():
            n_skipped += 1
            continue

        print(f"[{i + 1}/{len(records)}] Extracting {record.video_id} ({record.split}) ...")
        try:
            result = extract_one_video(record, mobilenet, clip_model, clip_preprocess, smile, device, tmp_root)
            if result is None:
                print(f"  WARNING: no audio track, skipping {record.video_id}")
                n_failed += 1
                continue

            visual, rppg_feat, clip_feat, smile_feat = result
            np.savez(
                cache_path,
                visual=visual, rppg=rppg_feat, clip=clip_feat, smile=smile_feat,
                label=record.label, split=record.split, video_id=record.video_id,
            )
            n_extracted += 1
        except Exception as e:
            print(f"  ERROR on {record.video_id}: {e}")
            n_failed += 1

    print(f"\n================ EXTRACTION SUMMARY ================")
    print(f"Newly extracted : {n_extracted}")
    print(f"Already cached  : {n_skipped}")
    print(f"Failed          : {n_failed}")

    # ---- Combine every cached video into one file ----
    print(f"\nCombining all cached features from {cache_dir} ...")
    cached_files = sorted(cache_dir.glob("*.npz"))
    print(f"Found {len(cached_files)} cached video(s) total.")

    visual_X, rppg_X, clip_X, smile_X, y, split, video_ids = [], [], [], [], [], [], []
    for f in cached_files:
        data = np.load(f, allow_pickle=True)
        visual_X.append(data["visual"])
        rppg_X.append(data["rppg"])
        clip_X.append(data["clip"])
        smile_X.append(data["smile"])
        y.append(float(data["label"]))
        split.append(str(data["split"]))
        video_ids.append(str(data["video_id"]))

    combined_path = root / "hospital_features_combined.npz"
    np.savez(
        combined_path,
        visual_X=np.stack(visual_X), rppg_X=np.stack(rppg_X),
        clip_X=np.stack(clip_X), smile_X=np.stack(smile_X),
        y=np.array(y, dtype=np.float32), split=np.array(split), video_ids=np.array(video_ids),
    )
    print(f"Saved combined dataset to: {combined_path}")

    for s in ("train", "dev", "test"):
        count = sum(1 for sp in split if sp == s)
        print(f"  {s}: {count} samples")


if __name__ == "__main__":
    main()