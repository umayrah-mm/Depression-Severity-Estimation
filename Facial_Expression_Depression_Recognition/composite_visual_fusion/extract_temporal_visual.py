"""
extract_temporal_visual.py

Builds the TEMPORAL member features of the composite visual doctor:
the WIN A05 regional stream for one video (audio stripped).

Per video:
    64-frame plan (32 segments x 2 frames, 100 ms pair gap)
    -> batched MTCNN -> eyes-only similarity align
    -> 5 region crops (full/eyes/mouth/cheek_left/cheek_right)
    -> frozen MobileNetV3-Small -> region features
    -> (64, 5, 576) + frame_valid (64,) + segment_index (64,)

Output: output/composite_visual_branch/temporal_visual/<video_id>.npz

Resume-safe: existing files are skipped.

Run:
    python extract_temporal_visual.py            # all splits
    python extract_temporal_visual.py --limit 3  # quick wiring check
"""

import argparse

import numpy as np
import pandas as pd
import torch

import config
from win_extraction import (
    atomic_npz,
    build_detector,
    extract_temporal_task,
    load_region_backbone,
    video_paths_index,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N videos (smoke test)")
    args = parser.parse_args()

    config.ensure_output_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    labels_df = pd.read_csv(config.LABELS_PATH)
    labels_df = labels_df.rename(columns={"video": "video_id"})
    raw_index = video_paths_index()

    video_ids = sorted(set(labels_df["video_id"].astype(str)))
    missing_on_disk = [v for v in video_ids if v not in raw_index]
    if missing_on_disk:
        print(
            f"WARNING: {len(missing_on_disk)} labels.csv videos not found raw; skipping them.")
    video_ids = [v for v in video_ids if v in raw_index]
    if args.limit is not None:
        video_ids = video_ids[: args.limit]
    print(f"Videos to process: {len(video_ids)}")

    detector = build_detector(device)
    backbone = load_region_backbone(device)

    done, skipped, failed, no_face = 0, 0, 0, 0
    for i, video_id in enumerate(video_ids):
        out_path = config.TEMPORAL_VISUAL_DIR / f"{video_id}.npz"
        if out_path.exists():
            skipped += 1
            continue
        print(f"  [{i + 1}/{len(video_ids)}] {video_id}", end="\r")
        try:
            feats, valid, timestamps, segments = extract_temporal_task(
                raw_index[video_id], detector, backbone, device
            )
            atomic_npz(out_path, {
                "region_features": feats.astype(np.float32),
                "frame_valid": valid.astype(bool),
                "frame_timestamps_s": timestamps.astype(np.float32),
                "segment_index": segments.astype(np.int16),
                "region_names": np.asarray(config.REGION_NAMES),
                "video_id": np.asarray(video_id),
            })
            if not valid.any():
                no_face += 1
            done += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"\n    ERROR {video_id}: {exc}")

    print()
    print("=============== TEMPORAL EXTRACTION SUMMARY ===============")
    print(f"Extracted this run : {done}   (no valid frames: {no_face})")
    print(f"Skipped (existing) : {skipped}")
    print(f"Failed             : {failed}")
    print(f"Output dir         : {config.TEMPORAL_VISUAL_DIR}")

    if failed:
        print("Re-run this script after fixing failures; it resumes where it left off.")


if __name__ == "__main__":
    main()
