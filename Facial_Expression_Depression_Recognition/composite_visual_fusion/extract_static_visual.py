"""
extract_static_visual.py

Builds the STATIC member features of the composite visual doctor:
one mean-pooled whole-face MobileNetV3-Small embedding per video.

Input : raw AVEC2014 mp4s listed in labels.csv (read-only)
Output: output/composite_visual_branch/static_visual/<video_id>.npz
            features      float32 (576,)
            valid         bool     scalar
            video_id      str

Resume-safe: existing <video_id>.npz files are skipped.

Run:
    python extract_static_visual.py            # all splits
    python extract_static_visual.py --limit 3  # quick wiring check
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
from win_extraction import (
    atomic_npz,
    build_detector,
    extract_static_task,
    load_static_backbone,
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

    # All videos that appear in labels.csv, in deterministic order.
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
    backbone, weights = load_static_backbone(device)

    done, skipped, failed, invalid = 0, 0, 0, 0
    for i, video_id in enumerate(video_ids):
        out_path = config.STATIC_VISUAL_DIR / f"{video_id}.npz"
        if out_path.exists():
            skipped += 1
            continue
        print(f"  [{i + 1}/{len(video_ids)}] {video_id}", end="\r")
        try:
            emb, ok = extract_static_task(
                raw_index[video_id], detector, backbone, weights, device)
            atomic_npz(out_path, {
                "features": emb.astype(np.float32),
                "valid": np.asarray(ok, bool),
                "video_id": np.asarray(video_id),
            })
            if not ok:
                invalid += 1
            done += 1
        except Exception as exc:  # noqa: BLE001 - log and continue like the baseline extractor
            failed += 1
            print(f"\n    ERROR {video_id}: {exc}")

    print()
    print("================ STATIC EXTRACTION SUMMARY ================")
    print(f"Extracted this run : {done}   (invalid/zero vectors: {invalid})")
    print(f"Skipped (existing) : {skipped}")
    print(f"Failed             : {failed}")
    print(f"Output dir         : {config.STATIC_VISUAL_DIR}")

    if done + skipped == 0:
        print("Nothing processed - check paths in config.py.")
    if failed:
        print("Re-run this script after fixing failures; it resumes where it left off.")


if __name__ == "__main__":
    main()
