"""
prepare_avec_labels.py

Builds labels.csv from the raw AVEC2014 .mat label files, matching
each video to its BDI-II score by participant/session ID (since each
session has multiple videos - Freeform and Northwind tasks - sharing
one label).

Paths are read from config.py - edit config.py's DATA_ROOT and
RAW_VIDEO_ROOT to point at your own dataset, not this file.

Expects RAW_VIDEO_ROOT to contain a "label" subfolder with:
    train_label.mat, develop_label.mat, test_label.mat

Run:
    python prepare_avec_labels.py
"""

from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.io

import config

LABEL_ROOT = config.RAW_VIDEO_ROOT / "label"

SPLITS = {
    "train_label.mat": ("train", "Training"),
    "develop_label.mat": ("dev", "Development"),
    "test_label.mat": ("test", "Testing"),
}


def get_numeric_arrays(mat_data):
    arrays = {}

    for key, value in mat_data.items():
        if key.startswith("__"):
            continue

        arr = np.array(value).squeeze()

        if np.issubdtype(arr.dtype, np.number):
            arrays[key] = arr.flatten()

    return arrays


def get_session_id(video_path):
    """
    Example:
    203_1_Freeform_video.mp4   -> 203_1
    203_1_Northwind_video.mp4  -> 203_1
    """
    name = video_path.stem
    parts = name.split("_")
    return f"{parts[0]}_{parts[1]}"


def main():
    all_rows = []

    for mat_filename, (video_folder_name, split_name) in SPLITS.items():
        mat_path = LABEL_ROOT / mat_filename
        video_folder = config.RAW_VIDEO_ROOT / video_folder_name

        print("\n" + "=" * 80)
        print(f"Reading label file: {mat_path}")
        print(f"Matching videos from: {video_folder}")
        print("=" * 80)

        mat_data = scipy.io.loadmat(mat_path)
        numeric_arrays = get_numeric_arrays(mat_data)

        video_files = sorted(video_folder.glob("*.mp4"))

        print(f"Found {len(video_files)} videos in {video_folder_name}")
        print("Numeric arrays found inside .mat file:")

        for key, arr in numeric_arrays.items():
            print(f"  {key}: shape={arr.shape}, length={arr.size}, first_values={arr[:10]}")

        session_to_videos = defaultdict(list)

        for video_path in video_files:
            session_id = get_session_id(video_path)
            session_to_videos[session_id].append(video_path)

        session_ids = sorted(session_to_videos.keys())

        print(f"Found {len(session_ids)} unique sessions")

        label_array = None
        chosen_key = None

        for key, arr in numeric_arrays.items():
            if len(arr) == len(session_ids):
                label_array = arr
                chosen_key = key
                break

        if label_array is None:
            raise ValueError(
                f"Could not match labels in {mat_filename}. "
                f"Videos found: {len(video_files)}. "
                f"Unique sessions found: {len(session_ids)}. "
                f"Check printed .mat keys above."
            )

        print(f"Using '{chosen_key}' as the label array.")

        for session_id, label in zip(session_ids, label_array):
            videos_for_session = sorted(session_to_videos[session_id])

            for video_path in videos_for_session:
                all_rows.append({
                    "video": video_path.stem,
                    "split": split_name,
                    "BDI-II": float(label),
                })

    df = pd.DataFrame(all_rows)

    config.LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.LABELS_PATH, index=False)

    print("\nSaved labels CSV to:")
    print(config.LABELS_PATH)

    print("\nPreview:")
    print(df.head(10))

    print("\nSplit counts:")
    print(df["split"].value_counts())

    print("\nExample labels:")
    print(df.groupby("split").head(4))


if __name__ == "__main__":
    main()