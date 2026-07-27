"""
prepare_avec_frames.py

Extracts frames from raw AVEC2014 videos at a fixed FPS, resizes them,
and saves them into the folder structure the rest of the pipeline
expects (FRAMES_ROOT / <split> / <video_id> / frame_*.jpg).

Paths are read from config.py - edit config.py's DATA_ROOT and
RAW_VIDEO_ROOT to point at your own dataset, not this file.

Run:
    python prepare_avec_frames.py
"""

import cv2
from tqdm import tqdm

import config

SPLIT_MAP = {
    "train": "Training",
    "dev": "Development",
    "test": "Testing",
}

TARGET_FPS = 3
MAX_FRAMES = 500
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]


def extract_frames(video_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open: {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 30

    frame_interval = max(int(round(original_fps / TARGET_FPS)), 1)

    frame_id = 0
    saved_id = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_id % frame_interval == 0:
            frame = cv2.resize(frame, (224, 224))
            save_path = output_dir / f"frame_{saved_id:06d}.jpg"
            cv2.imwrite(str(save_path), frame)
            saved_id += 1

            if saved_id >= MAX_FRAMES:
                break

        frame_id += 1

    cap.release()
    print(f"{video_path.name}: saved {saved_id} frames")


def main():
    for original_split, new_split in SPLIT_MAP.items():
        input_split_dir = config.RAW_VIDEO_ROOT / original_split
        output_split_dir = config.FRAMES_ROOT / new_split

        if not input_split_dir.exists():
            print(f"Missing folder: {input_split_dir}")
            continue

        video_files = []
        for ext in VIDEO_EXTENSIONS:
            video_files.extend(input_split_dir.glob(f"*{ext}"))

        print(f"{original_split}: found {len(video_files)} videos")

        for video_path in tqdm(video_files, desc=f"Processing {original_split}"):
            video_name = video_path.stem
            output_dir = output_split_dir / video_name
            extract_frames(video_path, output_dir)


if __name__ == "__main__":
    main()