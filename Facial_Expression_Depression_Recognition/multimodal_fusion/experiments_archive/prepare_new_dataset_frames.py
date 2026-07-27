"""
prepare_new_dataset_frames.py

Adapted from prepare_avec_frames.py, for the new dataset your supervisor
provided (2 videos: 0001.MP4 for train, 0006.MP4 for test).

Same extraction logic as your original AVEC2014 pipeline (3 FPS, resize
to 224x224, max 500 frames) - kept identical so these frames are
processed the same way your model was trained to expect. Output goes to
a completely SEPARATE folder from your original processed_frames, so
nothing in your existing dataset is touched or mixed up.

Run:
    python prepare_new_dataset_frames.py
"""

import cv2
from pathlib import Path

RAW_ROOT = Path(r"C:\Users\HP\Desktop\Dataset")
OUTPUT_ROOT = Path(r"C:\Users\HP\Desktop\NewDataset_processed\processed_frames")

# video_id -> (subfolder inside RAW_ROOT, output split name)
VIDEOS = {
    "0001": ("train", "Train"),
    "0006": ("test", "Test"),
}

TARGET_FPS = 3
MAX_FRAMES = 500


def extract_frames(video_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open: {video_path}")
        return 0

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
    return saved_id


def main():
    print(f"Looking for videos in: {RAW_ROOT}\n")

    for video_id, (subfolder, split_name) in VIDEOS.items():
        video_dir = RAW_ROOT / subfolder
        matches = list(video_dir.glob(f"{video_id}.*"))

        if not matches:
            print(f"MISSING: no file found for {video_id} in {video_dir}")
            continue

        video_path = matches[0]
        output_dir = OUTPUT_ROOT / split_name / video_id

        print(f"Processing {video_path} -> {output_dir}")
        n_saved = extract_frames(video_path, output_dir)
        print(f"  Saved {n_saved} frames\n")

    print("Done. Check the output folder to verify frame counts look reasonable.")


if __name__ == "__main__":
    main()