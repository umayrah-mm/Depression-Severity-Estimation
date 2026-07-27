"""
extract_audio.py

Extracts the audio track from every raw AVEC2014 video and saves it as a
mono 16kHz WAV file. This is a one-time preprocessing step: run it once,
and every later script (openSMILE, librosa) reads the cached .wav files
instead of re-decoding video every time.

Input:
    - config.LABELS_PATH        (video, split, BDI-II)
    - config.RAW_VIDEO_ROOT / <split> / <video_id>.mp4

Output:
    - config.AUDIO_DIR / <video_id>.wav   (one per sample)
"""

import pandas as pd
from tqdm import tqdm

import config

# moviepy changed its import path between version 1.x and 2.x.
# This try/except lets the script work regardless of which version is installed.
try:
    from moviepy import VideoFileClip  # moviepy 2.x style
except ImportError:
    from moviepy.editor import VideoFileClip  # moviepy 1.x style


OVERWRITE = False  # set to True if you want to re-extract audio that already exists


def extract_one(video_id: str, split: str) -> str:
    """
    Extracts audio for a single sample.

    Returns one of: "ok", "skipped_exists", "missing_video", "no_audio_track", "error:<message>"
    """
    raw_folder_name = config.CSV_SPLIT_TO_RAW_FOLDER[split]
    video_path = config.RAW_VIDEO_ROOT / raw_folder_name / f"{video_id}.mp4"
    audio_path = config.AUDIO_DIR / f"{video_id}.wav"

    if audio_path.exists() and not OVERWRITE:
        return "skipped_exists"

    if not video_path.exists():
        return "missing_video"

    clip = None
    try:
        clip = VideoFileClip(str(video_path))

        if clip.audio is None:
            return "no_audio_track"

        clip.audio.write_audiofile(
            str(audio_path),
            fps=config.AUDIO_SAMPLE_RATE,
            nbytes=2,          # 16-bit samples, standard for speech features
            codec="pcm_s16le", # uncompressed WAV encoding
            logger=None,       # silence moviepy's own progress bar (tqdm handles ours)
        )
        return "ok"

    except Exception as e:
        return f"error:{e}"

    finally:
        # Always release the file handle, even if something went wrong.
        if clip is not None:
            clip.close()


def main():
    config.ensure_output_dirs()

    df = pd.read_csv(config.LABELS_PATH)
    print(f"Loaded {len(df)} rows from labels.csv")

    results = {"ok": 0, "skipped_exists": 0, "missing_video": 0, "no_audio_track": 0}
    errors = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting audio"):
        video_id = row["video"]
        split = row["split"]

        status = extract_one(video_id, split)

        if status.startswith("error:"):
            errors.append((video_id, status))
        else:
            results[status] = results.get(status, 0) + 1

    print("\n================ SUMMARY ================")
    print(f"Successfully extracted : {results.get('ok', 0)}")
    print(f"Already existed (skip) : {results.get('skipped_exists', 0)}")
    print(f"Missing video file     : {results.get('missing_video', 0)}")
    print(f"Video has no audio     : {results.get('no_audio_track', 0)}")
    print(f"Errors                 : {len(errors)}")

    if errors:
        print("\nFirst 10 errors:")
        for video_id, msg in errors[:10]:
            print(f"  {video_id}: {msg}")

    total_wav_files = len(list(config.AUDIO_DIR.glob("*.wav")))
    print(f"\nTotal .wav files now in {config.AUDIO_DIR}: {total_wav_files}")


if __name__ == "__main__":
    main()