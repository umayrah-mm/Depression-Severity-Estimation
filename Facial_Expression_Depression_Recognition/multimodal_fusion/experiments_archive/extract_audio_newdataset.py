"""
extract_audio_newdataset.py

Adapted from extract_audio.py for the 2 new videos (0001.MP4, 0006.MP4).
Same moviepy-based extraction, same sample rate/format settings as your
original pipeline - identical logic, just pointed at the new dataset's
video files.

Input:
    C:\\Users\\HP\\Desktop\\Dataset\\train\\0001.MP4
    C:\\Users\\HP\\Desktop\\Dataset\\test\\0006.MP4

Output:
    C:\\Users\\HP\\Desktop\\NewDataset_processed\\audio\\0001.wav
    C:\\Users\\HP\\Desktop\\NewDataset_processed\\audio\\0006.wav

Run:
    python extract_audio_newdataset.py
"""

from pathlib import Path

import config

try:
    from moviepy import VideoFileClip  # moviepy 2.x style
except ImportError:
    from moviepy.editor import VideoFileClip  # moviepy 1.x style


RAW_ROOT = Path(r"C:\Users\HP\Desktop\Dataset")
AUDIO_OUTPUT_DIR = Path(r"C:\Users\HP\Desktop\NewDataset_processed\audio")

# video_id -> (subfolder, actual filename)
VIDEOS = {
    "0001": ("train", "0001.MP4"),
    "0006": ("test", "0006.MP4"),
}


def main():
    AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for video_id, (subfolder, filename) in VIDEOS.items():
        video_path = RAW_ROOT / subfolder / filename
        audio_path = AUDIO_OUTPUT_DIR / f"{video_id}.wav"

        print(f"Processing {video_path} -> {audio_path}")

        if not video_path.exists():
            print(f"  MISSING: {video_path}")
            continue

        clip = None
        try:
            clip = VideoFileClip(str(video_path))

            if clip.audio is None:
                print(f"  No audio track found for {video_id}")
                continue

            clip.audio.write_audiofile(
                str(audio_path),
                fps=config.AUDIO_SAMPLE_RATE,
                nbytes=2,
                codec="pcm_s16le",
                logger=None,
            )
            print(f"  Saved: {audio_path}")

        except Exception as e:
            print(f"  ERROR: {e}")

        finally:
            if clip is not None:
                clip.close()

    total_wav_files = len(list(AUDIO_OUTPUT_DIR.glob("*.wav")))
    print(f"\nTotal .wav files now in {AUDIO_OUTPUT_DIR}: {total_wav_files}  (expect 2)")


if __name__ == "__main__":
    main()