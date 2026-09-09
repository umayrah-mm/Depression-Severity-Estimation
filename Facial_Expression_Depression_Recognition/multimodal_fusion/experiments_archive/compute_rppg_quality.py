"""
compute_rppg_quality.py

Recomputes a Power Spectral Density (PSD)-based signal quality score for
each participant's rPPG signal, using the SAME already-saved frame files
your original rPPG extraction script used. Does NOT touch or overwrite
rppg_features.csv - saves a new file, rppg_quality.csv, with one quality
score per video_id.

What "quality" means here: a real heartbeat signal concentrates most of
its energy in a specific frequency band (roughly 36-240 beats per
minute). Noise (motion, lighting changes, camera artifacts) spreads
energy across all frequencies instead. The quality score Q compares
"energy inside the heartbeat band" to "energy everywhere else" - higher
Q means a cleaner, more believable heartbeat signal.

Formula (same as the original LightFusionNet paper):
    Q = power_in_heartbeat_band / (total_power - power_in_heartbeat_band)

Run:
    python compute_rppg_quality.py
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, welch
from tqdm import tqdm

FRAME_ROOT = Path(r"C:\Users\HP\Desktop\AVEC2014_processed\processed_frames")
LABELS_PATH = Path(r"C:\Users\HP\Desktop\AVEC2014_processed\labels.csv")
OUTPUT_CSV = Path(r"C:\Users\HP\Desktop\AVEC2014_processed\rppg_quality.csv")

FPS = 3
LOWCUT = 0.6
HIGHCUT = 1.4

# Physiologically plausible heart rate range, in Hz (36-240 bpm / 60)
Q_BAND_LOW = 0.6
Q_BAND_HIGH = 4.0


def bandpass_filter(signal, fs=FPS, low=LOWCUT, high=HIGHCUT, order=3):
    nyquist = 0.5 * fs
    low = low / nyquist
    high = high / nyquist
    if high >= 1:
        high = 0.99
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def extract_green_signal(frame_paths):
    signal = []
    for frame_path in frame_paths:
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        h, w, _ = img.shape
        y1, y2 = int(0.2 * h), int(0.8 * h)
        x1, x2 = int(0.2 * w), int(0.8 * w)
        roi = img[y1:y2, x1:x2]
        green_mean = roi[:, :, 1].mean()
        signal.append(green_mean)
    return np.array(signal, dtype=np.float32)


def compute_quality(signal):
    if len(signal) < 20:
        return None

    signal = signal - np.mean(signal)

    try:
        filtered = bandpass_filter(signal)
    except Exception:
        return None

    freqs, power = welch(filtered, fs=FPS, nperseg=min(256, len(filtered)))

    heartbeat_band = (freqs >= Q_BAND_LOW) & (freqs <= Q_BAND_HIGH)
    power_in_band = np.trapz(power[heartbeat_band], freqs[heartbeat_band]) if heartbeat_band.any() else 0
    total_power = np.trapz(power, freqs)
    power_outside_band = total_power - power_in_band

    if power_outside_band <= 1e-10:
        return None  # avoid divide-by-zero for a degenerate signal

    Q = power_in_band / power_outside_band
    return Q


def main():
    labels = pd.read_csv(LABELS_PATH)
    rows = []

    for _, row in tqdm(labels.iterrows(), total=len(labels)):
        video = str(row["video"])
        split = row["split"]

        video_dir = FRAME_ROOT / split / video
        if not video_dir.exists():
            print(f"Missing frames: {video_dir}")
            continue

        frame_paths = sorted(video_dir.glob("*.jpg"))
        signal = extract_green_signal(frame_paths)
        Q = compute_quality(signal)

        if Q is None:
            print(f"Could not compute quality for {video}")
            continue

        rows.append({"video_id": video, "split": split, "rppg_quality": Q})

    output_df = pd.DataFrame(rows)
    output_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"\nQuality score summary:")
    print(output_df["rppg_quality"].describe())

    n_below_2 = (output_df["rppg_quality"] < 2.0).sum()
    print(f"\nSamples with Q < 2.0 (paper's threshold for 'exclude as too noisy'): "
          f"{n_below_2} out of {len(output_df)}")


if __name__ == "__main__":
    main()