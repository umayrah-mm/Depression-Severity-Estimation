"""
extract_rppg_newdataset.py

Adapted from your original rPPG extraction script for the 2 new videos
(0001, 0006). Same green-channel signal extraction, same bandpass
filter, same HRV feature computation logic - identical to your original
pipeline, just pointed at the new dataset's frame folders and saving to
a new location.

Input:
    C:\\Users\\HP\\Desktop\\NewDataset_processed\\processed_frames\\Train\\0001\\*.jpg
    C:\\Users\\HP\\Desktop\\NewDataset_processed\\processed_frames\\Test\\0006\\*.jpg

Output:
    C:\\Users\\HP\\Desktop\\NewDataset_processed\\rppg_features_new.csv

Run:
    python extract_rppg_newdataset.py
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from scipy.signal import butter, filtfilt, welch, find_peaks

FRAMES_ROOT = Path(r"C:\Users\HP\Desktop\NewDataset_processed\processed_frames")
OUTPUT_CSV = Path(r"C:\Users\HP\Desktop\NewDataset_processed\rppg_features_new.csv")

VIDEOS = {
    "0001": "Train",
    "0006": "Test",
}

FPS = 3
LOWCUT = 0.6
HIGHCUT = 1.4


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


def compute_features(signal):
    if len(signal) < 20:
        return None

    signal = signal - np.mean(signal)

    try:
        filtered = bandpass_filter(signal)
    except Exception:
        return None

    freqs, power = welch(filtered, fs=FPS, nperseg=min(256, len(filtered)))

    valid = (freqs >= 0.6) & (freqs <= 1.4)
    if valid.any():
        peak_freq = freqs[valid][np.argmax(power[valid])]
    else:
        peak_freq = 1.0
    hr = peak_freq * 60

    peaks, _ = find_peaks(filtered, distance=max(int(FPS * 0.4), 1))

    if len(peaks) > 2:
        rr_intervals = np.diff(peaks) / FPS
        rr_mean = np.mean(rr_intervals)
        sdnn = np.std(rr_intervals)
        if len(rr_intervals) > 2:
            rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
        else:
            rmssd = 0
    else:
        rr_mean = 0
        sdnn = 0
        rmssd = 0

    lf_band = (freqs >= 0.04) & (freqs < 0.15)
    hf_band = (freqs >= 0.15) & (freqs <= 0.4)

    lf_power = np.trapezoid(power[lf_band], freqs[lf_band]) if lf_band.any() else 0
    hf_power = np.trapezoid(power[hf_band], freqs[hf_band]) if hf_band.any() else 0
    lf_hf = lf_power / (hf_power + 1e-8)

    sample_entropy = np.std(filtered)
    dfa_alpha = 1.0

    return {
        "HR": hr, "RR": rr_mean, "SDNN": sdnn, "RMSSD": rmssd,
        "LF": lf_power, "HF": hf_power, "LF_HF": lf_hf,
        "sample_entropy": sample_entropy, "dfa_alpha": dfa_alpha,
    }


def main():
    rows = []

    for video_id, split_name in VIDEOS.items():
        video_dir = FRAMES_ROOT / split_name / video_id
        print(f"Processing {video_id} ({split_name}) from {video_dir}")

        frame_paths = sorted(video_dir.glob("*.jpg"))
        signal = extract_green_signal(frame_paths)
        features = compute_features(signal)

        if features is None:
            print(f"  Could not extract rPPG features for {video_id}")
            continue

        row = {"video_id": video_id}
        row.update(features)
        rows.append(row)
        print(f"  Extracted: {features}")

    output_df = pd.DataFrame(rows)
    output_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved: {OUTPUT_CSV}")
    print(output_df)


if __name__ == "__main__":
    main()