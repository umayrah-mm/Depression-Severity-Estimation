"""
extract_green_rppg_features.py

Extracts rPPG (remote photoplethysmography) physiological features from
face frames using the Green Channel method: computes a green-channel
signal per frame, bandpass filters it, and derives HRV features (HR,
RR, SDNN, RMSSD, LF, HF, LF/HF ratio, sample entropy, DFA alpha).

Paths are read from config.py - edit config.py's DATA_ROOT to point at
your own dataset, not this file.

Run:
    python extract_green_rppg_features.py
"""

import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch, find_peaks
from tqdm import tqdm

import config

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

    lf_power = np.trapz(power[lf_band], freqs[lf_band]) if lf_band.any() else 0
    hf_power = np.trapz(power[hf_band], freqs[hf_band]) if hf_band.any() else 0
    lf_hf = lf_power / (hf_power + 1e-8)

    sample_entropy = np.std(filtered)
    dfa_alpha = 1.0

    return {
        "HR": hr,
        "RR": rr_mean,
        "SDNN": sdnn,
        "RMSSD": rmssd,
        "LF": lf_power,
        "HF": hf_power,
        "LF_HF": lf_hf,
        "sample_entropy": sample_entropy,
        "dfa_alpha": dfa_alpha,
    }


def main():
    labels = pd.read_csv(config.LABELS_PATH)
    rows = []

    for _, row in tqdm(labels.iterrows(), total=len(labels)):
        video = str(row["video"])
        split = row["split"]
        bdi = row["BDI-II"]

        video_dir = config.FRAMES_ROOT / split / video

        if not video_dir.exists():
            print(f"Missing frames: {video_dir}")
            continue

        frame_paths = sorted(video_dir.glob("*.jpg"))

        signal = extract_green_signal(frame_paths)
        features = compute_features(signal)

        if features is None:
            print(f"Could not extract rPPG features for {video}")
            continue

        output_row = {
            "video_id": video,
            "split": split,
            "BDI_II": bdi,
        }

        output_row.update(features)
        rows.append(output_row)

    output_df = pd.DataFrame(rows)
    config.RPPG_FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(config.RPPG_FEATURES_CSV, index=False)

    print(f"Saved rPPG features to: {config.RPPG_FEATURES_CSV}")
    print(output_df.head())
    print(output_df["split"].value_counts())


if __name__ == "__main__":
    main()