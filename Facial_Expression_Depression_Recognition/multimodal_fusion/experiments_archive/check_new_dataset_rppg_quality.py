"""
check_new_dataset_rppg_quality.py

Checks rPPG signal quality (same PSD-based Q score used in
compute_rppg_quality.py) for the 2 new dataset videos, so we know
whether their HR/HRV numbers meet the same quality bar your original
pipeline used (Q >= 2.0), before trusting them in a prediction.
"""

import cv2
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, welch

FRAMES_ROOT = Path(r"C:\Users\HP\Desktop\NewDataset_processed\processed_frames")

VIDEOS = {
    "0001": "Train",
    "0006": "Test",
}

FPS = 3
LOWCUT = 0.6
HIGHCUT = 1.4
Q_BAND_LOW = 0.6
Q_BAND_HIGH = 4.0


def bandpass_filter(signal, fs=FPS, low=LOWCUT, high=HIGHCUT, order=3):
    nyquist = 0.5 * fs
    low = low / nyquist
    high = min(high / nyquist, 0.99)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def compute_quality(signal):
    signal = signal - np.mean(signal)
    filtered = bandpass_filter(signal)
    freqs, power = welch(filtered, fs=FPS, nperseg=min(256, len(filtered)))

    band = (freqs >= Q_BAND_LOW) & (freqs <= Q_BAND_HIGH)
    power_in_band = np.trapezoid(power[band], freqs[band]) if band.any() else 0
    total_power = np.trapezoid(power, freqs)
    power_outside_band = total_power - power_in_band

    if power_outside_band <= 1e-10:
        return None

    return power_in_band / power_outside_band


def main():
    for video_id, split_name in VIDEOS.items():
        video_dir = FRAMES_ROOT / split_name / video_id
        frame_paths = sorted(video_dir.glob("*.jpg"))

        signal = []
        for f in frame_paths:
            img = cv2.imread(str(f))
            if img is None:
                continue
            h, w, _ = img.shape
            roi = img[int(0.2 * h):int(0.8 * h), int(0.2 * w):int(0.8 * w)]
            signal.append(roi[:, :, 1].mean())

        signal = np.array(signal, dtype=np.float32)
        Q = compute_quality(signal)

        if Q is None:
            print(f"{video_id}: could not compute quality")
        else:
            verdict = "PASSES" if Q >= 2.0 else "FAILS"
            print(f"{video_id}: Q={Q:.3f}  {verdict} the 2.0 threshold")


if __name__ == "__main__":
    main()