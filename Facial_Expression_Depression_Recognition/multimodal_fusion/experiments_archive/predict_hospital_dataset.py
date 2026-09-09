"""
predict_hospital_dataset.py

Generalized version of experiments_archive/predict_new_dataset.py:
scans a Hospital_Data-shaped folder (any number of videos, via
hospital_data_loader.py) instead of a hardcoded 2-video list, extracts
all 4 modalities for each video, normalizes with your ORIGINAL AVEC2014
training statistics, and predicts BDI-II-scale severity scores.

IMPORTANT: this dataset's labels are on the HAMD scale, not BDI-II.
Predictions here are exploratory - not directly comparable to HAMD
ground truth without a separate calibration step.

Run:
    python predict_hospital_dataset.py --root "E:\\path\\to\\Hospital_Data"
    python predict_hospital_dataset.py --root "E:\\path\\to\\Hospital_Data" --limit 3

Outputs:
    <root>\\predictions.csv   (video_id, split, true_HAMD, predicted_BDI_II)
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import open_clip
import opensmile
from PIL import Image

import config
from hospital_data_loader import scan_hospital_data

sys.path.insert(0, str(config.LIGHTFUSIONNET_CODE_DIR))
from dataset import IMAGE_TRANSFORM  # noqa: E402
from features import select_expressive_frames, extract_multi_region_features, enhanced_weighted_pooling  # noqa: E402
import torchvision.models as tv_models  # noqa: E402
import torch.nn as nn  # noqa: E402

from models_attention import DepressionPredictionModelAttention, MODALITY_ORDER  # noqa: E402

try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip


TARGET_FPS = 3
MAX_FRAMES = 500
MAX_EXPRESSIVE_FRAMES = 100


def extract_frames_from_video(video_path: Path, tmp_dir: Path):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(int(round(original_fps / TARGET_FPS)), 1)

    frame_id, saved_id = 0, 0
    saved_paths = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_id % frame_interval == 0:
            frame = cv2.resize(frame, (224, 224))
            save_path = tmp_dir / f"frame_{saved_id:06d}.jpg"
            cv2.imwrite(str(save_path), frame)
            saved_paths.append(save_path)
            saved_id += 1
            if saved_id >= MAX_FRAMES:
                break
        frame_id += 1
    cap.release()
    return saved_paths


def build_mobilenet_extractor():
    full_model = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    extractor = nn.Sequential(full_model.features, full_model.avgpool)
    for p in extractor.parameters():
        p.requires_grad = False
    extractor.eval()
    return extractor


def extract_visual(frame_paths, extractor, device):
    frames = torch.stack([IMAGE_TRANSFORM(Image.open(f).convert("RGB")) for f in frame_paths])
    if frames.shape[0] > MAX_EXPRESSIVE_FRAMES:
        frames, _ = select_expressive_frames(frames, extractor, device, MAX_EXPRESSIVE_FRAMES)
    region_feats = extract_multi_region_features(frames, extractor, device)
    return enhanced_weighted_pooling(region_feats).numpy()  # (2304,)


def extract_clip(frame_paths, clip_model, preprocess, device, max_frames):
    n = len(frame_paths)
    selected = frame_paths if n <= max_frames else [frame_paths[i] for i in np.linspace(0, n - 1, max_frames).astype(int)]
    embeddings = []
    with torch.no_grad():
        for f in selected:
            img = preprocess(Image.open(f).convert("RGB")).unsqueeze(0).to(device)
            embeddings.append(clip_model.encode_image(img).squeeze(0).cpu().numpy())
    return np.mean(np.stack(embeddings), axis=0).astype(np.float32)  # (512,)


def extract_rppg(frame_paths):
    from scipy.signal import butter, filtfilt, welch, find_peaks
    from scipy.integrate import trapezoid

    def bandpass(signal, fs=3, low=0.6, high=1.4, order=3):
        nyq = 0.5 * fs
        low, high = low / nyq, min(high / nyq, 0.99)
        b, a = butter(order, [low, high], btype="band")
        return filtfilt(b, a, signal)

    signal = []
    for f in frame_paths:
        img = cv2.imread(str(f))
        h, w, _ = img.shape
        roi = img[int(0.2 * h):int(0.8 * h), int(0.2 * w):int(0.8 * w)]
        signal.append(roi[:, :, 1].mean())
    signal = np.array(signal, dtype=np.float32)
    signal = signal - np.mean(signal)
    filtered = bandpass(signal)
    freqs, power = welch(filtered, fs=3, nperseg=min(256, len(filtered)))

    valid = (freqs >= 0.6) & (freqs <= 1.4)
    peak_freq = freqs[valid][np.argmax(power[valid])] if valid.any() else 1.0
    hr = peak_freq * 60

    peaks, _ = find_peaks(filtered, distance=max(int(3 * 0.4), 1))
    if len(peaks) > 2:
        rr = np.diff(peaks) / 3
        rr_mean, sdnn = np.mean(rr), np.std(rr)
        rmssd = np.sqrt(np.mean(np.diff(rr) ** 2)) if len(rr) > 2 else 0
    else:
        rr_mean = sdnn = rmssd = 0

    lf_band = (freqs >= 0.04) & (freqs < 0.15)
    hf_band = (freqs >= 0.15) & (freqs <= 0.4)
    lf = trapezoid(power[lf_band], freqs[lf_band]) if lf_band.any() else 0
    hf = trapezoid(power[hf_band], freqs[hf_band]) if hf_band.any() else 0
    lf_hf = lf / (hf + 1e-8)

    return np.array([hr, rr_mean, sdnn, rmssd, lf, hf, lf_hf, np.std(filtered), 1.0], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Path to Hospital_Data root folder")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N videos (for testing)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"\nScanning {args.root} ...")
    records = scan_hospital_data(args.root)
    if args.limit:
        records = records[: args.limit]
    print(f"Processing {len(records)} video(s).")

    mobilenet = build_mobilenet_extractor().to(device)
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_model = clip_model.to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)

    stats = np.load(config.ALIGNED_DATA_DIR / "normalization_stats.npz")

    model = DepressionPredictionModelAttention().to(device)
    model.load_state_dict(torch.load(config.MOE_FUSION_BRANCH_DIR / "cv_safe_attention_final_fold1.pt", map_location=device))
    model.eval()

    tmp_root = Path(args.root) / "_tmp_frames"
    results = []

    for record in records:
        print(f"\n--- {record.video_id} ({record.split}) ---")

        tmp_dir = tmp_root / record.video_id
        frame_paths = extract_frames_from_video(record.path, tmp_dir)
        print(f"  {len(frame_paths)} frames extracted")

        visual = extract_visual(frame_paths, mobilenet, device)
        clip_feat = extract_clip(frame_paths, clip_model, clip_preprocess, device, config.MAX_FRAMES_FOR_CLIP)
        rppg_feat = extract_rppg(frame_paths)

        audio_path = tmp_dir / "audio.wav"
        clip_obj = VideoFileClip(str(record.path))
        if clip_obj.audio is not None:
            clip_obj.audio.write_audiofile(str(audio_path), fps=config.AUDIO_SAMPLE_RATE, nbytes=2, codec="pcm_s16le", logger=None)
            smile_feat = smile.process_file(str(audio_path)).iloc[0].to_numpy(dtype=np.float32)
        else:
            print("  WARNING: no audio track, skipping")
            clip_obj.close()
            continue
        clip_obj.close()

        # NOTE: stats["*_mean"] and stats["*_std"] are saved with shape
        # (1, D), so subtracting them from a (D,) array already produces a
        # (1, D) result via broadcasting. We must NOT unsqueeze on top of
        # that, or we end up with an incorrect (1, 1, D) shape that breaks
        # the attention layer downstream. reshape(1, -1) guarantees the
        # correct final shape regardless of what broadcasting produced.
        visual_norm = (visual - stats["visual_X_mean"]) / stats["visual_X_std"]
        rppg_norm = (rppg_feat - stats["rppg_X_mean"]) / stats["rppg_X_std"]
        clip_norm = (clip_feat - stats["clip_X_mean"]) / stats["clip_X_std"]
        smile_norm = (smile_feat - stats["smile_X_mean"]) / stats["smile_X_std"]

        batch = {
            "visual": torch.tensor(visual_norm, dtype=torch.float32).reshape(1, -1).to(device),
            "rppg": torch.tensor(rppg_norm, dtype=torch.float32).reshape(1, -1).to(device),
            "clip": torch.tensor(clip_norm, dtype=torch.float32).reshape(1, -1).to(device),
            "smile": torch.tensor(smile_norm, dtype=torch.float32).reshape(1, -1).to(device),
        }

        with torch.no_grad():
            pred, _ = model(batch)

        pred_score = pred.item()
        print(f"  Predicted (BDI-II scale): {pred_score:.2f}   |   True label (HAMD scale): {record.label}")
        results.append({"video_id": record.video_id, "split": record.split, "true_HAMD": record.label, "predicted_BDI_II_scale": pred_score})

    out_path = Path(args.root) / "predictions.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print("\nNOTE: predicted values are on the BDI-II scale (0-63); true_HAMD is on")
    print("a DIFFERENT scale. Do not compare these numbers directly without a")
    print("calibration step. Treat this as exploratory, not validated accuracy.")


if __name__ == "__main__":
    main()