"""
compute_full_pipeline_efficiency.py

Times the FULL pipeline end-to-end: raw frames -> MobileNet visual
extraction -> CLIP extraction -> rPPG extraction -> openSMILE audio
extraction -> fusion model prediction, using one real, already-
extracted AVEC2014 sample (frames + audio already on disk from earlier
today's work) - not synthetic data, so these numbers are trustworthy.

Reports each stage separately, since the frozen extractors (MobileNet,
CLIP, openSMILE) are a ONE-TIME per-video cost during preprocessing,
while the fusion model inference happens every time a prediction is
made - these are different costs and should be reported separately in
a paper, not combined into one misleading number.

Run:
    python compute_full_pipeline_efficiency.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import open_clip
import opensmile
import pandas as pd
from PIL import Image

import config

sys.path.insert(0, str(config.LIGHTFUSIONNET_CODE_DIR))
from dataset import IMAGE_TRANSFORM  # noqa: E402
from features import select_expressive_frames, extract_multi_region_features, enhanced_weighted_pooling  # noqa: E402

from models_attention import DepressionPredictionModelAttention, MODALITY_ORDER  # noqa: E402

MAX_EXPRESSIVE_FRAMES = 100


def find_sample_video():
    """Picks one real AVEC2014 video (frames + audio) already on disk from today's work."""
    labels_df = pd.read_csv(config.LABELS_PATH)
    for _, row in labels_df.iterrows():
        video_id = row["video"]
        split = row["split"]
        frame_dir = config.FRAMES_ROOT / split / video_id
        audio_path = config.AUDIO_DIR / f"{video_id}.wav"
        if frame_dir.exists() and audio_path.exists():
            frame_files = sorted(frame_dir.glob("*.jpg"))
            if len(frame_files) > 0:
                return video_id, frame_files, audio_path
    raise FileNotFoundError("Could not find any AVEC2014 sample with both frames and audio already extracted.")


def time_stage(name, fn):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    print(f"  {name:<30} {elapsed*1000:8.2f} ms")
    return result, elapsed


def main():
    device = "cpu"
    print(f"Device: {device} (CPU, for a fair deployability comparison)\n")

    print("Finding a real sample video (frames + audio already extracted today)...")
    video_id, frame_files, audio_path = find_sample_video()
    print(f"Using: {video_id}  ({len(frame_files)} frames)\n")

    print("Loading models (not timed - these load once, at startup, not per-video)...")
    mobilenet_full = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    mobilenet = nn.Sequential(mobilenet_full.features, mobilenet_full.avgpool)
    for p in mobilenet.parameters():
        p.requires_grad = False
    mobilenet.eval()

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)

    fusion_model = DepressionPredictionModelAttention().to(device)
    fusion_model.eval()

    print("Loading model done.\n")
    print("================ PER-STAGE TIMING (one video) ================")

    # ---- Stage 1: MobileNet visual extraction ----
    def visual_stage():
        frames = torch.stack([IMAGE_TRANSFORM(Image.open(f).convert("RGB")) for f in frame_files])
        if frames.shape[0] > MAX_EXPRESSIVE_FRAMES:
            frames, _ = select_expressive_frames(frames, mobilenet, device, MAX_EXPRESSIVE_FRAMES)
        region_feats = extract_multi_region_features(frames, mobilenet, device)
        return enhanced_weighted_pooling(region_feats).numpy()

    visual_feat, t_visual = time_stage("MobileNet visual extraction", visual_stage)

    # ---- Stage 2: CLIP extraction ----
    def clip_stage():
        n = len(frame_files)
        max_frames = config.MAX_FRAMES_FOR_CLIP
        selected = frame_files if n <= max_frames else [frame_files[i] for i in np.linspace(0, n - 1, max_frames).astype(int)]
        embeddings = []
        with torch.no_grad():
            for f in selected:
                img = clip_preprocess(Image.open(f).convert("RGB")).unsqueeze(0)
                embeddings.append(clip_model.encode_image(img).squeeze(0).numpy())
        return np.mean(np.stack(embeddings), axis=0).astype(np.float32)

    clip_feat, t_clip = time_stage("CLIP extraction", clip_stage)

    # ---- Stage 3: rPPG extraction ----
    def rppg_stage():
        import cv2
        from scipy.signal import butter, filtfilt, welch, find_peaks
        from scipy.integrate import trapezoid

        def bandpass(signal, fs=3, low=0.6, high=1.4, order=3):
            nyq = 0.5 * fs
            low, high = low / nyq, min(high / nyq, 0.99)
            b, a = butter(order, [low, high], btype="band")
            return filtfilt(b, a, signal)

        signal = []
        for f in frame_files:
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

    rppg_feat, t_rppg = time_stage("rPPG extraction", rppg_stage)

    # ---- Stage 4: openSMILE audio extraction ----
    def smile_stage():
        return smile.process_file(str(audio_path)).iloc[0].to_numpy(dtype=np.float32)

    smile_feat, t_smile = time_stage("openSMILE audio extraction", smile_stage)

    # ---- Stage 5: Fusion model inference ----
    fake_batch = {
        "visual": torch.tensor(visual_feat, dtype=torch.float32).reshape(1, -1),
        "rppg": torch.tensor(rppg_feat, dtype=torch.float32).reshape(1, -1),
        "clip": torch.tensor(clip_feat, dtype=torch.float32).reshape(1, -1),
        "smile": torch.tensor(smile_feat, dtype=torch.float32).reshape(1, -1),
    }

    def fusion_stage():
        with torch.no_grad():
            fusion_model(fake_batch)

    _, t_fusion = time_stage("Fusion model inference", fusion_stage)

    total = t_visual + t_clip + t_rppg + t_smile + t_fusion

    print(f"\n================ SUMMARY (one video, CPU, {len(frame_files)} frames) ================")
    print(f"Feature extraction (one-time per video, all 4 modalities): {(t_visual + t_clip + t_rppg + t_smile)*1000:.2f} ms")
    print(f"Fusion model inference (every prediction)                : {t_fusion*1000:.2f} ms")
    print(f"TOTAL end-to-end (raw video -> prediction)                : {total*1000:.2f} ms  (~{total:.2f} s)")

    print("\nBreakdown by stage:")
    for name, t in [("MobileNet visual", t_visual), ("CLIP", t_clip), ("rPPG", t_rppg),
                     ("openSMILE audio", t_smile), ("Fusion model", t_fusion)]:
        pct = 100 * t / total
        print(f"  {name:<20} {t*1000:8.2f} ms  ({pct:5.1f}% of total)")

    print("\nNote: feature extraction is a ONE-TIME cost per video (done once during")
    print("preprocessing/caching); fusion model inference (~1ms) is what runs every")
    print("time a prediction is made from already-extracted features.")


if __name__ == "__main__":
    main()