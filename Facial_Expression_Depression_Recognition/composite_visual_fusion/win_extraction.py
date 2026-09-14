"""
win_extraction.py

Ported extraction primitives from the WIN/ATLAS project (E:/Depression/win/lib.py),
adapted from session-level (Freeform + Northwind) to single-video level.

The two feature streams this module produces:

1. STATIC visual (WIN A01 static path, audio stripped):
     decode at ~2 fps -> MTCNN 5-point align -> frozen MobileNetV3-Small
     (children()[:-1] + weights.transforms()) -> mean-pool over valid
     frames -> (576,) embedding per video.

2. TEMPORAL visual (WIN A05 regional path, audio stripped):
     64-frame plan (32 segments x 2 frames, 100 ms gap) -> batched MTCNN ->
     eyes-only similarity align -> 5 region crops -> frozen MobileNetV3-Small
     (features + avgpool) -> (64, 5, 576) features + frame_valid mask +
     segment index per video.

Everything here is deterministic given the video file and the frozen
backbone weights. No labels are read.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import config

# ---------------------------------------------------------------------------
# Video id -> raw mp4 path resolution
# ---------------------------------------------------------------------------

_VIDEO_RE_PARTS = ("Freeform", "Northwind")


def video_paths_index() -> dict[str, Path]:
    """
    Scan the raw AVEC2014 tree once and map every video_id found in
    labels.csv (e.g. "203_1_Freeform_video") to its mp4 path.

    Returns {video_id: Path}.
    """
    index: dict[str, Path] = {}
    for split_name, folder in config.CSV_SPLIT_TO_RAW_FOLDER.items():
        video_dir = config.RAW_VIDEO_ROOT / folder
        if not video_dir.exists():
            continue
        for path in sorted(video_dir.glob("*.mp4")):
            index[path.stem] = path
    return index


# ---------------------------------------------------------------------------
# Face picking / backbones (ported verbatim from win/lib.py)
# ---------------------------------------------------------------------------


def pick_face(boxes, probabilities) -> int | None:
    candidates = []
    for index, (box, probability) in enumerate(zip(boxes, probabilities)):
        if box is None or probability is None or not np.isfinite(float(probability)):
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        candidates.append(
            (-float(probability), -max(0.0, x2 - x1) * max(0.0, y2 - y1), x1, y1, index))
    if not candidates:
        return None
    return sorted(candidates)[0][-1]


def load_static_backbone(device: torch.device):
    """Static path backbone: children()[:-1] + weights.transforms()."""
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.DEFAULT
    visual = mobilenet_v3_small(weights=weights).to(device).eval()
    backbone = nn.Sequential(*list(visual.children())[:-1]).to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    return backbone, weights


def load_region_backbone(device: torch.device):
    """Regional path backbone: features + avgpool with manual ImageNet normalize."""
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    backbone = nn.Sequential(model.features, model.avgpool).to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    return backbone


def build_detector(device: torch.device):
    """MTCNN face detector exactly as WIN constructs it."""
    from facenet_pytorch import MTCNN

    return MTCNN(keep_all=True, device=device)


_IMAGENET_MEAN = np.asarray(config.IMAGENET_MEAN, dtype=np.float32)
_IMAGENET_STD = np.asarray(config.IMAGENET_STD, dtype=np.float32)


@torch.inference_mode()
def embed_uint8(backbone, batch_uint8: np.ndarray, device: torch.device, chunk: int = 160) -> np.ndarray:
    """Region-path embedding: manual ImageNet normalize, frozen forward."""
    outputs = []
    for start in range(0, len(batch_uint8), chunk):
        block = batch_uint8[start: start + chunk].astype(np.float32) / 255.0
        block = (block - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(np.ascontiguousarray(
            block.transpose(0, 3, 1, 2))).to(device)
        outputs.append(backbone(tensor).flatten(1).float().cpu().numpy())
    return (
        np.concatenate(outputs, axis=0).astype(np.float32)
        if outputs
        else np.zeros((0, config.MOBILENET_EMBED_DIM), np.float32)
    )


# ---------------------------------------------------------------------------
# Static visual (A01-style, video level)
# ---------------------------------------------------------------------------


def sample_frames_2fps(video_path: Path) -> list[np.ndarray]:
    """Deterministic 2 fps frame sampling (ported from win/lib.py)."""
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"cannot decode {video_path}")
    next_target, frames, previous, used_indices, decode_index = 0.0, [], None, set(), 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            current = (decode_index, timestamp, cv2.cvtColor(
                frame_bgr, cv2.COLOR_BGR2RGB))
            while timestamp + 1e-9 >= next_target:
                candidates = [current] if previous is None else [
                    previous, current]
                candidate = min(candidates, key=lambda item: (
                    abs(item[1] - next_target), item[1], item[0]))
                if candidate[0] not in used_indices and abs(candidate[1] - next_target) <= config.MAX_SAMPLE_ERROR_S:
                    frames.append(candidate[2])
                    used_indices.add(candidate[0])
                next_target += 0.5
            previous = current
            decode_index += 1
    finally:
        cap.release()
    return frames


_FACE_TEMPLATE = np.asarray(config.FACE_TEMPLATE_224, dtype=np.float32)


def align_face_5point(frame_rgb: np.ndarray, detector) -> np.ndarray | None:
    """MTCNN 5-point similarity align to 224x224 (static path)."""
    boxes, probabilities, landmarks = detector.detect(
        frame_rgb, landmarks=True)
    if boxes is None or probabilities is None or landmarks is None:
        return None
    index = pick_face(boxes, probabilities)
    if index is None:
        return None
    matrix, _ = cv2.estimateAffinePartial2D(
        np.asarray(landmarks[index], dtype=np.float32), _FACE_TEMPLATE, method=cv2.LMEDS
    )
    if matrix is None:
        return None
    return cv2.warpAffine(
        frame_rgb, matrix, (224, 224), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


@torch.inference_mode()
def extract_static_task(video_path: Path, detector, backbone, weights, device) -> tuple[np.ndarray, bool]:
    """Mean-pooled whole-face embedding over 2 fps frames. (576,), valid."""
    transform = weights.transforms()
    embeddings = []
    for frame in sample_frames_2fps(video_path):
        face = align_face_5point(frame, detector)
        if face is None:
            continue
        tensor = transform(Image.fromarray(face)).unsqueeze(0).to(device)
        emb = backbone(tensor).flatten(1).float().cpu().numpy()[0]
        embeddings.append(emb)
    if not embeddings:
        return np.zeros(config.MOBILENET_EMBED_DIM, np.float32), False
    return np.mean(np.stack(embeddings), axis=0).astype(np.float32), True


# ---------------------------------------------------------------------------
# Temporal visual (A05-style, video level)
# ---------------------------------------------------------------------------


def frame_plan(n_frames: int, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """32 segments x 2 frames with PAIR_GAP_S spacing (ported)."""
    gap = max(1, int(round(config.PAIR_GAP_S * fps)))
    indices, segments = [], []
    for segment in range(config.N_SEGMENTS):
        start = segment * n_frames / config.N_SEGMENTS
        stop = (segment + 1) * n_frames / config.N_SEGMENTS
        anchor = int(np.floor((start + stop) / 2.0))
        for offset in range(config.FRAMES_PER_SEGMENT):
            index = min(max(anchor + offset * gap, 0), n_frames - 1)
            indices.append(index)
            segments.append(segment)
    return np.asarray(indices, np.int64), np.asarray(segments, np.int16)


def decode_planned_frames(video_path: Path):
    """Decode only the 64 planned frames (ported)."""
    capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode {video_path}")
    try:
        n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        indices, segments = frame_plan(n_frames, fps)
        wanted: dict[int, list[int]] = {}
        for slot, index in enumerate(indices):
            wanted.setdefault(int(index), []).append(slot)
        frames: list[np.ndarray | None] = [None] * len(indices)
        timestamps = np.full(len(indices), np.nan, np.float32)
        decoded = 0
        highest = max(wanted)
        while decoded <= highest:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if decoded in wanted:
                timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                for slot in wanted[decoded]:
                    frames[slot] = rgb
                    timestamps[slot] = timestamp
            decoded += 1
    finally:
        capture.release()
    return frames, timestamps, segments


def eye_similarity_matrix(left_eye, right_eye) -> np.ndarray | None:
    """Eyes-only similarity transform (temporal path alignment)."""
    source = np.asarray([left_eye, right_eye], dtype=np.float64)
    target = np.asarray(
        [config.EYE_LEFT_TARGET, config.EYE_RIGHT_TARGET], dtype=np.float64)
    source_delta = source[1] - source[0]
    target_delta = target[1] - target[0]
    source_norm = float(np.hypot(*source_delta))
    if not np.isfinite(source_norm) or source_norm < 1e-3:
        return None
    scale = float(np.hypot(*target_delta)) / source_norm
    angle = float(
        np.arctan2(target_delta[1], target_delta[0]) -
        np.arctan2(source_delta[1], source_delta[0])
    )
    cos, sin = np.cos(angle) * scale, np.sin(angle) * scale
    rotation = np.asarray([[cos, -sin], [sin, cos]], dtype=np.float64)
    translation = target[0] - rotation @ source[0]
    return np.hstack([rotation, translation.reshape(2, 1)]).astype(np.float64)


def align_crop_eyes(frame_rgb: np.ndarray, landmarks: np.ndarray) -> np.ndarray | None:
    matrix = eye_similarity_matrix(landmarks[0], landmarks[1])
    if matrix is None:
        return None
    return cv2.warpAffine(
        frame_rgb, matrix, (config.CROP_SIZE,
                            config.CROP_SIZE), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def region_stack(crop: np.ndarray) -> np.ndarray:
    """Cut one aligned crop into the 5 region patches, resized to 224x224."""
    regions = np.empty(
        (len(config.REGION_NAMES), config.CROP_SIZE, config.CROP_SIZE, 3), np.uint8)
    for i, name in enumerate(config.REGION_NAMES):
        y0, y1, x0, x1 = config.REGION_BOXES[name]
        patch = crop[y0:y1, x0:x1]
        if patch.shape[0] != config.CROP_SIZE or patch.shape[1] != config.CROP_SIZE:
            patch = cv2.resize(
                patch, (config.CROP_SIZE, config.CROP_SIZE), interpolation=cv2.INTER_LINEAR)
        regions[i] = patch
    return regions


def extract_temporal_task(video_path: Path, detector, backbone, device, detect_batch: int = 16):
    """
    (T_FRAMES, n_regions, 576) features + (T_FRAMES,) valid mask +
    (T_FRAMES,) segment index for one video.
    """
    frames, timestamps, segments = decode_planned_frames(video_path)
    features = np.zeros(
        (config.T_FRAMES, len(config.REGION_NAMES),
         config.MOBILENET_EMBED_DIM), np.float32
    )
    valid = np.zeros(config.T_FRAMES, bool)
    timestamps = np.nan_to_num(timestamps, nan=0.0).astype(np.float32)
    present = [slot for slot, frame in enumerate(frames) if frame is not None]
    crops: dict[int, np.ndarray] = {}
    for start in range(0, len(present), detect_batch):
        slots = present[start: start + detect_batch]
        batch = np.stack([frames[slot] for slot in slots])
        boxes, probabilities, landmarks = detector.detect(
            batch, landmarks=True)
        for slot, box, probability, landmark in zip(slots, boxes, probabilities, landmarks):
            if box is None or landmark is None:
                continue
            index = pick_face(box, probability)
            if index is None:
                continue
            crop = align_crop_eyes(frames[slot], np.asarray(
                landmark[index], dtype=np.float64))
            if crop is None:
                continue
            crops[slot] = crop
            valid[slot] = True
    if crops:
        slots = sorted(crops)
        stacked = np.concatenate([region_stack(crops[slot])
                                 for slot in slots], axis=0)
        embeddings = embed_uint8(backbone, stacked, device)
        embeddings = embeddings.reshape(len(slots), len(
            config.REGION_NAMES), config.MOBILENET_EMBED_DIM)
        for position, slot in enumerate(slots):
            features[slot] = embeddings[position]
    return features, valid, timestamps, segments


# ---------------------------------------------------------------------------
# Atomic NPZ writer (ported)
# ---------------------------------------------------------------------------


def atomic_npz(path: Path, arrays: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        tmp = Path(handle.name)
    try:
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Smoke test on 2 videos (fast; run directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("win_extraction.py smoke test: 2 videos, no label access...\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    index = video_paths_index()
    sample_ids = sorted(index)[:2]
    print(f"Raw tree contains {len(index)} videos; testing: {sample_ids}\n")

    detector = build_detector(device)
    static_backbone, static_weights = load_static_backbone(device)
    region_backbone = load_region_backbone(device)

    for video_id in sample_ids:
        path = index[video_id]
        print(f"--- {video_id} ---")
        emb, ok = extract_static_task(
            path, detector, static_backbone, static_weights, device)
        print(
            f"  static : shape={emb.shape} valid={ok} finite={np.isfinite(emb).all()}")
        feats, valid, times, segs = extract_temporal_task(
            path, detector, region_backbone, device)
        print(
            f"  temporal: shape={feats.shape} valid_frames={int(valid.sum())}/"
            f"{config.T_FRAMES} finite={np.isfinite(feats).all()}"
        )
    print("\nSmoke test finished. Extraction primitives are working.")
    sys.exit(0)
