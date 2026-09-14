"""
dataset_composite.py

Aligns SIX feature sources by video_id and saves per-split NPZ files:

    1. labels.csv                     (read-only, existing pipeline)
    2. static_visual/<id>.npz         (576,)   + valid          [new]
    3. temporal_visual/<id>.npz       (64,5,576) + frame_valid   [new]
    4. clip_branch/*.npy              (512,)                    (read-only)
    5. opensmile_branch/*.npy         (88,)                     (read-only)
    6. rppg_features.csv              (9,)                      (read-only)

Normalization conventions (identical philosophy to multimodal_fusion/dataset.py):
    - static / clip / rppg / smile : z-scored with TRAINING-split stats,
      saved normalized. Same convention as the baseline evidence pipeline,
      so the outer doctors see comparable inputs.
    - temporal : stored RAW in the split NPZs. Fold-local preprocessing
      (per-region z-score + PCA-48 + motion tokens, WIN recipe) happens
      inside the trainer and is fit on train-fold rows only.

Visual availability:
    - static_valid  / temporal_valid come straight from extraction NPZs.
    - A video with BOTH visual sub-branches failed is KEPT with
      visual doctor mask = 0 (the evidence model handles missing doctors
      natively). Only the common intersection with labels/clip/smile/rppg
      decides membership, matching the baseline's 297-sample set as
      closely as possible.

Run:
    python dataset_composite.py
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_labels() -> pd.DataFrame:
    labels_df = pd.read_csv(config.LABELS_PATH)
    labels_df = labels_df.rename(
        columns={"video": "video_id", "BDI-II": "bdi_labels"})
    return labels_df.set_index("video_id", drop=False)


def load_scalar_sources() -> dict:
    """labels + clip + smile + rppg, exactly like the baseline loader."""
    labels_df = load_labels()

    clip_X = np.load(config.CLIP_BRANCH_DIR / "clip_features.npy")
    clip_ids = np.load(config.CLIP_BRANCH_DIR /
                       "clip_sample_ids.npy", allow_pickle=True)

    smile_X = np.load(config.OPENSMILE_BRANCH_DIR / "opensmile_features.npy")
    smile_ids = np.load(config.OPENSMILE_BRANCH_DIR /
                        "opensmile_sample_ids.npy", allow_pickle=True)

    rppg_df = pd.read_csv(config.RPPG_FEATURES_CSV)
    rppg_df = rppg_df.rename(columns={"BDI_II": "bdi_rppg"})
    rppg_df = rppg_df.set_index("video_id", drop=False)

    return {
        "labels_df": labels_df,
        "clip_X": clip_X, "clip_ids": clip_ids,
        "smile_X": smile_X, "smile_ids": smile_ids,
        "rppg_df": rppg_df,
    }


def build_id_to_row_map(sample_ids):
    id_to_row = {}
    for i, vid in enumerate(sample_ids):
        if vid in id_to_row:
            raise ValueError(
                f"Duplicate video_id found: {vid}. This should never happen.")
        id_to_row[vid] = i
    return id_to_row


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def align_all(verbose=True) -> dict:
    """
    Returns everything reordered onto one sorted common video_id list that
    exists in labels + clip + smile + rppg AND has both visual NPZ files
    on disk (regardless of their valid flags - a fully-failed visual video
    is kept with visual mask 0).
    """
    src = load_scalar_sources()
    labels_df = src["labels_df"]

    clip_map = build_id_to_row_map(src["clip_ids"])
    smile_map = build_id_to_row_map(src["smile_ids"])
    rppg_df = src["rppg_df"]

    labels_df = labels_df.set_index("video_id", drop=False)
    if rppg_df.index.duplicated().any() or labels_df.index.duplicated().any():
        raise ValueError(
            "Duplicate video_id in labels.csv or rppg_features.csv.")

    # Visual NPZ availability (file presence, not validity).
    static_ids = {p.stem for p in config.STATIC_VISUAL_DIR.glob("*.npz")}
    temporal_ids = {p.stem for p in config.TEMPORAL_VISUAL_DIR.glob("*.npz")}

    id_sets = [
        set(labels_df.index),
        set(rppg_df.index),
        set(clip_map.keys()),
        set(smile_map.keys()),
        static_ids,
        temporal_ids,
    ]
    common_ids = sorted(set.intersection(*id_sets))

    if verbose:
        print("Sample counts per source:")
        print(f"  labels.csv            : {len(labels_df)}")
        print(f"  rppg_features.csv     : {len(rppg_df)}")
        print(f"  clip features         : {len(clip_map)}")
        print(f"  opensmile features    : {len(smile_map)}")
        print(f"  static visual NPZs    : {len(static_ids)}")
        print(f"  temporal visual NPZs  : {len(temporal_ids)}")
        print(f"  COMMON to all six     : {len(common_ids)}")

    dropped = set.union(*id_sets) - set(common_ids)
    if dropped and verbose:
        print(
            f"\n{len(dropped)} sample(s) dropped (missing in at least one source):")
        for vid in sorted(dropped):
            print(f"  {vid}")

    # ---- Reorder scalar sources onto common_ids ----
    clip_X = src["clip_X"][[clip_map[v] for v in common_ids]]
    smile_X = src["smile_X"][[smile_map[v] for v in common_ids]]
    rppg_X = rppg_df.loc[common_ids, config.RPPG_FEATURE_COLUMNS].to_numpy(
        dtype=np.float32)

    y = labels_df.loc[common_ids, "bdi_labels"].to_numpy(dtype=np.float32)
    split = labels_df.loc[common_ids, "split"].to_numpy()

    # ---- Label cross-check against rppg csv (baseline discipline) ----
    y_rppg_check = rppg_df.loc[common_ids,
                               "bdi_rppg"].to_numpy(dtype=np.float32)
    mismatch = np.abs(y - y_rppg_check) > 1e-6
    if mismatch.any() and verbose:
        print(f"\n*** WARNING: {mismatch.sum()} samples have MISMATCHED labels "
              f"between labels.csv and rppg_features.csv! ***")
    elif verbose:
        print("\nLabel consistency check PASSED (labels.csv vs rppg_features.csv).")

    # ---- Load per-video visual NPZs in common_ids order ----
    n = len(common_ids)
    static_X = np.zeros((n, 576), np.float32)
    static_valid = np.zeros(n, bool)
    temporal_X = np.zeros((n, config.T_FRAMES, len(
        config.REGION_NAMES), config.MOBILENET_EMBED_DIM), np.float32)
    temporal_valid = np.zeros(n, bool)          # any valid frame
    temporal_frame_valid = np.zeros((n, config.T_FRAMES), bool)

    for i, vid in enumerate(common_ids):
        with np.load(config.STATIC_VISUAL_DIR / f"{vid}.npz", allow_pickle=False) as data:
            static_X[i] = np.asarray(data["features"], np.float32)
            static_valid[i] = bool(np.asarray(data["valid"], bool))

    for i, vid in enumerate(common_ids):
        with np.load(config.TEMPORAL_VISUAL_DIR / f"{vid}.npz", allow_pickle=False) as data:
            block = np.asarray(data["region_features"], np.float32)
            fvalid = np.asarray(data["frame_valid"], bool)
            if block.shape[0] != config.T_FRAMES:
                raise ValueError(f"{vid}: temporal block has {block.shape[0]} frames, "
                                 f"expected {config.T_FRAMES}")
        temporal_X[i] = block
        temporal_frame_valid[i] = fvalid
        temporal_valid[i] = bool(fvalid.any())

    if verbose:
        print(f"\nVisual validity: static {int(static_valid.sum())}/{n}, "
              f"temporal {int(temporal_valid.sum())}/{n}, "
              f"both-failed {int((~static_valid & ~temporal_valid).sum())}/{n} (kept, mask=0)")

    return {
        "static_X": static_X,
        "static_valid": static_valid,
        "temporal_X": temporal_X,                # raw
        "temporal_frame_valid": temporal_frame_valid,
        "temporal_valid": temporal_valid,
        "clip_X": clip_X,
        "smile_X": smile_X,
        "rppg_X": rppg_X,
        "y": y,
        "split": split,
        "video_ids": np.array(common_ids),
    }


# ---------------------------------------------------------------------------
# Normalization (train-only stats) for the scalar modalities
# ---------------------------------------------------------------------------


def normalize_with_train_stats(X_train, X_dev, X_test):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1e-8
    return (
        (X_train - mean) / std,
        (X_dev - mean) / std,
        (X_test - mean) / std,
        mean,
        std,
    )


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------


class CompositeDataset(Dataset):
    """
    One split of the aligned composite dataset.

    Per sample:
        static      : (576,)     z-scored (train-split stats), zeroed if invalid
        static_mask : float      1 = valid
        clip        : (512,)     z-scored
        rppg        : (9,)       z-scored
        smile       : (88,)      z-scored
        temporal    : (64, 5, 576) RAW (fold-local preprocessing in trainer)
        frame_valid : (64,) bool
        y           : scalar
    """

    def __init__(self, static_X, static_valid, temporal_X, temporal_frame_valid,
                 clip_X, rppg_X, smile_X, y, video_ids):
        self.static_X = torch.tensor(static_X, dtype=torch.float32)
        self.static_valid = torch.tensor(static_valid, dtype=torch.float32)
        # Zero out invalid static vectors (they were saved as zeros already,
        # but be explicit for safety).
        self.static_X = self.static_X * self.static_valid.unsqueeze(-1)
        self.temporal_X = torch.tensor(np.asarray(temporal_X, np.float32))
        self.frame_valid = torch.tensor(np.asarray(temporal_frame_valid, bool))
        self.clip_X = torch.tensor(clip_X, dtype=torch.float32)
        self.rppg_X = torch.tensor(rppg_X, dtype=torch.float32)
        self.smile_X = torch.tensor(smile_X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.video_ids = video_ids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "static": self.static_X[idx],
            "static_mask": self.static_valid[idx],
            "temporal": self.temporal_X[idx],
            "frame_valid": self.frame_valid[idx],
            "clip": self.clip_X[idx],
            "rppg": self.rppg_X[idx],
            "smile": self.smile_X[idx],
            "y": self.y[idx],
            "video_id": self.video_ids[idx],
        }


# ---------------------------------------------------------------------------
# Build & save
# ---------------------------------------------------------------------------


def build_and_save_datasets():
    config.ensure_output_dirs()
    aligned = align_all()

    masks = {name: aligned["split"] == name for name in config.SPLIT_NAMES}
    for name in config.SPLIT_NAMES:
        n = masks[name].sum()
        print(f"\n{name}: {n} samples")
        if n == 0:
            raise ValueError(f"Split '{name}' has 0 samples after alignment.")

    train_mask, dev_mask, test_mask = masks["Training"], masks["Development"], masks["Testing"]

    saved_stats = {}
    normalized = {}
    for modality in ["static_X", "clip_X", "rppg_X", "smile_X"]:
        X_train, X_dev, X_test, mean, std = normalize_with_train_stats(
            aligned[modality][train_mask],
            aligned[modality][dev_mask],
            aligned[modality][test_mask],
        )
        normalized[modality] = {"Training": X_train,
                                "Development": X_dev, "Testing": X_test}
        saved_stats[modality] = {"mean": mean, "std": std}

    for name, mask in masks.items():
        out_path = config.ALIGNED_DATA_DIR / f"{name.lower()}.npz"
        np.savez_compressed(
            out_path,
            static_X=normalized["static_X"][name],
            static_valid=aligned["static_valid"][mask],
            temporal_X=aligned["temporal_X"][mask],               # raw
            temporal_frame_valid=aligned["temporal_frame_valid"][mask],
            clip_X=normalized["clip_X"][name],
            rppg_X=normalized["rppg_X"][name],
            smile_X=normalized["smile_X"][name],
            y=aligned["y"][mask],
            video_ids=aligned["video_ids"][mask],
        )
        print(f"Saved: {out_path}")

    np.savez(
        config.ALIGNED_DATA_DIR / "normalization_stats.npz",
        **{f"{mod}_mean": saved_stats[mod]["mean"] for mod in saved_stats},
        **{f"{mod}_std": saved_stats[mod]["std"] for mod in saved_stats},
    )
    print(
        f"Saved normalization stats to: {config.ALIGNED_DATA_DIR / 'normalization_stats.npz'}")


def load_split_dataset(split_name: str) -> CompositeDataset:
    """Loads a pre-saved split NPZ and wraps it as a CompositeDataset."""
    path = config.ALIGNED_DATA_DIR / f"{split_name.lower()}.npz"
    # All arrays are plain numeric / unicode dtypes saved by our own scripts,
    # so no pickle deserialization is needed or allowed here.
    data = np.load(path, allow_pickle=False)
    return CompositeDataset(
        data["static_X"], data["static_valid"],
        data["temporal_X"], data["temporal_frame_valid"],
        data["clip_X"], data["rppg_X"], data["smile_X"],
        data["y"], data["video_ids"],
    )


def dataset_to_arrays(ds: CompositeDataset) -> dict:
    """Flat numpy view of a loaded split (used by the trainer and eval scripts)."""
    return {
        "static_X": ds.static_X.numpy(),
        "static_valid": ds.static_valid.numpy(),
        "temporal_X": ds.temporal_X.numpy(),
        "frame_valid": ds.frame_valid.numpy(),
        "clip_X": ds.clip_X.numpy(),
        "rppg_X": ds.rppg_X.numpy(),
        "smile_X": ds.smile_X.numpy(),
        "y": ds.y.numpy(),
        "video_ids": np.asarray(ds.video_ids),
    }


if __name__ == "__main__":
    print("================ Building composite aligned dataset ================\n")
    build_and_save_datasets()

    print("\n================ Final verification ================")
    for name in config.SPLIT_NAMES:
        ds = load_split_dataset(name)
        sample = ds[0]
        print(f"\n{name}: {len(ds)} samples")
        print(
            f"  static   shape: {tuple(sample['static'].shape)}  mask={sample['static_mask'].item():.0f}")
        print(
            f"  temporal shape: {tuple(sample['temporal'].shape)}  valid frames={int(sample['frame_valid'].sum())}")
        print(f"  clip     shape: {tuple(sample['clip'].shape)}")
        print(f"  rppg     shape: {tuple(sample['rppg'].shape)}")
        print(f"  smile    shape: {tuple(sample['smile'].shape)}")
        print(
            f"  y: {sample['y'].item():.2f}   video_id: {sample['video_id']}")

    print("\nIf all shapes match and there are no WARNING lines above, alignment succeeded.")
