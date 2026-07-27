"""
dataset.py

The most safety-critical module in this project: aligns four independently
extracted feature branches (visual, rPPG, CLIP, openSMILE) by video_id,
verifies label consistency, splits into train/dev/test, and normalizes
using train-only statistics.

NEVER assumes row order matches across files. Every value is looked up
explicitly by video_id string.

Run directly to perform alignment and save the result:
    python dataset.py
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config


RPPG_FEATURE_COLUMNS = [
    "HR", "RR", "SDNN", "RMSSD", "LF", "HF", "LF_HF", "sample_entropy", "dfa_alpha"
]


def load_all_sources():
    """
    Loads every raw source file. Returns a dict of dataframes/arrays,
    completely unaligned still - alignment happens in align_all().
    """
    labels_df = pd.read_csv(config.LABELS_PATH)
    labels_df = labels_df.rename(columns={"video": "video_id", "BDI-II": "bdi_labels"})

    rppg_df = pd.read_csv(config.RPPG_FEATURES_CSV)
    rppg_df = rppg_df.rename(columns={"BDI_II": "bdi_rppg"})

    visual_X = np.load(config.VISUAL_RAW_BRANCH_DIR / "visual_embeddings.npy")
    visual_ids = np.load(config.VISUAL_RAW_BRANCH_DIR / "visual_sample_ids.npy", allow_pickle=True)

    clip_X = np.load(config.CLIP_BRANCH_DIR / "clip_features.npy")
    clip_ids = np.load(config.CLIP_BRANCH_DIR / "clip_sample_ids.npy", allow_pickle=True)

    smile_X = np.load(config.OPENSMILE_BRANCH_DIR / "opensmile_features.npy")
    smile_ids = np.load(config.OPENSMILE_BRANCH_DIR / "opensmile_sample_ids.npy", allow_pickle=True)

    return {
        "labels_df": labels_df,
        "rppg_df": rppg_df,
        "visual_X": visual_X, "visual_ids": visual_ids,
        "clip_X": clip_X, "clip_ids": clip_ids,
        "smile_X": smile_X, "smile_ids": smile_ids,
    }


def build_id_to_row_map(sample_ids):
    """
    Given an array of video_id strings (possibly in any order), returns a
    dict mapping video_id -> row index in that array. Used to reorder any
    array into a common, agreed-upon sample order.
    """
    id_to_row = {}
    for i, vid in enumerate(sample_ids):
        if vid in id_to_row:
            raise ValueError(f"Duplicate video_id found: {vid}. This should never happen.")
        id_to_row[vid] = i
    return id_to_row


def align_all():
    """
    Core alignment logic. Returns a dict with everything reordered onto a
    single, common, sorted list of video_ids that exist in ALL sources.
    """
    src = load_all_sources()

    labels_df = src["labels_df"]
    rppg_df = src["rppg_df"]

    # ---- Build id -> row maps for the three .npy-based sources ----
    visual_map = build_id_to_row_map(src["visual_ids"])
    clip_map = build_id_to_row_map(src["clip_ids"])
    smile_map = build_id_to_row_map(src["smile_ids"])

    # rPPG and labels are DataFrames; index by video_id directly.
    rppg_df = rppg_df.set_index("video_id", drop=False)
    labels_df = labels_df.set_index("video_id", drop=False)

    if rppg_df.index.duplicated().any() or labels_df.index.duplicated().any():
        raise ValueError("Duplicate video_id found in labels.csv or rppg_features.csv.")

    # ---- Find the common set of video_ids present in ALL five sources ----
    id_sets = [
        set(labels_df.index),
        set(rppg_df.index),
        set(visual_map.keys()),
        set(clip_map.keys()),
        set(smile_map.keys()),
    ]
    common_ids = sorted(set.intersection(*id_sets))

    print(f"Sample counts per source:")
    print(f"  labels.csv        : {len(labels_df)}")
    print(f"  rppg_features.csv : {len(rppg_df)}")
    print(f"  visual embeddings : {len(visual_map)}")
    print(f"  clip features     : {len(clip_map)}")
    print(f"  opensmile features: {len(smile_map)}")
    print(f"  COMMON to all five: {len(common_ids)}")

    dropped = set.union(*id_sets) - set(common_ids)
    if dropped:
        print(f"\n{len(dropped)} sample(s) dropped due to missing data in at least one source:")
        for vid in sorted(dropped):
            print(f"  {vid}")

    # ---- Reorder every source into the SAME common_ids order ----
    visual_X = src["visual_X"][[visual_map[v] for v in common_ids]]
    clip_X = src["clip_X"][[clip_map[v] for v in common_ids]]
    smile_X = src["smile_X"][[smile_map[v] for v in common_ids]]
    rppg_X = rppg_df.loc[common_ids, RPPG_FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    y = labels_df.loc[common_ids, "bdi_labels"].to_numpy(dtype=np.float32)
    split = labels_df.loc[common_ids, "split"].to_numpy()
    video_ids = np.array(common_ids)

    # ---- Safety check: do the two independently-stored labels agree? ----
    y_rppg_check = rppg_df.loc[common_ids, "bdi_rppg"].to_numpy(dtype=np.float32)
    label_mismatch = np.abs(y - y_rppg_check) > 1e-6
    if label_mismatch.any():
        print(f"\n*** WARNING: {label_mismatch.sum()} samples have MISMATCHED labels")
        print("between labels.csv and rppg_features.csv! Investigate before proceeding: ***")
        for vid in video_ids[label_mismatch][:10]:
            print(f"  {vid}")
    else:
        print("\nLabel consistency check PASSED: labels.csv and rppg_features.csv agree on every sample.")

    return {
        "visual_X": visual_X, "rppg_X": rppg_X, "clip_X": clip_X, "smile_X": smile_X,
        "y": y, "split": split, "video_ids": video_ids,
    }


def split_by_name(aligned, split_name):
    """Returns a boolean mask selecting rows belonging to one split."""
    return aligned["split"] == split_name


def normalize_with_train_stats(X_train, X_dev, X_test):
    """
    Z-score normalization: (x - mean) / std, using statistics computed
    ONLY from X_train, then applied unchanged to dev/test. This avoids
    leaking dev/test distribution information into preprocessing.
    """
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1e-8  # avoid divide-by-zero for constant features

    X_train_norm = (X_train - mean) / std
    X_dev_norm = (X_dev - mean) / std
    X_test_norm = (X_test - mean) / std
    return X_train_norm, X_dev_norm, X_test_norm, mean, std


class MultimodalDataset(Dataset):
    """
    A PyTorch Dataset holding already-aligned, already-normalized features
    for ONE split (Training, Development, or Testing).

    Shapes per sample:
        visual : (2304,)
        rppg   : (9,)
        clip   : (512,)
        smile  : (88,)
        y      : scalar
    """

    def __init__(self, visual_X, rppg_X, clip_X, smile_X, y, video_ids):
        self.visual_X = torch.tensor(visual_X, dtype=torch.float32)
        self.rppg_X = torch.tensor(rppg_X, dtype=torch.float32)
        self.clip_X = torch.tensor(clip_X, dtype=torch.float32)
        self.smile_X = torch.tensor(smile_X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.video_ids = video_ids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "visual": self.visual_X[idx],
            "rppg": self.rppg_X[idx],
            "clip": self.clip_X[idx],
            "smile": self.smile_X[idx],
            "y": self.y[idx],
            "video_id": self.video_ids[idx],
        }


def build_and_save_datasets():
    """
    Full pipeline: align -> split -> normalize -> save to disk as .npz
    files (one per split), so train_moe.py can load instantly without
    redoing alignment every time.
    """
    config.ensure_output_dirs()
    aligned = align_all()

    masks = {name: split_by_name(aligned, name) for name in config.SPLIT_NAMES}

    for name in config.SPLIT_NAMES:
        n = masks[name].sum()
        print(f"\n{name}: {n} samples")
        if n == 0:
            raise ValueError(f"Split '{name}' has 0 samples after alignment. Check split names.")

    train_mask, dev_mask, test_mask = masks["Training"], masks["Development"], masks["Testing"]

    # Normalize each modality independently, using train-only statistics.
    saved_stats = {}
    normalized = {}
    for modality in ["visual_X", "rppg_X", "clip_X", "smile_X"]:
        X_train, X_dev, X_test, mean, std = normalize_with_train_stats(
            aligned[modality][train_mask],
            aligned[modality][dev_mask],
            aligned[modality][test_mask],
        )
        normalized[modality] = {"Training": X_train, "Development": X_dev, "Testing": X_test}
        saved_stats[modality] = {"mean": mean, "std": std}

    for name, mask in masks.items():
        out_path = config.ALIGNED_DATA_DIR / f"{name.lower()}.npz"
        np.savez(
            out_path,
            visual_X=normalized["visual_X"][name],
            rppg_X=normalized["rppg_X"][name],
            clip_X=normalized["clip_X"][name],
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
    print(f"Saved normalization stats to: {config.ALIGNED_DATA_DIR / 'normalization_stats.npz'}")


def load_split_dataset(split_name):
    """
    Convenience loader for train_moe.py / evaluate.py: loads a pre-saved,
    already-aligned, already-normalized split and wraps it as a
    MultimodalDataset.
    """
    path = config.ALIGNED_DATA_DIR / f"{split_name.lower()}.npz"
    data = np.load(path, allow_pickle=True)
    return MultimodalDataset(
        data["visual_X"], data["rppg_X"], data["clip_X"], data["smile_X"],
        data["y"], data["video_ids"],
    )


if __name__ == "__main__":
    print("================ Building aligned multimodal dataset ================\n")
    build_and_save_datasets()

    print("\n================ Final verification ================")
    for name in config.SPLIT_NAMES:
        ds = load_split_dataset(name)
        sample = ds[0]
        print(f"\n{name}: {len(ds)} samples")
        print(f"  visual shape: {tuple(sample['visual'].shape)}  (expect (2304,))")
        print(f"  rppg   shape: {tuple(sample['rppg'].shape)}    (expect (9,))")
        print(f"  clip   shape: {tuple(sample['clip'].shape)}    (expect (512,))")
        print(f"  smile  shape: {tuple(sample['smile'].shape)}   (expect (88,))")
        print(f"  y: {sample['y'].item():.2f}   video_id: {sample['video_id']}")

    print("\nIf all shapes match and there are no WARNING lines above, alignment succeeded.")