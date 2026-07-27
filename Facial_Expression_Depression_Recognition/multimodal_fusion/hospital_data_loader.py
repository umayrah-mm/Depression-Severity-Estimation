"""
hospital_data_loader.py

Generalized loader for Hospital_Data-shaped datasets (train/dev/test
folders of videos + a label CSV), instead of a hardcoded video list.

Handles two real, confirmed issues with this kind of dataset:
  1. Duplicate/identical split-specific label CSVs (train_label.csv,
     dev_label.csv, test_label.csv can be accidental copies of each
     other) - we always prefer the master label.csv if present.
  2. Video ID zero-padding mismatch: video files are often named
     "0001.MP4" but label CSVs frequently store the ID as "1" (pandas
     auto-converts numeric-looking strings) - normalize_video_id()
     fixes this by always zero-padding to a fixed width.

Split membership is determined by which folder (train/dev/test) a
video physically sits in - NOT by which label CSV it's listed in,
since those are unreliable per the issue above.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
_ID_RE = re.compile(r"^(\d+)$")


def normalize_video_id(raw, width: int = 4) -> str:
    """Maps 1 / 0001 / 0001.MP4 -> '0001' (consistent zero-padded string)."""
    s = str(raw).strip()
    if "." in s:
        s = Path(s).stem
    s = s.strip()
    if not _ID_RE.match(s):
        raise ValueError(f"Expected a numeric video id, got: {raw!r}")
    return s.zfill(width)


def _pick_column(fieldnames, candidates):
    if not fieldnames:
        raise ValueError("CSV has no header row")
    lower = {f.lower(): f for f in fieldnames}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    raise KeyError(f"None of {candidates} found in columns: {fieldnames}")


def resolve_label_csv(root: Path) -> Path:
    """Prefers the master label.csv; falls back to any *_label.csv."""
    root = Path(root)
    for name in ("label.csv", "train_label.csv", "dev_label.csv", "test_label.csv"):
        p = root / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"No label CSV found under {root}")


def load_label_table(label_csv: Path, score_column_candidates=("HAMD", "hamd", "label", "score", "BDI-II")):
    """
    Loads video_id -> score from a label CSV. Reads the ID column as
    TEXT explicitly (not letting it be silently converted to an int)
    to avoid the zero-padding bug.
    """
    path = Path(label_csv)
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        id_key = _pick_column(fields, ("video_id", "filename", "id", "video", "序号"))
        score_key = _pick_column(fields, score_column_candidates)

        out = {}
        for row in reader:
            vid = normalize_video_id(row[id_key])
            out[vid] = float(row[score_key])
    return out


def list_split_videos(split_dir: Path):
    """Returns unique video file paths in a split folder."""
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        return []
    seen = {}
    for p in sorted(split_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            seen.setdefault(p.name.lower(), p)
    return list(seen.values())


@dataclass(frozen=True)
class HospitalVideoRecord:
    video_id: str
    split: str
    path: Path
    label: float


def scan_hospital_data(root, id_width: int = 4):
    """
    Scans root/train, root/dev, root/test for video files, matches each
    to its label via the master label CSV, and returns a list of
    HospitalVideoRecord - one per video, however many there are.
    """
    root = Path(root)
    label_path = resolve_label_csv(root)
    labels = load_label_table(label_path)

    records = []
    missing_label = []

    for split in ("train", "dev", "test"):
        for video_path in list_split_videos(root / split):
            vid = normalize_video_id(video_path.stem, width=id_width)
            if vid not in labels:
                missing_label.append(f"{split}/{video_path.name}")
                continue
            records.append(HospitalVideoRecord(
                video_id=vid, split=split, path=video_path.resolve(), label=labels[vid]
            ))

    if missing_label:
        preview = ", ".join(missing_label[:8])
        print(f"WARNING: {len(missing_label)} video(s) have no matching label "
              f"in {label_path.name}: {preview}")

    return records


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python hospital_data_loader.py <path_to_hospital_data_root>")
        sys.exit(1)

    records = scan_hospital_data(sys.argv[1])
    print(f"\nFound {len(records)} videos with valid labels:")
    for split in ("train", "dev", "test"):
        n = sum(1 for r in records if r.split == split)
        print(f"  {split}: {n}")