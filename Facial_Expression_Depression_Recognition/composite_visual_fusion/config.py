"""
config.py

Central configuration for the COMPOSITE VISUAL FUSION extension.

This is a self-contained experiment folder. It does NOT modify or import
anything from the sibling multimodal_fusion/ pipeline: that pipeline
(evidence fusion, MAE 7.29) stays untouched as the control baseline.

What is new here:
    The single time-collapsed visual "doctor" of the evidence-fusion model
    is replaced by a COMPOSITE visual doctor built from two members whose
    errors were proven complementary in the WIN/ATLAS project:
        - static member : whole-face MobileNetV3 mean-pooled embedding
        - temporal member: eyes/mouth/cheek region sequence + motion tokens
                           through a small Transformer (WIN regional recipe)
    The two members are combined with NESTED inverse-variance weighting
    (each member has its own score + log-variance head), and the resulting
    composite doctor takes the visual slot in the outer 4-doctor evidence
    panel (composite-visual, CLIP, rPPG, openSMILE).

Paths: reuses the raw AVEC2014 videos and the already-extracted baseline
features (labels, CLIP, openSMILE, rPPG) READ-ONLY, and writes all of its
own artifacts under output/composite_visual_branch/.
"""

from pathlib import Path

# ============================================================
# EDIT THESE LINES FOR YOUR OWN SETUP
# ============================================================

# Raw AVEC2014 tree (train/, dev/, test/, label/). Same root the
# multimodal_fusion pipeline uses.
RAW_VIDEO_ROOT = Path(r"E:\AVEC2014")

# The output folder that the EXISTING multimodal_fusion pipeline created.
# We only READ from it (labels, CLIP features, openSMILE features, rPPG csv).
BASELINE_DATA_ROOT = Path(__file__).resolve().parent.parent / "output"

# ============================================================
# Everything below is built automatically. No need to edit.
# ============================================================

PACKAGE_DIR = Path(__file__).resolve().parent

# --- Read-only inputs from the existing pipeline ----------------------
LABELS_PATH = BASELINE_DATA_ROOT / "labels.csv"
CLIP_BRANCH_DIR = BASELINE_DATA_ROOT / "clip_branch"
OPENSMILE_BRANCH_DIR = BASELINE_DATA_ROOT / "opensmile_branch"
RPPG_FEATURES_CSV = BASELINE_DATA_ROOT / "rppg_features.csv"

# --- Our own outputs ---------------------------------------------------
OUTPUTS_ROOT = BASELINE_DATA_ROOT / "composite_visual_branch"

STATIC_VISUAL_DIR = OUTPUTS_ROOT / "static_visual"          # <video_id>.npz
TEMPORAL_VISUAL_DIR = OUTPUTS_ROOT / "temporal_visual"      # <video_id>.npz
ALIGNED_DATA_DIR = OUTPUTS_ROOT / "aligned_dataset"
COMPOSITE_BRANCH_DIR = OUTPUTS_ROOT / "composite_models"    # fold checkpoints

# --- Splits (must match labels.csv "split" column) ---------------------
SPLIT_NAMES = ["Training", "Development", "Testing"]

CSV_SPLIT_TO_RAW_FOLDER = {
    "Training": "train",
    "Development": "dev",
    "Testing": "test",
}

# video_id in labels.csv looks like "203_1_Freeform_video".
# The raw mp4 file name is "<video_id>.mp4".


def raw_video_path(video_id: str) -> Path:
    """Map an aligned video_id to its raw mp4 path."""
    # subject_session_Taskname_video -> split comes from labels.csv,
    # but the raw folder must be derived from the subject's partition.
    raise NotImplementedError("use video_paths_index() instead")


# --- Model / training hyperparameters ----------------------------------
BATCH_SIZE = 16
EPOCHS = 120
LEARNING_RATE = 1e-3          # outer doctors + static member (evidence recipe)
WEIGHT_DECAY = 1e-2           # outer doctors + static member
TEMPORAL_LEARNING_RATE = 3e-4  # temporal member (WIN regional recipe)
TEMPORAL_WEIGHT_DECAY = 0.05   # temporal member
EMBED_DIM = 128               # shared embedding space of the evidence panel
RANDOM_SEED = 42

# Modality dropout probabilities (two-level; see composite_training_utils)
SUBBRANCH_DROPOUT_PROB = 0.15  # drop static-only or temporal-only
DOCTOR_DROPOUT_PROB = 0.15     # drop the whole composite visual doctor
OUTER_DROPOUT_PROB = 0.15      # drop clip / rppg / smile doctors

AUX_LOSS_WEIGHT = 0.3          # heteroscedastic auxiliary loss weight

# --- WIN regional extraction constants (ported, video-level) -----------
N_SEGMENTS = 32
FRAMES_PER_SEGMENT = 2
T_FRAMES = N_SEGMENTS * FRAMES_PER_SEGMENT   # 64
PAIR_GAP_S = 0.1
CROP_SIZE = 224
EYE_LEFT_TARGET = (72.0, 85.0)
EYE_RIGHT_TARGET = (152.0, 85.0)
MOBILENET_EMBED_DIM = 576
REGION_NAMES = ("full", "eyes", "mouth", "cheek_left", "cheek_right")
REGION_BOXES = {
    "full": (0, CROP_SIZE, 0, CROP_SIZE),
    "eyes": (0, CROP_SIZE // 2, 0, CROP_SIZE),
    "mouth": (2 * CROP_SIZE // 3, CROP_SIZE, 0, CROP_SIZE),
    "cheek_left": (CROP_SIZE // 3, 2 * CROP_SIZE // 3, 0, CROP_SIZE // 2),
    "cheek_right": (CROP_SIZE // 3, 2 * CROP_SIZE // 3, CROP_SIZE // 2, CROP_SIZE),
}
A05_REGIONS = ("eyes", "mouth", "cheek_left", "cheek_right")
PCA_DIM = 48

# Static visual extraction
MAX_SAMPLE_ERROR_S = (1 / 30) + 1e-6
FACE_TEMPLATE_224 = [
    [76.5892, 103.3926],
    [147.0636, 103.0034],
    [112.0504, 143.4732],
    [83.0986, 184.7310],
    [141.4598, 184.4082],
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# rPPG feature columns (identical to multimodal_fusion/dataset.py)
RPPG_FEATURE_COLUMNS = [
    "HR", "RR", "SDNN", "RMSSD", "LF", "HF", "LF_HF", "sample_entropy", "dfa_alpha"
]


def ensure_output_dirs():
    """Create every output directory if it doesn't already exist."""
    for d in [STATIC_VISUAL_DIR, TEMPORAL_VISUAL_DIR, ALIGNED_DATA_DIR, COMPOSITE_BRANCH_DIR]:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("Checking paths for composite_visual_fusion:")
    for name, path in [
        ("RAW_VIDEO_ROOT", RAW_VIDEO_ROOT),
        ("LABELS_PATH", LABELS_PATH),
        ("CLIP_BRANCH_DIR", CLIP_BRANCH_DIR),
        ("OPENSMILE_BRANCH_DIR", OPENSMILE_BRANCH_DIR),
        ("RPPG_FEATURES_CSV", RPPG_FEATURES_CSV),
    ]:
        status = "EXISTS" if path.exists() else "MISSING <-- fix config"
        print(f"  {name}: {status}  ({path})")

    print("\nCreating output directories (if missing)...")
    ensure_output_dirs()
    print(f"  Output root: {OUTPUTS_ROOT}")
    print("\nIf all inputs say EXISTS, config.py is working correctly.")
