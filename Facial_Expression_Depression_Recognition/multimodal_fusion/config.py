"""
Central configuration for the multimodal fusion extension of LightFusionNet.

Every other script in multimodal_fusion/ imports paths and constants from
this file instead of hardcoding them.

============================================================
TO USE THIS PROJECT ON YOUR OWN COMPUTER / DATASET:
Edit ONLY the settings inside the block below. Nothing else in this
file needs to change.
============================================================
"""

from pathlib import Path

# ============================================================
# EDIT THESE 3 LINES FOR YOUR OWN SETUP
# ============================================================

# Where your project's data lives (raw videos, extracted frames, features,
# and all outputs will be created here). This is the ONE folder you need
# to point at your own dataset location.
DATA_ROOT = Path(r"C:\Users\HP\Desktop\AVEC2014_processed")

# Where your raw, original videos are stored (as downloaded/unzipped).
RAW_VIDEO_ROOT = Path(r"C:\Users\HP\Desktop\AVEC-2014_Dataset_Folder\AVEC2014")

# Where this repository's LightFusionNet subfolder lives (usually you
# don't need to change this if you keep the repo's folder structure intact -
# it's built automatically below from the repo's own location).
LIGHTFUSIONNET_CODE_DIR = Path(__file__).resolve().parent.parent / "LightFusionNet"

# ============================================================
# Everything below this line is built automatically. No need to edit.
# ============================================================

# Labels CSV produced by prepare_avec_labels.py
LABELS_PATH = DATA_ROOT / "labels.csv"

# Root of already-extracted face frames, produced by prepare_avec_frames.py
FRAMES_ROOT = DATA_ROOT / "processed_frames"

# The three split names EXACTLY as they appear in labels.csv "split" column
# and as the subfolder names under FRAMES_ROOT / RAW_VIDEO_ROOT.
SPLIT_NAMES = ["Training", "Development", "Testing"]

# labels.csv uses "Training"/"Development"/"Testing" in its `split` column,
# and FRAMES_ROOT subfolders use those same capitalized names too.
# BUT the RAW video folders use lowercase short names instead. This maps
# CSV split -> raw folder name.
CSV_SPLIT_TO_RAW_FOLDER = {
    "Training": "train",
    "Development": "dev",
    "Testing": "test",
}

# ---------------------------------------------------------------------------
# Outputs this project creates
# ---------------------------------------------------------------------------

OUTPUTS_ROOT = DATA_ROOT / "outputs"

VISUAL_BRANCH_DIR = OUTPUTS_ROOT / "visual_branch"
RPPG_BRANCH_DIR = OUTPUTS_ROOT / "rppg_branch"
FUSION_BRANCH_DIR = OUTPUTS_ROOT / "fusion_branch"

AUDIO_DIR = DATA_ROOT / "audio"
CLIP_BRANCH_DIR = OUTPUTS_ROOT / "clip_branch"
OPENSMILE_BRANCH_DIR = OUTPUTS_ROOT / "opensmile_branch"
RPPG_FEATURES_CSV = DATA_ROOT / "rppg_features.csv"
ALIGNED_DATA_DIR = OUTPUTS_ROOT / "aligned_dataset"
MOE_FUSION_BRANCH_DIR = OUTPUTS_ROOT / "moe_fusion_branch"
VISUAL_RAW_BRANCH_DIR = OUTPUTS_ROOT / "visual_branch_raw"
BASELINE_CV_DIR = OUTPUTS_ROOT / "baseline_cv_branch"
# ---------------------------------------------------------------------------
# Model / training hyperparameters
# ---------------------------------------------------------------------------

BATCH_SIZE = 16
EPOCHS = 120
LEARNING_RATE = 1e-3
EMBED_DIM = 128
LOAD_BALANCE_WEIGHT = 0.1
RANDOM_SEED = 42
TASK_TYPE = "regression"

# ---------------------------------------------------------------------------
# Frame / audio extraction settings
# ---------------------------------------------------------------------------

MAX_FRAMES_FOR_CLIP = 16
AUDIO_SAMPLE_RATE = 16000


def ensure_output_dirs():
    """
    Create every output directory if it doesn't already exist.
    Safe to call every time a script starts.
    """
    for d in [AUDIO_DIR, CLIP_BRANCH_DIR, OPENSMILE_BRANCH_DIR, MOE_FUSION_BRANCH_DIR,
              VISUAL_RAW_BRANCH_DIR, ALIGNED_DATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("Checking existing paths (should all say EXISTS):")
    for name, path in [
        ("RAW_VIDEO_ROOT", RAW_VIDEO_ROOT),
        ("LABELS_PATH", LABELS_PATH),
        ("FRAMES_ROOT", FRAMES_ROOT),
        ("LIGHTFUSIONNET_CODE_DIR", LIGHTFUSIONNET_CODE_DIR),
    ]:
        status = "EXISTS" if path.exists() else "MISSING <-- fix DATA_ROOT / RAW_VIDEO_ROOT above"
        print(f"  {name}: {status}  ({path})")

    print("\nCreating output directories (if missing)...")
    ensure_output_dirs()
    for name, path in [
        ("AUDIO_DIR", AUDIO_DIR),
        ("CLIP_BRANCH_DIR", CLIP_BRANCH_DIR),
        ("OPENSMILE_BRANCH_DIR", OPENSMILE_BRANCH_DIR),
        ("MOE_FUSION_BRANCH_DIR", MOE_FUSION_BRANCH_DIR),
    ]:
        status = "EXISTS" if path.exists() else "STILL MISSING <-- BUG"
        print(f"  {name}: {status}  ({path})")

    print("\nIf all existing paths say EXISTS and all new dirs say EXISTS, config.py is working correctly.")