# Depression Severity Estimation

Extends LightFusionNet — a lightweight facial-video + rPPG depression
severity estimator — with a frozen CLIP visual-semantic encoder and
openSMILE acoustic features, fused via three alternative mechanisms: a
modality-adaptive gate (MoE), a self-attention fusion mechanism, and an
evidence-based (uncertainty-weighted) fusion mechanism.

**Best result:** MAE 7.33 (self-attention fusion, 5-fold cross-validated
ensemble) vs. reproduced baseline MAE 8.10, on the AVEC2014 dataset
(BDI-II depression severity, 0-63 scale). A newer evidence-based fusion
model ties this result on MAE (7.29) with substantially more stable
per-fold performance, and is the only one of the three fusion mechanisms
that degrades gracefully when one or two modalities are missing at
inference time (see Section 3b).

This is a research prototype for estimating depression severity scores
from behavioural and physiological patterns. It is **not** a clinical
diagnostic tool and has not been validated for clinical use.

---

## Quick start (copy-paste, in order)
git clone https://github.com/umayrah-mm/Depression-Severity-Estimation.git

cd Depression-Severity-Estimation
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Then edit 2 lines in `config.py` (see Section 2 below) and run the
pipeline commands in Section 3.

---

## 1. Requirements

- Windows 10/11
- Python 3.10
- Git
- Several GB of free disk space for extracted features - does not
  include the raw video dataset itself

## 2. Point the project at your own dataset

Open `Facial_Expression_Depression_Recognition/multimodal_fusion/config.py`
and edit these 2 lines near the top:

```python
DATA_ROOT = Path(r"C:\path\to\your\processed_data_folder")
RAW_VIDEO_ROOT = Path(r"C:\path\to\your\raw_AVEC2014_videos")
```

`DATA_ROOT` is where labels, extracted frames, features, and all model
outputs get created. `RAW_VIDEO_ROOT` must contain:

RAW_VIDEO_ROOT/ 

├── train/ 

├── dev/ 

├── test/ 

└── label/ 

├── train_label.mat 

├── develop_label.mat 

└── test_label.mat

Nothing else in `config.py` needs editing - every other path is built
automatically from these two.

**Verify:**
cd Facial_Expression_Depression_Recognition\multimodal_fusion
python config.py

Should print `EXISTS` for `RAW_VIDEO_ROOT` and `LIGHTFUSIONNET_CODE_DIR`.

## 3. Run the pipeline
- Step A: labels + frames

python prepare_avec_labels.py
python prepare_avec_frames.py

- Step B: extract features (any order)

python extract_visual_embeddings.py
python extract_clip_features.py
python extract_audio.py
python extract_opensmile_features.py
python extract_green_rppg_features.py

- Step C: align + normalize

python dataset.py

- Step D: train (recommended - best result, MAE 7.33)

python train_cv_safe_attention.py

- Step D (alternative 1): simpler MoE gate architecture, MAE 7.73

python train_cv_safe.py

- Step D (alternative 2): evidence-based (uncertainty-weighted) fusion,
  MAE 7.29 with far more stable per-fold results and the ability to
  degrade gracefully if a modality is missing at inference time (see
  Section 3b) - trades a small amount of PCC/CCC for this

python train_cv_safe_evidence.py

**What Step D actually does:** trains 5 models using GroupKFold
cross-validation (grouped by subject ID, so no one person's data ever
appears in both a training and validation fold - this prevents data
leakage that inflates results). The Testing split is never touched
until final, one-time evaluation. Watch the printed per-fold
train/validation loss gap - a large, consistently growing gap across
folds signals overfitting.

**Optional diagnostics:**

python check_label_distribution.py 
# severity band balance + majority-class baseline
python evaluate.py 
# single-model evaluation + gate weights
python combine_ensembles.py 
# checks if combining both trained ensembles helps

---

## 3b. Missing-modality robustness (evidence fusion only)

Only the evidence-based fusion model (`models_evidence.py`) is designed
to handle a modality being unavailable at inference time - the MoE and
self-attention models assume all 4 modalities are always present and
have not been tested otherwise.

After training with `train_cv_safe_evidence.py`, run:

python evaluate_missing_modality_robustness.py

**What this does:** loads the 5 trained fold checkpoints (no retraining)
and re-evaluates the ensemble on the real, held-out Testing split under
all 11 combinations of 0, 1, or 2 modalities being missing, using
inverse-variance-weighted fusion to automatically down-weight whichever
modalities are unavailable. Results save to
`outputs/evidence_branch/results_missing_modality_robustness.txt`.

**Result summary** (5-fold ensemble, 100 Testing samples):

| Scenario | MAE |
|---|---|
| None missing (baseline) | 7.29 |
| Missing: rppg | 7.26 |
| Missing: visual | 7.35 |
| Missing: smile | 7.40 |
| Missing: clip | 8.47 |
| Missing: visual+rppg | 7.29 |
| Missing: rppg+smile | 7.45 |
| Missing: visual+smile | 7.58 |
| Missing: clip+rppg | 8.29 |
| Missing: clip+smile | 8.71 |
| Missing: visual+clip | 10.59 |

No scenario collapses catastrophically. The degradation pattern
independently reproduces the modality-importance ranking from the
ablation results below (CLIP matters most, rPPG matters least) via a
completely different method - missing-modality testing rather than
outright removal-and-retrain.

---

## 4. Running on a new dataset (e.g. a hospital/clinical dataset)

For datasets shaped like:

Hospital_Data/

├── train/ videos, e.g. 0001.MP4

├── dev/

├── test/

├── label.csv master labels (preferred)

├── train_label.csv optional, often identical to the others

├── dev_label.csv

└── test_label.csv


Run:

python predict_hospital_dataset.py --root "E:\path\to\Hospital_Data" --limit 2


**Always test with `--limit N` first** (processes only the first N
videos found) before running against a full dataset, to confirm the
loader and extraction pipeline work correctly on that specific data
before committing to a long full run. Drop `--limit` to process every
video found.

**What this does:**
- Scans `train/`, `dev/`, `test/` folders directly - handles any number
  of videos, not a fixed list
- Reads labels from the master `label.csv` (or falls back to the
  split-specific files), with video IDs always read as text and
  zero-padded consistently, to avoid a common mismatch where video
  files are named e.g. `0001.MP4` but the label CSV stores the same ID
  as `1`
- Extracts all 4 modalities (visual, CLIP, rPPG, audio/openSMILE) using
  the exact same functions used in the main AVEC2014 pipeline
- Normalizes using the ORIGINAL AVEC2014 training statistics (never
  recomputed from the new data)
- Saves predictions to `<root>\predictions.csv`
  (video_id, split, true_HAMD, predicted_BDI_II_scale)

**Status: this loader has not yet been run against real Hospital_Data
video files** (verified via code import checks and the same
extraction logic already proven on AVEC2014, but not an actual live
run). Run the `--limit 2` test first and report back any errors before
trusting a full run.

**Important - target scale mismatch:** this dataset's labels use HAMD,
a different depression rating scale than the BDI-II scale this model
was trained on. Predicted values are on the BDI-II scale (0-63) and are
**not** directly comparable to the HAMD ground truth without a separate
calibration step. Treat predictions on this dataset as exploratory, not
as validated accuracy.

---



## Ablation results summary

| Change tested | Result | Verdict |
|---|---|---|

| SVR correction layer after fusion | No improvement | Not adopted |

| Reduce visual features 2304 → 500 | No improvement | Not adopted |

| Remove CLIP modality | MAE worsened (8.15 → 9.02) | CLIP retained - confirmed useful |

| Remove rPPG modality | Statistical tie | rPPG retained - no proven benefit, no proven harm |

| Weighted loss for rare severity bands | Overfitting, worse MAE | Not adopted |

| 10-fold instead of 5-fold ensemble | Worse (smaller, noisier validation groups) | 5-fold retained |

| Multi-task learning (score + band) | Band accuracy good (0.50), MAE worsened | Not adopted |

| Attention layer feeding into MoE gate | Worse on every metric (7.33 → 8.03) - gate collapsed onto a single modality | Not adopted |

| 5-fold cross-validated ensembling | **Improved** (8.15 → 7.73 MoE / 7.33 attention) | **Adopted** |

| Evidence-based (uncertainty-weighted) fusion | Tied MAE (7.33 → 7.29), far more stable per-fold (std 0.52 → 0.12), but lower PCC/CCC; uniquely supports graceful missing-modality degradation | **Adopted as a separate model choice** - see Section 3b |

Full detail on each experiment is in `experiments_archive/`.

## Results

Evaluated on AVEC2014 (BDI-II depression severity, 0-63 scale), Testing
split, 5-fold cross-validated ensemble unless noted.

| Model | MAE ↓ | RMSE ↓ | PCC ↑ | CCC ↑ |
|---|---|---|---|---|
| Reproduced baseline (single split) | 8.15 | 9.90 | 0.54 | 0.49 |
| MoE gate, 5-fold ensemble | 7.73 | 10.00 | 0.51 | 0.45 |
| **Self-attention fusion, 5-fold ensemble (best on PCC/CCC)** | **7.33** | **9.58** | **0.58** | **0.55** |
| Evidence-based fusion, 5-fold ensemble (best on MAE/RMSE/stability) | 7.29 | 9.49 | 0.56 | 0.47 |

Lower is better for MAE/RMSE; higher is better for PCC/CCC. With
~297 aligned samples, treat MAE differences smaller than ~0.4 points
between variants as within normal cross-validation noise rather than
confirmed improvements (see Known Limitations) - the self-attention vs.
evidence-fusion MAE difference (7.33 vs 7.29) falls within this noise
floor and should be read as a tie on that metric specifically; the two
models are better distinguished by PCC/CCC (attention wins) and by
per-fold stability and missing-modality robustness (evidence fusion
wins - see Section 3b).

## Known limitations

- Trained and validated on AVEC2014 (German/English-speaking subjects).
  An exploratory test on a different-language dataset showed strong
  domain shift (input features fell far outside the training
  distribution) - see Section 4 above.
- The rPPG (pulse) modality contributes minimally to predictions -
  confirmed via ablation and PSD-based signal-quality analysis (not a
  data-quality artefact; the signal itself is weakly correlated with
  severity on this corpus), and independently confirmed again via the
  evidence-fusion missing-modality test (Section 3b), where removing
  rPPG barely changes MAE.
- Small dataset (~297 aligned samples): treat MAE differences smaller
  than ~0.4 points between model variants as within normal
  cross-validation noise, not confirmed improvements.
- The evidence-based fusion model has not yet had its per-sample
  confidence estimates checked for calibration (i.e. whether predicted
  uncertainty actually correlates with true error on a per-sample
  basis) - not yet investigated.
- The Hospital_Data loader (`predict_hospital_dataset.py`) has not yet
  been run against real video data - see Section 4 for status.
- This system estimates a severity score from learned statistical
  patterns. It is a research tool, not a diagnostic instrument.