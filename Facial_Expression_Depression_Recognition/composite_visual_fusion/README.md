# Composite Visual Fusion (WIN static + temporal as one visual doctor)

This folder extends the evidence-fusion architecture WITHOUT touching the
existing `multimodal_fusion/` pipeline, which stays intact as the control
baseline (Testing MAE 7.29). Everything written here lives under
`Facial_Expression_Depression_Recognition/output/composite_visual_branch/`.

## Idea

The evidence panel has 4 doctors: visual, clip, rppg, smile. The baseline's
visual doctor was a single projector over one mean-pooled 2304-D embedding.
The WIN project (E:/Depression) proved that a **static whole-face member** and
a **temporal region-sequence member** make _complementary errors_ - their blend
beat either member alone (Test MAE 8.31 / 7.48 / 7.11).

This experiment replaces the visual doctor with those two members:

- **static member**: whole-face MobileNetV3-Small, 2 fps, 5-point aligned,
  mean-pooled (576-D)
- **temporal member**: 64-frame eyes/mouth/cheek region sequence through a
  small pre-norm Transformer (WIN regional recipe, audio stripped)

Each member has its own `(score, log-variance)` head; the doctor blends them
by **nested inverse-variance weighting**:

```
p_m    = exp(-s_m)                     m in {static, temporal}
w_m    = p_m / (p_static + p_temporal)
y_vis  = w_s * y_s + w_t * y_t
var_vis = 1 / (p_s + p_t)  <=  min(var_s, var_t)
```

so the doctor is _sharper than either member alone_ when both are confident,
and automatically defers when one fails or is missing (hard floor log_var=8).
The doctor's `(embedding, score, log-variance)` then occupies the visual slot
of the outer 4-doctor evidence panel, whose inverse-variance fusion and
missing-modality handling are unchanged. Member disagreement is paid for in
variance: it raises doctor uncertainty, which lowers its outer weight.

## Pipeline (run in this order)

All commands from this folder. `py` below is the project venv python
(`e:\TUKL\Depression-Severity-Estimation\.venv\Scripts\python.exe`).

```powershell
cd e:\TUKL\Depression-Severity-Estimation\Facial_Expression_Depression_Recognition\composite_visual_fusion
$py = "e:\TUKL\Depression-Severity-Estimation\.venv\Scripts\python.exe"

# 0. path check (fast)
& $py config.py

# 1. extraction smoke test on 2 videos (~1-3 min) - MTCNN + backbone wiring
& $py win_extraction.py

# 2. static member features, 297 videos, resume-safe (long; a few hours)
& $py extract_static_visual.py

# 3. temporal member features, 297 videos, resume-safe (long; faster than 2)
& $py extract_temporal_visual.py

# 4. align the 6 sources into split NPZs (fast)
& $py dataset_composite.py

# 5. subject-grouped 5-fold CV training (GPU, ~15-40 min)
& $py train_cv_safe_composite.py

# 6. ONE-TIME Testing evaluation of the 5-fold ensemble
& $py test_composite_model.py

# 7. missing-modality stress test incl. sub-branch scenarios
& $py evaluate_missing_modality_composite.py
```

Quick wiring check before committing to the long runs: add `--limit 3` to
steps 2 and 3. The produced files are valid outputs for those videos, so you
can simply continue the full run afterwards (resume-safe).

## Outputs

```
output/composite_visual_branch/
    static_visual/<video_id>.npz        (576,) features + valid flag
    temporal_visual/<video_id>.npz      (64, 5, 576) + frame_valid
    aligned_dataset/{training,development,testing}.npz
    composite_models/composite_fold_{1..5}.pt
    composite_models/composite_fold_{1..5}_preprocessor.npz
    composite_models/cv_results.txt         (per-fold + OOF summary)
    composite_models/oof_predictions.csv
    composite_models/test_predictions.csv
    composite_models/test_results.txt
    composite_models/missing_modality_results.txt
```

## Protocol / discipline (inherited from the baselines)

- video-level samples (297), identical alignment philosophy to
  `multimodal_fusion/dataset.py` (intersection of all sources; a video whose
  visual extraction fully failed is KEPT with visual mask 0)
- GroupKFold(5) over Training+Development grouped by subject, explicit
  leakage check; Testing touched exactly once (step 6)
- scalar modalities (static / clip / rppg / smile) z-scored with
  Training-split stats, like the baseline
- temporal features are preprocessed FOLD-LOCALLY (per-region missing-aware
  z-score + deterministic PCA-48 + segment motion, WIN recipe), fit on each
  fold's train rows only; the same fitted preprocessor is used for that
  fold's val rows and Testing rows
- optimizer: AdamW with two param groups - temporal member + head at WIN
  recipe (lr 3e-4, wd 0.05), everything else at evidence recipe (lr 1e-3,
  wd 1e-2); grad-norm clip 1.0; SmoothL1 main + 0.3 \* heteroscedastic aux
  over the 5 member heads; patience 15
- two-level modality dropout: sub-branch (0.15), whole visual doctor (0.15),
  outer doctors (0.15), always honoring true availability

## Success criteria

- Testing MAE <= 7.29 (evidence control)
- prediction std clearly above WIN's 8.2 collapse value; severe-band MAE
  below ~15 (range-squash audit, see WIN FINDINGS.md)
- graceful degradation: losing ONE visual member should cost little (that is
  the whole point of the nested design); losing the whole visual doctor
  should degrade no worse than the baseline did

## Environment

Runs in the project venv (torch 2.6.0+cu124 - identical to the WIN/ATLAS
lock). One extra package, installed with `--no-deps` so it cannot touch
torch:

```
pip install facenet-pytorch==2.6.0 --no-deps
```

See `requirements-composite.txt`.
