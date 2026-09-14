# Composite Visual Fusion: WIN Static + Temporal Members as One Visual Evidence Doctor — Complete Architecture Guide

> **Audience**: Undergraduate students, machine learning researchers, and developers who want a clear, mathematically rigorous understanding of how this system estimates depression severity (BDI-II) from video, audio, and physiological signals — and specifically how the single visual branch of the evidence-fusion model was replaced by a **two-member sub-committee** whose errors were proven complementary in the WIN/ATLAS project (E:/Depression).

> **Companion documents**: [ARCHITECTURE_EXPLAINED.md](ARCHITECTURE_EXPLAINED.md) (self-attention fusion), [EVIDENCE_FUSION_ARCHITECTURE_EXPLAINED.md](EVIDENCE_FUSION_ARCHITECTURE_EXPLAINED.md) (the 4-doctor evidence panel this system extends). This document assumes familiarity with the evidence-fusion concepts (dual-output heads, log-variance, inverse-variance weighting, hard-floor suppressor) and focuses on what is **new**.

---

## Table of Contents

1. [The Big Picture: From One Dr. Visual to a Two-Member Sub-Committee](#1-the-big-picture-from-one-dr-visual-to-a-two-member-sub-committee)
2. [What Changed in the Repository](#2-what-changed-in-the-repository)
3. [The Two Visual Members & Their Feature Extraction Pipelines](#3-the-two-visual-members--their-feature-extraction-pipelines)
4. [Architecture Deep-Dive: How Data Flows](#4-architecture-deep-dive-how-data-flows)
5. [The Mathematics of Nested Evidence Fusion](#5-the-mathematics-of-nested-evidence-fusion)
6. [Training Protocol: Fold-Local Preprocessing, Two-Level Dropout, Dual Optimizer Recipes](#6-training-protocol-fold-local-preprocessing-two-level-dropout-dual-optimizer-recipes)
7. [Results: New Best MAE With Two Levels of Graceful Degradation](#7-results-new-best-mae-with-two-levels-of-graceful-degradation)
8. [Caveats & Honest Assessment](#8-caveats--honest-assessment)
9. [The Severe-Band Problem: Diagnosis and the Road to Sub-6.5 MAE](#9-the-severe-band-problem-diagnosis-and-the-road-to-sub-65-mae)
10. [Step-by-Step Reproduction & Execution Guide](#10-step-by-step-reproduction--execution-guide)

---

## 1. The Big Picture: From One Dr. Visual to a Two-Member Sub-Committee

The evidence-fusion model consults four "doctors" — Visual, CLIP, rPPG, openSMILE — each giving a score and an uncertainty, combined by inverse-variance weighting. Its weakness: **Dr. Visual was a single generalist**, one projector over one mean-pooled 2304-D facial embedding. One representation, one set of blind spots.

The WIN/ATLAS project (E:/Depression) established a central empirical finding on the same AVEC2014 corpus: a **static whole-face model** and a **temporal region-sequence model** make **complementary errors**. The temporal member (eyes/mouth/cheek micro-dynamics through a small Transformer, Test MAE 7.48) beat the static member (mean-pooled appearance, Test MAE 8.31), but their late blend beat both (7.11) — because they fail on _different_ videos.

This architecture makes that complementarity **structural**. Dr. Visual becomes a sub-committee of two specialists, each with their own score _and_ their own uncertainty, blended by the same inverse-variance mathematics used by the outer panel:

```text
              ┌──────────────────────────────────────────────────────────────────┐
              │                        PATIENT VIDEO                            │
              └───────┬───────────────────────────────┬──────────────────────────┘
                      │                               │
                      ▼                               ▼
        ┌───────────────────────────┐     ┌───────────────────────────────────┐
        │  STATIC SPECIALIST        │     │  TEMPORAL SPECIALIST              │
        │  whole-face appearance,   │     │  eyes/mouth/cheek micro-dynamics, │
        │  2 fps, mean-pooled       │     │  64 frames, 4 regions, motion     │
        │  MobileNetV3 (576-D)      │     │  tokens, 1-layer Transformer      │
        │                           │     │                                   │
        │  "Score 22 (±5.1)"        │     │  "Score 31 (±6.8)"                │
        └─────────────┬─────────────┘     └───────────────┬───────────────────┘
                      │                                   │
                      └───────────────┬───────────────────┘
                                      ▼
              ┌──────────────────────────────────────────────────────────────────┐
              │        DR. VISUAL — nested inverse-variance sub-committee       │
              │  "Score 26.6 (±3.9)" — sharper than either member alone         │
              └───────────────────────────────┬──────────────────────────────────┘
                                              │
             ┌────────────────┬────────────────┼────────────────┐
             ▼                ▼                ▼                ▼
        [Dr. Visual]      [Dr. CLIP]      [Dr. rPPG]      [Dr. Smile]
             └────────────────┴────────┬───────┴────────────────┘
                                       ▼
                    OUTER INVERSE-VARIANCE PANEL (unchanged recipe)
                                       ▼
                         PREDICTED BDI-II SEVERITY SCORE
```

### Why not just concatenate the two members into one branch?

Concatenation forces a single projector to digest two very different feature geometries (a 576-D appearance vector vs. a 64×4×48 spatio-temporal tensor), and when one member's input fails, the zeros poison the shared representation. The nested-evidence design instead keeps each member a **standalone diagnostician with its own uncertainty**, so:

- If the temporal member is missing or unsure, the doctor's weight flows to the static member — _and_ the doctor's own uncertainty reflects the loss of evidence.
- If both members are confident and agree, the doctor is **mathematically sharper than either member alone** (Section 5.3).
- The doctor's uncertainty feeds the outer panel, so the _panel_ automatically discounts the whole visual channel when its internal evidence is thin — two independent levels of graceful degradation.

---

## 2. What Changed in the Repository

The experiment lives in a **new, fully self-contained folder**. The existing `multimodal_fusion/` pipeline — the working 7.29 MAE system — was **not modified by a single line** and remains the control baseline. All new artifacts are written under `output/composite_visual_branch/`.

```text
Depression-Severity-Estimation/
├── Facial_Expression_Depression_Recognition/
│   ├── multimodal_fusion/                  # [UNTOUCHED] control baseline (MAE 7.29)
│   ├── composite_visual_fusion/            # [NEW] this experiment
│   │   ├── config.py                       # paths + hyperparameters (dual optimizer recipes)
│   │   ├── win_extraction.py               # WIN extraction primitives, ported to video-level
│   │   ├── extract_static_visual.py        # static member features -> <video_id>.npz (resume-safe)
│   │   ├── extract_temporal_visual.py      # temporal member features -> <video_id>.npz (resume-safe)
│   │   ├── dataset_composite.py            # 6-source alignment -> 297 samples, 3 split NPZs
│   │   ├── models_composite.py             # TemporalMember, CompositeVisualDoctor, outer panel
│   │   ├── composite_training_utils.py     # fold-local preprocessor, 2-level dropout, 5-head loss
│   │   ├── train_cv_safe_composite.py      # leak-free GroupKFold(5) trainer
│   │   ├── test_composite_model.py         # ONE-TIME Testing evaluation + spread audit
│   │   ├── evaluate_missing_modality_composite.py   # 13-scenario stress test
│   │   ├── README.md                       # pipeline quickstart
│   │   └── requirements-composite.txt      # documents the facenet-pytorch delta
│   └── output/
│       └── composite_visual_branch/        # [NEW] all artifacts (features, models, results)
└── docs/
    └── COMPOSITE_VISUAL_FUSION_ARCHITECTURE_EXPLAINED.md   # [THIS FILE]
```

**Read-only dependencies on the old pipeline**: `labels.csv`, `clip_branch/`, `opensmile_branch/`, `rppg_features.csv` are consumed as-is, guaranteeing an apples-to-apples sample set with the baseline (same 297 videos — the 3 videos missing rPPG features are dropped in both systems).

**One new environment package**: `facenet-pytorch==2.6.0` installed with `--no-deps` (ATLAS decision D026), providing the MTCNN face detector. The venv otherwise already matches the WIN/ATLAS environment lock (torch 2.6.0+cu124).

---

## 3. The Two Visual Members & Their Feature Extraction Pipelines

Both members consume **frozen, ImageNet-pretrained MobileNetV3-Small** backbones — no visual encoder is fine-tuned; all learning happens in the small heads above the features. Both members are **purely visual**: the eGeMAPS audio that WIN's members also consumed is stripped, because the outer panel's Dr. Smile already owns acoustics (avoids double-counting the same evidence).

### 3.1 Static Member (WIN A01 recipe, video-level)

| Stage          | Detail                                                                                                                                           |
| :------------- | :----------------------------------------------------------------------------------------------------------------------------------------------- |
| Frame sampling | Deterministic timestamp-target **2 fps** sampling; a frame is accepted only if it is within 1/30 s of a 0.5 s grid point (same tolerance as WIN) |
| Face detection | MTCNN (`keep_all=True`); the best face is picked by (probability ↓, area ↓, position ↑)                                                          |
| Alignment      | 5-point landmark similarity transform (`estimateAffinePartial2D`, LMEDS) onto a fixed 224×224 face template                                      |
| Embedding      | `weights.transforms()` preprocessing → backbone with classifier stripped (`children()[:-1]`) → **(576,)** per frame                              |
| Pooling        | Mean over all valid frames → one **(576,)** vector per video                                                                                     |
| Failure mode   | No valid face in any sampled frame → zero vector + `valid=False` (mask 0)                                                                        |

### 3.2 Temporal Member (WIN A05 regional recipe, video-level)

| Stage          | Detail                                                                                                                                            |
| :------------- | :------------------------------------------------------------------------------------------------------------------------------------------------ |
| Frame plan     | **32 segments × 2 frames = 64 frames**, anchored at each segment midpoint, pair spacing 100 ms; only planned indices are decoded                  |
| Face detection | MTCNN on batches of 16 frames, with 5-point landmarks                                                                                             |
| Alignment      | **Eyes-only similarity transform** onto fixed eye targets (72, 85) / (152, 85) — a 2-point alignment that is robust where 5-point alignment fails |
| Region stack   | Each aligned crop is cut into 5 region boxes — full / eyes / mouth / left cheek / right cheek — each resized to 224×224                           |
| Embedding      | Manual ImageNet normalization → backbone `features + avgpool` → **(64, 5, 576)** per video, plus `frame_valid (64,)` and segment indices          |
| Failure mode   | Per-frame validity: a frame with no detectable face contributes zeros and `frame_valid=False`; the model's attention pooling skips invalid frames |

The temporal member uses only the **4 fine regions** (eyes, mouth, cheeks) — the full-face box is extracted and stored but unused, exactly as in WIN's A05 configuration.

### 3.3 Why these exact two members?

Because their error complementarity is not a hypothesis — it is a **measured result** on this corpus (WIN FINDINGS.md): static wins where holistic appearance carries the signal (grooming, fatigue, posture-adjacent framing), temporal wins where dynamics carry it (micro-movements, eye activity), and the videos where each fails are largely disjoint. In this experiment the same ordering reappears: temporal-only Testing MAE 7.16 < static-only 7.28 (Section 7.3).

---

## 4. Architecture Deep-Dive: How Data Flows

```text
┌────────────────────────────────────────────────────────────────────────────────┐
│ 1. INPUTS                                                                       │
│    static_feat (B, 576)    temporal: seq (B,64,4,48), seq_mask (B,64),          │
│    clip (B, 512)           motion (B,4,48),  rppg (B, 9),  smile (B, 88)        │
└────────────────────────────────────────────────────────────────────────────────┘
                    │                              │
                    ▼                              ▼
┌──────────────────────────────────┐   ┌────────────────────────────────────────┐
│ 2a. STATIC MEMBER                │   │ 2b. TEMPORAL MEMBER (43,777 params)    │
│    ModalityProjector             │   │    AttentionPool over 64 frames        │
│      Linear(576→128)             │   │      per region  (B,4,48)              │
│      LayerNorm + ReLU            │   │    Linear(48→64) projections           │
│      Dropout(0.3)                │   │      + region & stream embeddings      │
│    MemberHead                    │   │    1-layer pre-norm Transformer        │
│      LN→Linear(128→32)→GELU      │   │      d=64, 4 heads, FFN 64, drop 0.5  │
│      →Dropout→Linear(32→2)       │   │    mean-pool tokens → Linear(64→128)   │
│    outputs (ŷ_s, s_s)            │   │    MemberHead → (ŷ_t, s_t)             │
└───────────────┬──────────────────┘   └──────────────────┬─────────────────────┘
                │                                         │
                ▼                                         ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ 3. COMPOSITE VISUAL DOCTOR — NESTED INVERSE-VARIANCE FUSION                     │
│    p_s = exp(-s_s),  p_t = exp(-s_t)          (missing member → s = +8.0)      │
│    w_s = p_s/(p_s+p_t),   w_t = p_t/(p_s+p_t)                                   │
│    e_vis  = w_s·e_s + w_t·e_t                    (128-D doctor embedding)       │
│    ŷ_vis  = w_s·ŷ_s + w_t·ŷ_t                    (doctor standalone score)      │
│    log σ²_vis = -logaddexp(-s_s, -s_t)  → σ²_vis = 1/(p_s+p_t) ≤ min(σ²_s,σ²_t) │
└───────────────────────────────┬────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┬───────────────────────┐
        ▼                       ▼                       ▼                       ▼
┌──────────────┐       ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
│ Dr. Visual   │       │ Dr. CLIP     │        │ Dr. rPPG     │        │ Dr. Smile    │
│ e_vis, s_vis │       │ proj+head    │        │ proj+head    │        │ proj+head    │
└──────┬───────┘       └──────┬───────┘        └──────┬───────┘        └──────┬───────┘
       └──────────────────────┴────────┬───────────────┴───────────────────────┘
                                       ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ 4. OUTER PANEL — inverse-variance fusion over 4 doctors (baseline recipe)       │
│    hard floor s=+8.0 for missing doctors;  weights w_d = p_d/Σp;               │
│    fused = Σ w_d·e_d  →  Linear(128→64) → ReLU → Linear(64→1) → BDI-II         │
└────────────────────────────────────────────────────────────────────────────────┘
```

**Parameter inventory** (227,564 total — 41% fewer than the baseline evidence model's 383,753):

| Component                  |                                          Parameters |
| :------------------------- | --------------------------------------------------: |
| Static projector + head    |                                              78,562 |
| Temporal member + head     | 48,227 (member alone: 43,777 ≈ WIN's 43,203 budget) |
| CLIP projector + head      |                                              70,370 |
| rPPG projector + head      |                                               5,986 |
| openSMILE projector + head |                                              16,098 |
| Final prediction head      |                                               8,321 |
| **Total**                  |                                         **227,564** |

---

## 5. The Mathematics of Nested Evidence Fusion

### 5.1 Each member is a full diagnostician

Both members and all three scalar doctors end in a `MemberHead`: a 128-D embedding → $(\hat{y}_m, s_m)$ where $s_m = \log \sigma_m^2$ is clamped to $[-8, +8]$. There are **five** such heads in the system (static, temporal, clip, rppg, smile) — each trained with its own heteroscedastic loss (Section 6.3).

### 5.2 The nested blend

Inside the doctor, precisions and weights are computed exactly like the outer panel, but over two members:

$$p_m = e^{-s_m}, \qquad w_m = \frac{p_m}{p_s + p_t}, \qquad \hat{y}_{\text{vis}} = w_s \hat{y}_s + w_t \hat{y}_t$$

A missing member gets the hard floor $s = +8.0$, so $p \approx 0.000335$ and the surviving member receives weight $\approx 1.0$ — the doctor gracefully becomes that member.

### 5.3 The key inequality: the doctor is sharper than its parts

The variance of the precision-weighted mean is the reciprocal of the summed precision:

$$\sigma^2_{\text{vis}} = \frac{1}{p_s + p_t} \;\le\; \frac{1}{p_s} = \sigma^2_s \qquad\text{and likewise}\qquad \sigma^2_{\text{vis}} \le \sigma^2_t$$

In log-space this is computed stably as $\log \sigma^2_{\text{vis}} = -\mathrm{logaddexp}(-s_s, -s_t)$. **When both members are confident, the doctor is more certain than either member alone** — two independent pieces of evidence genuinely stack. When one fails (or is missing), the doctor defers to the other at no loss. This is the structural encoding of "complementary errors."

### 5.4 How member disagreement is paid for

The formula in 5.3 assumes unbiased members. The mechanism that makes it honest is the **heteroscedastic training loss**: a member that is systematically wrong on some subpopulation (e.g., static on severe cases) learns a high $s_m$ on those very samples, because false confidence is exponentially punished (Section 5.3 of the evidence-fusion doc). So on the samples where static errs, $p_s$ collapses, $w_t \to 1$, and $\sigma^2_{\text{vis}} \to \sigma^2_t$: the doctor automatically routes around its failing member — _without any explicit routing machinery_. The routing is an emergent property of calibrated uncertainty.

### 5.5 The doctor as one outer citizen

The doctor exposes $(\mathbf{e}_{\text{vis}}, \hat{y}_{\text{vis}}, s_{\text{vis}})$ to the outer panel exactly as a single projector would. If **both** members are missing, the doctor's log-variance is hard-floored to +8.0, and the outer panel mutes the entire visual channel (weight ≈ 0.0001) and redistributes to CLIP / rPPG / openSMILE. Two independent levels of degradation:

```text
Member fails      →  doctor reroutes to the other member (nested level)
Whole doctor gone →  panel reroutes to the other doctors  (outer level)
```

### 5.6 What nested fusion _cannot_ do (important!)

Inverse-variance fusion computes a precision-weighted **mean** of member scores. It is the optimal combination rule **only for unbiased estimators**. If _every_ member systematically under-predicts a subpopulation — a shared bias — the weighted mean inherits that bias, and the doctor's extra sharpness can even make the confident-but-wrong consensus harder to overturn. **Fusion cures variance (disagreement); it cannot cure bias (shared blindness).** Section 9 shows this is precisely the severe-band situation, which is why the fix must come from the loss/data side, not from more fusion machinery.

---

## 6. Training Protocol: Fold-Local Preprocessing, Two-Level Dropout, Dual Optimizer Recipes

### 6.1 Sample set and leak-free cross-validation

- **297 video-level samples** (98 Training / 99 Development / 100 Testing) — the exact intersection of all 6 sources, matching the baseline's sample set (3 videos lack rPPG features and are dropped by both systems: `234_1_Freeform`, `241_2_Freeform`, `308_3_Freeform`).
- Training uses **Training + Development concatenated (197 samples, 58 subjects)**, `GroupKFold(5)` grouped by subject ID (`video_id.split("_")[0]`), with an explicit per-fold overlap check that raises `RuntimeError` on leakage.
- **Testing is touched exactly once**, by `test_composite_model.py`, after all training decisions are frozen.
- Scalar modalities (static / clip / rppg / smile) are z-scored with **Training-split statistics only** (baseline convention). A video whose visual extraction entirely failed would be kept with visual mask 0 — in practice all 297 extractions succeeded (static 297/297, temporal 297/297).

### 6.2 Fold-local temporal preprocessing (WIN discipline)

The temporal member's raw features are **not** normalized with global statistics. Inside each fold, a `TemporalPreprocessor` is **fit on the train rows only** and then applied to that fold's validation rows (and later, the same fitted instance transforms Testing rows):

1. **Per-region MissingAwareZScore** — mean/std computed over _observed_ frames only, low-variance dims zeroed, clipped to ±10.
2. **DeterministicPCA to 48 dims per region** — SVD with sign-fixing (the largest-magnitude entry of each component is forced positive), fit on observed train-fold rows; yields 4 × 48 appearance tokens per video.
3. **Motion tokens** — mean absolute delta between the two frames of each complete 2-frame segment, per region: 4 × 48 motion tokens.

This is the exact WIN recipe, and it is what keeps the PCA whitening leak-free: no validation or testing statistics ever enter the pipeline. Each fold's fitted preprocessor is saved alongside its checkpoint (as plain NPZ, bit-exact reloadable — no pickle).

### 6.3 Two-level modality dropout

Training-time augmentation over the availability hierarchy, each level at $p = 0.15$:

| Level      | What can be dropped               | Safety guarantee                                                     |
| :--------- | :-------------------------------- | :------------------------------------------------------------------- |
| Sub-branch | static member or temporal member  | if both drawn off, the first _available_ one is flipped back on      |
| Doctor     | the whole visual doctor           | doctor can only be "alive" if ≥ 1 sub-branch is alive                |
| Outer      | clip / rppg / smile independently | if all doctors drawn off, they are flipped back on in fallback order |

Crucially, the draws always **honor true base availability**: dropout can remove a member that exists but can never resurrect one that is genuinely missing. Dropped features are zeroed and their mask entries set to 0, so the hard-floor suppressor mutes them exactly as at test time.

### 6.4 Loss

$$\mathcal{L}_{\text{total}} = \underbrace{\text{SmoothL1}(\hat{y}_{\text{final}}, y^*)}_{\text{main}} \;+\; 0.3 \cdot \underbrace{\frac{1}{\sum_m \text{mask}_m}\sum_{m \,\mid\, \text{mask}_m=1} \tfrac{1}{2}e^{-s_m}(\hat{y}_m - y^*)^2 + \tfrac{1}{2}s_m}_{\text{heteroscedastic loss over the FIVE heads}}$$

The doctor's own derived score gets **no separate loss** — gradients reach the members through the nested weights, so a member is pushed to be accurate _and_ to price its uncertainty honestly.

### 6.5 Dual optimizer recipes

One AdamW with two parameter groups, honoring each sub-network's proven training regime:

| Group    | Parameters                 |   LR | Weight decay | Source of recipe                                                 |
| :------- | :------------------------- | ---: | -----------: | :--------------------------------------------------------------- |
| Temporal | temporal member + its head | 3e-4 |         0.05 | WIN regional member (delicate Transformer, heavy regularization) |
| Panel    | everything else            | 1e-3 |         1e-2 | baseline evidence model                                          |

Gradient norm clipped at 1.0; batch 16; max 120 epochs; early stopping on validation MAE with patience 15 (min improvement 1e-4) and best-state restore; seed 42 + fold index. The 5 best fold checkpoints form the ensemble (mean prediction), evaluated once on Testing.

---

## 7. Results: New Best MAE With Two Levels of Graceful Degradation

### 7.1 Headline benchmark (Testing split, 100 videos, 5-fold ensemble)

| Model                       |    MAE ↓ |   RMSE ↓ | PCC ↑ | CCC ↑ |      Params |
| :-------------------------- | -------: | -------: | ----: | ----: | ----------: |
| Single-MLP baseline         |     8.15 |     9.90 |  0.54 |  0.49 |           — |
| MoE fusion                  |     7.73 |    10.00 |  0.51 |  0.45 |       ~378k |
| Self-attention fusion       |   7.3274 |     9.58 |  0.58 |  0.55 |       ~614k |
| Evidence fusion (control)   |     7.29 |     9.49 |  0.56 |  0.47 |     383,753 |
| **Composite visual fusion** | **6.99** | **9.49** |  0.57 |  0.52 | **227,564** |

Out-of-fold (Training+Development, 197 samples): **MAE 6.92, RMSE 9.16, PCC 0.649, CCC 0.586** — OOF ≈ Test (6.92 vs 6.99), a healthy generalization signal. Per-fold Testing MAEs are tight: 7.59 / 7.55 / 7.16 / 7.17 / 7.24.

### 7.2 Missing-modality stress test (13 scenarios, Testing split)

| Scenario                | Composite MAE |             Baseline ref | Drift |
| :---------------------- | ------------: | -----------------------: | ----: |
| all present             |     **6.992** |                     7.29 |     — |
| missing rppg            |         6.982 |                     7.26 | +0.00 |
| missing smile           |         7.067 |                        — | +0.08 |
| missing clip            |     **7.793** |                     8.47 | +0.80 |
| missing visual doctor   |         7.226 | 7.35 (baseline "visual") | +0.23 |
| visual static-only      |         7.277 |                        — | +0.29 |
| visual temporal-only    |         7.160 |                        — | +0.17 |
| visual both members out |         7.226 |                        — | +0.23 |
| missing visual+clip     |     **9.360** |                    10.59 | +2.37 |
| missing visual+rppg     |         7.450 |                        — | +0.46 |
| missing clip+rppg       |         7.657 |                        — | +0.67 |
| only visual             |         7.779 |                        — | +0.79 |
| only clip               |         7.862 |                        — | +0.87 |

Three findings worth highlighting:

1. **The complementary-error design works as intended.** Losing _either_ visual member costs only 0.17–0.29 MAE, and losing the entire visual doctor costs 0.23 — the panel absorbs member failures almost for free. ("Both members out" exactly equals "missing visual doctor," as it must — a built-in sanity check.)
2. **The composite visual doctor carries more information than the baseline's visual branch ever did.** Removing it degrades the composite by +0.23, whereas removing the baseline's visual branch degraded the baseline by only +0.06. The WIN features are genuinely stronger evidence.
3. **Every shared scenario beats the baseline's absolute numbers** — the robustness advantage is uniform, not scenario-specific (e.g., visual+clip blackout: 9.36 vs 10.59).

Also note the ordering **temporal-only (7.16) < static-only (7.28)** — the same member hierarchy WIN measured (regional 7.48 < static 8.31), reproduced independently inside a different fusion architecture.

### 7.3 The spread audit — and the one thing that did NOT improve

| Diagnostic                 | Value      | Interpretation                                          |
| :------------------------- | :--------- | :------------------------------------------------------ |
| True label std (Testing)   | 11.48      | —                                                       |
| Predicted std (Testing)    | **7.42**   | variance collapse persists (WIN: 8.2)                   |
| Residual–label correlation | **−0.765** | systematic under-prediction of high labels (WIN: −0.72) |

Per-band Testing breakdown (the model's predicted mean vs the true mean of each band):

| BDI-II band        |      n |        MAE | Over-predicted | Pred mean | True mean |
| :----------------- | -----: | ---------: | -------------: | --------: | --------: |
| minimal (0–13)     |     50 |      6.137 |      **42/50** |     10.41 |      4.96 |
| mild (14–19)       |     20 |      3.919 |           7/20 |     14.72 |     16.20 |
| moderate (20–28)   |     16 |      5.392 |           6/16 |     21.44 |     25.00 |
| **severe (29–63)** | **14** | **16.262** |       **0/14** | **17.88** | **34.14** |

The pattern is a textbook **two-sided squeeze**: minimal cases are dragged up (+5.45 mean bias), severe cases are dragged down (−16.26), and the model's effective output range is roughly **[10, 28]** against a true range of [0, 48]. The overall 6.99 MAE is earned almost entirely on the 86 non-severe videos; the severe band contributes 14/100 × 16.26 ≈ **2.28 MAE points** on its own.

---

## 8. Caveats & Honest Assessment

1. **The 0.30 MAE margin over the control is near the noise floor.** The repo's own documentation estimates ~0.4 MAE of run-to-run noise on this corpus. The _robust_ claims are: (a) it beats all four baselines simultaneously, (b) it dominates the baseline in every shared missing-modality scenario, and (c) it does so with 41% fewer parameters. The single-number 6.99-vs-7.29 gap should be reported with that caveat.
2. **The severe band did not improve — it slightly worsened** (16.26 vs WIN's session-level 15.3, though the units differ: 14 videos here vs 7 sessions there, so the numbers are not directly comparable). The range-squash failure mode survived the architecture change completely intact. Section 9 is about exactly this.
3. **CV fold variance is higher than the baseline's.** Per-fold validation MAEs were 5.32 / 6.58 / 8.93 / 5.30 / 8.52 (std 1.54) — folds 3 and 5 held hard subjects — versus the baseline evidence model's celebrated std 0.12. The heteroscedastic auxiliary loss still stabilizes training (no fold diverged), but the composite is a more complex function of a small dataset. The _Testing_ fold spread is much tighter (7.16–7.59), and the ensemble absorbs the fold variance.
4. **All 297 visual extractions succeeded** (0 invalid static, 0 no-face temporal), so the missing-member machinery was exercised only via dropout, never by a genuinely failed video. The stress-test scenarios are therefore the only evidence of real failure behavior.
5. **Frozen ImageNet backbones.** Neither member fine-tunes its visual encoder — all discrimination comes from ~227k head parameters over generic features. This is a strength (no overfitting a 197-sample dataset with a 2M-parameter backbone) and a ceiling (Section 9, Tier T3).
6. **One-time Testing access maintained.** Testing was evaluated exactly once, after all design decisions were frozen; the stress test reuses the same frozen ensemble with masks only.

---

## 9. The Severe-Band Problem: Diagnosis and the Road to Sub-6.5 MAE

### 9.1 Why the composite could not have fixed it (the bias/variance argument)

The evidence panel is a precision-weighted **mean** combiner. Section 5.6 showed why that is exactly the wrong tool for this failure: severe under-prediction is a **shared bias**, not disagreement. Every one of the five heads is trained on 197 samples of which only **34 (17%)** are severe (BDI ≥ 29), under a symmetric SmoothL1 loss. The Bayes-optimal answer to "minimize symmetric loss with scarce high labels" is a conditional mean — pulled toward the middle. All members inherit it; fusing them averages five copies of the same bias; the doctor's sharpening (σ²_vis ≤ min σ²_m) even makes the biased consensus _more confident_. **No amount of fusion machinery fixes this. The levers are the loss, the data, and the backbone capacity.**

The arithmetic of why this is THE lever: the severe band contributes 2.28 of the 6.99 Testing MAE, and the minimal band's over-prediction contributes 3.07.

| If severe-band MAE becomes… | Overall Testing MAE becomes |
| --------------------------: | --------------------------: |
|             16.26 (current) |                        6.99 |
|                        12.0 |                        6.40 |
|                        10.0 |                    **6.12** |
|                         8.0 |                        5.84 |

(Note that a spread-preserving fix also shrinks the minimal band's +5.45 bias at the same time, so these figures _underestimate_ the payoff — each point of severe improvement typically buys extra minimal/mild improvement for free.)

### 9.2 Roadmap, in priority order

#### Tier T0 — Post-hoc spread calibration on OOF (hours; zero training risk)

Fit a single affine correction $\hat{y}' = a\hat{y} + b$ (or isotonic map) on the **OOF predictions only** (197 samples), choosing $(a, b)$ to minimize band-balanced MAE while matching the OOF label std (the post-hoc form of the "spread rule" from the WIN A07 blueprint). OOF pred std is 7.68 vs label std 11.93 — a slope near $a \approx 1.4$ restores the spread. Apply to the frozen Test ensemble predictions.

- **Expected**: modest (~0.1–0.3 overall), and it _will_ trade some minimal-band accuracy for severe accuracy — the point is partly diagnostic: it upper-bounds what "the model knows but doesn't say."
- **Discipline**: fit on OOF only; Testing is not touched during fitting.

#### Tier T1 — Loss surgery (days; the primary lever) — _recommended first_

Extend `combined_loss_composite` with two or three additive terms:

1. **Spread-preservation penalty**: per training batch, $\lambda_{\text{spread}} \cdot \left(\mathrm{std}(\hat{y}) - \mathrm{std}(y)\right)^2$. Directly attacks the [10, 28] output collapse, and because it widens _both_ tails it helps the minimal band and the severe band simultaneously.
2. **Asymmetric band-weighted error**: replace plain SmoothL1 with per-sample weights $w(y^*)$ (e.g., ×3 for $y^* \ge 29$) and/or an asymmetric penalty that charges more for under-prediction on high labels: $\beta \cdot \max(0, y^* - \hat{y})^2$ for $y^* \ge 29$.
3. **Pairwise ranking loss**: within each batch, for pairs with $|y_i - y_j| \ge 10$, a hinge on the sign of $(\hat{y}_i - \hat{y}_j)$. MAE is blind to ordering; ranking is not. This is the ranking-loss item of the KD_PLAN / A07 blueprint.

Acceptance gates (echoing KD_PLAN's abort rules — learned from the A07 EXP0006 trap where a 7.042 model hid a _worse_ variance collapse): a change is accepted only if **severe MAE ↓ AND pred std ↑ AND overall MAE ≤ ~6.8**. If overall improves but severe worsens — reject; that is another collapse in disguise.

#### Tier T2 — Data-side interventions (days)

1. **Severe-aware fold sampling**: oversample the 34 severe OOF videos within train folds (group-aware, capped ×3) or apply focal-style weighting. Cheapest way to blunt the 17% imbalance.
2. **Watch-list audit (post-hoc only)**: WIN FINDINGS identified buckets of severe-labeled videos where the pixels genuinely do not look severe (bucket C: questionnaire-vs-pixels mismatch) or the face is unusable (bucket A). No loss can fix a video that shows no distress — but the architecture has a native answer: flag such videos with a raised log-variance floor so the doctor formally distrusts them rather than fitting them. (Audit after training; never train on the audit.)
3. **Session-level aggregation at inference**: average the panel's predictions over the Freeform and Northwind videos of the same subject-session. The sample set contains both tasks per subject; WIN's session-level unit showed this smoothing helps, and it requires zero retraining.

#### Tier T3 — Capacity: heavy-backbone distillation (1–2 weeks; only if T1/T2 stall)

The 576-D frozen MobileNetV3-Small features are the input-side ceiling: if the encoder simply cannot _see_ severe depression, no loss can regress what was never extracted. The A07 blueprint's strongest forward path: extract static-member features with a heavy backbone (ConvNeXt-Tiny / ViT-B), then either (a) train the same heads on the richer features (cheap swap — re-extract 300 videos, rerun), or (b) distill the heavy teacher into the deployable student. Combine with the KD_PLAN E1–E4 privileged-information scheme (teacher sees the BDI bin during training only) for spread-preserving soft targets.

#### What would falsify this plan

If T1 raises pred std but PCC _drops_, the added variance is not label-correlated — the information isn't in the features, and the answer is T3 (capacity), not more loss engineering. That single diagnostic separates "the model knows but is too timid" from "the model cannot know."

### 9.3 Recommended sequence

**T0 this week** (it quantifies the timidity gap) → **T1 spread + asymmetric loss** (highest expected MAE per engineering hour) → **T2 severe oversampling + session aggregation** → **T3** only if the T1 diagnostic says capacity is the binding constraint. With severe MAE brought to ~10–12 — the level T1+T2 plausibly reach — the overall score lands at **6.1–6.4**, and every robustness advantage of Section 7.2 is retained because the fusion architecture itself is untouched by these interventions.

---

## 10. Step-by-Step Reproduction & Execution Guide

All commands from `Facial_Expression_Depression_Recognition/composite_visual_fusion/` (project venv activated). Steps 2–3 are resume-safe: re-running skips already-extracted videos. Extraction has already been run once; the features are on disk, so a fresh training run starts at step 5.

```text
python win_extraction.py                      # 0. MTCNN/backbone smoke test (2 videos)
python extract_static_visual.py               # 1. static features  -> 300 NPZs  (hours)
python extract_temporal_visual.py             # 2. temporal features -> 300 NPZs (hours)
python dataset_composite.py                   # 3. align 6 sources -> 297 samples, 3 split NPZs
python train_cv_safe_composite.py             # 4. 5-fold CV training (~2 min total on CUDA)
python test_composite_model.py                # 5. ONE-TIME Testing evaluation + spread audit
python evaluate_missing_modality_composite.py # 6. 13-scenario stress test
```

Unit tests (fast, no data required):

```text
python models_composite.py                    # shapes, weight sums, hard-floor behavior
python composite_training_utils.py            # dropout guarantees, preprocessor round trip, loss backward
```

All artifacts land under `output/composite_visual_branch/`:

```text
static_visual/<video_id>.npz                  (576,) + valid
temporal_visual/<video_id>.npz                (64, 5, 576) + frame_valid + segments
aligned_dataset/{training,development,testing}.npz + normalization_stats.npz
composite_models/composite_fold_{1..5}.pt     fold checkpoints (weights_only-loadable)
composite_models/composite_fold_{1..5}_preprocessor.npz   fold-local temporal preprocessors
composite_models/cv_results.txt               full training log + OOF summary
composite_models/oof_predictions.csv          per-video OOF predictions (input for Tier T0)
composite_models/test_predictions.csv         per-video Test predictions, ensemble + 5 folds
composite_models/test_results.txt             Testing report incl. spread audit
composite_models/missing_modality_results.txt 13-scenario stress table
```
