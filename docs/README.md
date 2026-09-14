# Automated Depression Assessment (ADA) via AI & Deep Learning
### A Comprehensive, Beginner-Friendly Guide to the 4 Research Papers in this Repository

> **Target Audience Note**: This document is written specifically for undergraduate students and researchers new to machine learning, computer vision, and affective computing. No PhD or prior specialized knowledge is assumed. Jargon is demystified with plain English explanations, analogies, and step-by-step visual architecture breakdowns.

---

## Table of Contents
1. [Executive Overview of the Repository](#1-executive-overview-of-the-repository)
2. [The "No-PhD" Field Guide: Foundational Concepts & Terminology](#2-the-no-phd-field-guide-foundational-concepts--terminology)
   - [Clinical Measures: BDI-II, HAMD, and PHQ-8](#clinical-measures-how-doctors-measure-depression)
   - [Standard Benchmark Datasets (AVEC, DAIC-WOZ, CZ2024)](#standard-benchmark-datasets)
   - [Performance Metrics: RMSE, MAE, PCC, and CCC](#performance-metrics-how-we-grade-ai-models)
   - [Modalities & Features (Action Units, Gaze, Prosody, Spectrograms)](#modalities--features-what-the-ai-observes)
   - [Core Deep Learning Building Blocks Explained](#core-deep-learning-building-blocks-explained)
3. [Paper 1: Explainable Depression Assessment from Face Videos by Weakly Supervised Learning (AAAI 2026)](#3-paper-1-explainable-depression-assessment-from-face-videos-by-weakly-supervised-learning-aaai-2026)
4. [Paper 2: Multi-Time Scale Feature Extraction and Attention Networks for Automatic Depression Level Prediction (Applied Soft Computing 2026)](#4-paper-2-multi-time-scale-feature-extraction-and-attention-networks-for-automatic-depression-level-prediction-applied-soft-computing-2026)
5. [Paper 3: Deep Multi-Modal Network Based Automated Depression Severity Estimation (IEEE TAFFC 2023)](#5-paper-3-deep-multi-modal-network-based-automated-depression-severity-estimation-ieee-taffc-2023)
6. [Paper 4: ERBMA-Net: Enhanced Random Binary Multilevel Attention Network for Facial Depression Recognition](#6-paper-4-erbma-net-enhanced-random-binary-multilevel-attention-network-for-facial-depression-recognition)
7. [Side-by-Side Comparative Matrix of All 4 Papers](#7-side-by-side-comparative-matrix-of-all-4-papers)
8. [Evolutionary Timeline & Key Takeaways for Undergrad Researchers](#8-evolutionary-timeline--key-takeaways-for-undergrad-researchers)

---

## 1. Executive Overview of the Repository

This repository contains four state-of-the-art research papers tackling **Automated Depression Assessment (ADA)** using Artificial Intelligence. 

### Why is this field important?
Depression (Major Depressive Disorder) affects over 280 million people worldwide and is a leading cause of disability. Currently, diagnosing depression relies heavily on face-to-face psychiatric interviews and self-report surveys. These traditional methods have serious bottlenecks:
1. **Subjectivity and Bias**: Patients may downplay or exaggerate symptoms, or struggle to put their feelings into words due to social stigma or memory recall issues.
2. **Shortage of Specialists**: There simply aren't enough psychiatrists and clinical psychologists to evaluate everyone who needs help.
3. **Logistical Burden**: In-person clinical assessments are time-consuming and expensive.

### The AI Solution
When a person experiences depression, it involuntarily affects how their brain controls their body. These show up as nonverbal biological cues:
- **Facial expressions**: Flatter affect, less smiling, slower smiling, drooping mouth, brow furrowing, reduced eye contact.
- **Speech/Vocal patterns**: Slower speaking rate, monotonic pitch (lack of vocal inflection), longer pauses between words, reduced loudness.
- **Head movements & Gaze**: Looking downwards, infrequent upward nodding, rigid head posture.

The four papers in this repository use computer vision and audio deep learning to automatically detect and quantify depression severity from recorded interviews.

| File Name in Repo | Short Name | Primary Modality | Core Innovation |
| :--- | :--- | :--- | :--- |
| `11079-AAAI26.LiaoR-CO.pdf` | **ExpADA** (AAAI 2026) | Face Video | Weakly Supervised Learning to pick out multi-scale segments that actually show depressive cues, providing timestamps/explainability. |
| `Applied_Soft_Computing_journal-7.pdf` | **MSFE-CTA** (App. Soft Comp. 2026) | Face Video | Ultra-lightweight (0.85M params) Inception-TCN with logarithmic dilations + Dilated Channel & Temporal Attention across milliseconds to minutes. |
| `Deep_Multi_Modal_Network_Based_Automated_Depression_Severity_Estimation.pdf` | **Deep Multi-Modal** (IEEE TAFFC 2023) | Audio + Face Video | Novel VLDSP texture descriptor + 1D/2D ResNets + Temporal Attentive Pooling (TAP) + Multimodal Factorized Bilinear (MFB) fusion. |
| `ERBMA-Net-Enhanced-Random-Binary-Multilevel-Attention-Network-for-Facial-Depression-Recognition (1).pdf` | **ERBMA-Net** | Face Video (Face, Eyes, Mouth) | Enhanced Random Binary CNNs (stochastic binary filters + refinement) + Coordinate Attention + Self-Attention on real clinical hospital data. |

---

## 2. The "No-PhD" Field Guide: Foundational Concepts & Terminology

Before diving into individual papers, here is an intuitive dictionary of all the concepts you will encounter.

### Clinical Measures: How Doctors Measure Depression
AI models in these papers do not output a vague "depressed or not" guess; they predict continuous clinical severity scores:

1. **BDI-II (Beck Depression Inventory-II)**:
   - A 21-question self-report inventory scored from **0 to 63**.
   - Standard clinical cutoffs:
     - `0 - 13`: Minimal / No Depression
     - `14 - 19`: Mild Depression
     - `20 - 28`: Moderate Depression
     - `29 - 63`: Severe Depression
2. **HAMD (Hamilton Depression Rating Scale)**:
   - Administered directly by a clinician (usually 17 or 21 items).
   - Scored from **0 to 50+**:
     - `0 - 7`: Normal / Minimal
     - `8 - 19`: Mild
     - `20 - 34`: Moderate
     - `≥ 35`: Severe
3. **PHQ-8 (Patient Health Questionnaire-8)**:
   - An 8-item survey scored from **0 to 24**. Used heavily in US clinical and virtual agent benchmarks (e.g., DAIC-WOZ).

---

### Standard Benchmark Datasets
Collecting mental health video data is extremely difficult due to strict privacy and medical ethics laws. Researchers worldwide benchmark their algorithms on shared, anonymized challenges:

1. **AVEC 2013 & AVEC 2014 (Audio/Visual Emotion Challenge)**:
   - Recorded at Ulm University, Germany. 82 to 84 native German speakers performing Human-Computer Interaction tasks.
   - Ground truth: BDI-II scores (0–63).
   - In AVEC 2014, subjects perform two distinct tasks:
     - **Northwind**: Participants read aloud a fixed fable ("The North Wind and the Sun"). Everyone says the exact same words, so differences in facial movement and voice are purely behavioral, not vocabulary-driven.
     - **Freeform**: Participants answer open emotional questions (e.g., "Describe a sad childhood memory" or "What is your favorite dish?"). This reveals spontaneous emotional responses.
2. **AVEC 2017 & 2019 (DAIC-WOZ / E-DAIC)**:
   - Distress Analysis Interview Corpus. English speakers interacting with an animated virtual interviewer (an AI avatar named "Ellie").
   - Ground truth: PHQ-8 scores (0–24).
3. **CZ2024 (Changzhou No. 2 People's Hospital Dataset)**:
   - Introduced in Paper 4. A real-world clinical dataset of 401 psychiatric patients in China diagnosed with clinical depression, rated via HAMD scores.

---

### Performance Metrics: How We Grade AI Models
Since the target is predicting a continuous score (e.g., predicting 24.5 when the true score is 26.0), this is treated as a **regression task**:

1. **MAE (Mean Absolute Error)**:
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$
   - *Plain English*: On average, how many points off is the AI's prediction? If true BDI-II is 20 and AI says 25, the absolute error is 5. Lower is better.
2. **RMSE (Root Mean Square Error)**:
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$
   - *Plain English*: Similar to MAE, but it squares the differences before averaging. This heavily penalizes large catastrophic errors (e.g., predicting 5 when the patient is at 45). Lower is better.
3. **PCC (Pearson Correlation Coefficient)**:
   - Measures how well the predicted scores move in the same direction as the true scores (ranges from -1.0 to +1.0). Closer to +1.0 means great correlation.
4. **CCC (Concordance Correlation Coefficient)**:
   - Stricter than PCC: checks not only if predictions correlate, but if they lie directly on the $45^\circ$ line ($y = \hat{y}$).

---

### Modalities & Features: What the AI Observes

1. **Action Units (AUs)**:
   - Defined by the Facial Action Coding System (FACS). Instead of vague labels like "happy", FACS breaks the face into tiny individual muscle movements:
     - `AU 12`: Lip corner puller (smiling)
     - `AU 14`: Dimpler (tightening lip corners, often associated with forced or non-engaging expressions)
     - `AU 15`: Lip corner depressor (frowning)
     - Depressed individuals typically show fewer socially engaging AUs (like AU 12) and more non-engaging or tense AUs (like AU 14).
2. **Gaze and Head Pose**:
   - 3D pitch, yaw, and roll of the head, and eye direction vectors. Depressed individuals spend significantly more time looking down and avoiding eye contact.
3. **OpenFace Toolkit**:
   - A widely used computer vision software that detects faces in video and automatically outputs AU intensities, 68 facial landmarks, head pose, and eye gaze for every frame.
4. **Audio Features**:
   - **Spectrogram**: A visual image of audio showing frequencies over time.
   - **MFCCs (Mel-Frequency Cepstral Coefficients)**: A compact mathematical representation of the timbre/shape of the vocal tract.
   - **Raw Waveform**: Slicing the actual 1D sound wave amplitudes directly.

---

### Core Deep Learning Building Blocks Explained

1. **1D vs 2D vs 3D CNNs**:
   - **1D CNN**: Slides a filter across a 1-dimensional sequence (e.g., sound waveform over time, or a series of AU values over time). Very fast!
   - **2D CNN**: Slides filters over height and width (images). Great for static facial expressions.
   - **3D CNN**: Slides filters over height, width, AND time (video clips). Captures appearance and motion together, but requires massive GPU memory and millions of parameters.
2. **TCN (Temporal Convolutional Network) & Dilated Convolutions**:
   - Regular convolutions only see 3 adjacent frames at a time. To see what happened 5 minutes ago, you would need hundreds of layers!
   - **Dilated Convolutions**: The convolution filter skips steps like a comb ($d=1, 2, 4, 8, \dots$). With each layer, its "view" (receptive field) grows exponentially without adding extra weights or computations. It can easily connect milliseconds to minutes.
3. **Attention Mechanisms (Spatial, Temporal, Channel, Self-Attention)**:
   - *Analogy*: When reading a medical chart, a doctor doesn't read every word with equal focus; their eyes jump straight to the abnormal lab values.
   - **Spatial Attention**: Asks *where* to look on the face (e.g., eyes and mouth rather than ears or forehead).
   - **Temporal Attention**: Asks *when* to look (e.g., the moment the person sighs or looks away vs. when they are just blinking).
   - **Channel Attention**: Asks *which features* are most informative (e.g., mouth droop features vs. lighting changes).
   - **Self-Attention**: Compares every location in a feature map with every other location to see how distant parts relate (e.g., how the eyes relate to the mouth shape simultaneously).
4. **Binary Networks (LBCNN vs RBCNN)**:
   - Standard neural networks use 32-bit floating point numbers ($0.385721\dots$). Multiplying millions of floats is power-hungry.
   - **Binary Convolutions**: Constrain weights to just $\{-1, 0, +1\}$. Multiplications turn into simple additions and subtractions!
   - **LBCNN (Local Binary CNN)**: Uses fixed, pre-calculated binary filters. Fast, but rigid and inflexible.
   - **RBCNN (Random Binary CNN)**: Randomly initializes binary filters at the start, giving richer pattern diversity.
5. **Weakly Supervised Learning (WSL)**:
   - In video depression datasets, the doctor gave **one single score** for a 30-minute interview. We do *not* have labels telling us: "At minute 3:12 the patient showed depression, but at minute 7:00 they looked completely normal."
   - WSL is an AI strategy that trains on the overall video label, while mathematically forcing the model to discover which specific seconds or minutes actually drove that label.

---

## 3. Paper 1: Explainable Depression Assessment from Face Videos by Weakly Supervised Learning (AAAI 2026)

### Citation & Authors
- **Title**: Explainable Depression Assessment from Face Videos by Weakly Supervised Learning
- **Authors**: Rongfan Liao, Xiangyu Kong, Shiqing Tang, Lang He, Changzeng Fu, Weicheng Xie, Xiaofeng Liu, Lu Liu, Siyang Song
- **Affiliations**: University of Leicester, University of Exeter, Shanghai Univ of Science & Tech, Xi'an Univ of Posts & Telecom, Northeastern Univ, Shenzhen Univ, Hohai Univ
- **Venue**: Proceedings of the 40th AAAI Conference on Artificial Intelligence (AAAI-26), 2026
- **Code**: [https://github.com/liaorongfan/ExpADA](https://github.com/liaorongfan/ExpADA)

---

### Abstract in Simple Words
Most AI systems that detect depression from videos cut the video into short, equal-length chunks (or individual frames) and assume every single chunk is equally depressed. But human beings don't display depressive cues every single second! A person might smile or look neutral for 80% of an interview, and only show depressive cues during a few key moments. Treating all seconds equally confuses the AI and makes it impossible to explain *why* the AI made its diagnosis. 

This paper invents **ExpADA**, an AI system that:
1. Breaks facial video into multiple time horizons (32 frames, 64 frames, 96 frames).
2. Uses a **weakly supervised selection algorithm** that scores each segment's relevance to the overall depression category (none, mild, moderate, severe).
3. Throws away the uninformative or noisy 80% of segments and predicts the final depression score using only the **top 20% most confident, depression-related moments**.
4. Produces a temporal heatmap showing clinicians exactly *which seconds* showed depressive cues.

---

### The Problem & Motivation
1. **Problem 1 (Equal Weighting Fallacy)**: Traditional systems take a video of someone with severe depression, slice it into 1-second clips, and label every single clip as "severe". If the patient smiles politely during second #14, the AI is falsely taught that smiling means severe depression!
2. **Problem 2 (Variable Temporal Duration)**: Facial depressive behaviors don't have fixed lengths. A micro-expression lasts 100 milliseconds; a prolonged vacant stare lasts 4 seconds. Fixed-window models miss this multi-scale nature.
3. **Problem 3 (Black-Box Predictions)**: Doctors cannot trust an AI that gives a score of 32 without pointing to the exact evidence in the video.

---

### Architecture & Step-by-Step Methodology

```
+-------------------------------------------------------------------------------+
|                             EXPADA ARCHITECTURE                               |
+-------------------------------------------------------------------------------+

[ Raw Video V ]
       |
       v  (Sliced into equal 32-frame segments {vn})
[ Video Action Transformer (VAT) ]
       |
       v
[ Segment Feature Units (FUs): fv_n in R^D ]
       |
       +--------------------+--------------------+
       | (Scale 1: 32 f)    | (Scale 2: 64 f)    | (Scale 3: 96 f)
       v                    v                    v
  [ Sliding Win 1 ]    [ Sliding Win 2 ]    [ Sliding Win 3 ]
  [    MLP_1      ]    [    MLP_2      ]    [    MLP_3      ]
       +--------------------+--------------------+
       |
       v
[ Multi-Scale Feature Set F = {F^(1), F^(2), ..., F^(S)} ]
       |
       +------------------------------------+
       |                                    |
       v                                    v
[ Category Branch: P_n,c ]          [ Relevance Branch: R_n,c ]
(MLP -> Softmax over categories)    (MLP -> Softmax relevance scores)
       \                                    /
        \                                  /
         v                                v
     [ Segment Depression Relevance (SDR): R_hat = P (x) R ]
                                  |
                                  v
     [ Mode Voting: Identify Dominant Category I = argmax_c z_c ]
                                  |
                                  v
     [ Top-p% Selection: Pick Top 20% Segments Supporting Category I ]
                                  |
                                  v
                     [ F_sel (Selected Features) ]
                                  |
                                  v
                   [ Prediction Head (FC Layers) ]
                                  |
                                  v
               [ Final Video BDI-II Score = Mean(FC(F_sel)) ]
```

1. **Step 1: Multi-scale Temporal Behaviour Encoder (MTE)**:
   - Slices video into 32-frame segments.
   - Extracts a spatio-temporal feature vector $f_n^v$ from each segment using a 3D **Video Action Transformer (VAT)**.
   - Glides sliding windows of sizes $s \in \{1, 2, 3\}$ (corresponding to 32, 64, and 96 frames) across consecutive feature units and projects them through scale-specific Multi-Layer Perceptrons (MLPs). This produces a pool of multi-scale representations $F$.
2. **Step 2: Discriminative Feature Selection (DFS)**:
   - Every temporal representation $f_n^{(s)}$ is sent into two parallel MLP branches:
     - **Classification Branch ($P_{n,c}$)**: Predicts the probability of 4 depression severity categories ($c \in \{\text{none, mild, moderate, severe}\}$).
     - **Relevance Branch ($R_{n,c}$)**: Estimates how relevant this specific segment is to each category $c$.
   - The element-wise product $\hat{R} = P \otimes R$ generates the **Segment Depression Relevance (SDR)** score $\hat{r}$.
   - The system tallies up evidence to determine the overall video-level category $I$, then picks the **top $p\%$ (experimentally optimal at 20%)** highest-scoring segments supporting category $I$. All other 80% irrelevant or noisy segments are discarded!
3. **Step 3: Two-Stage Training Strategy & Losses**:
   - *Stage 1*: Train the VAT spatio-temporal encoder on video segments using **MSE (Mean Squared Error)** regression + an **Ordinal Classification Loss ($L_{\text{ORD}}$)** that turns the 4 ordered severity levels into 3 progressive binary questions (e.g., "Is it above None?", "Is it above Mild?", "Is it above Moderate?").
   - *Stage 2*: Jointly train MTE and DFS using Weakly Supervised Learning ($L_{\text{WSL}}$), forcing video-level category alignment without frame-level manual labels.
   - Total Loss:
     $$\mathcal{L} = \mathcal{L}_{\text{WSL}} + \mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{BCE}}$$

---

### Experimental Results

Evaluated on official test splits of **AVEC 2013** and **AVEC 2014**:

| Dataset | Metric | Baseline (Meng et al.) | Previous SOTA (Xu et al. 2024) | ExpADA (This Paper) | Improvement / Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AVEC 2013** | **RMSE ↓** | 13.61 | 7.57 | **7.48** | Competitive with top models |
| | **MAE ↓** | 10.88 | 5.95 | **5.77** | Strong error reduction |
| **AVEC 2014** | **RMSE ↓** | 10.86 | 7.18 | **6.89** | **New State-of-the-Art** |
| | **MAE ↓** | 8.86 | 5.86 | **5.11** | **New State-of-the-Art** |

#### Ablation Study (What made it work?):
- **Base Regression (using all segments naively)**: AVEC 2013 RMSE = 9.15, AVEC 2014 Freeform RMSE = 9.26.
- **Adding Ordinal Classification**: AVEC 2013 RMSE dropped to 8.38 (-9.2%), Freeform dropped to 8.10 (-14.3%).
- **Adding Weakly Supervised Feature Selection (Top 20%)**: AVEC 2013 RMSE dropped to **7.48** (-12.0%), Freeform dropped to **7.64**, Northwind dropped to **7.02**.

#### Visual Explainability Findings:
- **t-SNE Visualizations**: When plotting the high-relevance top 20% features, representations belonging to "none", "mild", "moderate", and "severe" form distinct, clean clusters. When plotting the low-relevance bottom 20% features, the classes are completely overlapping and chaotic, proving the bottom 80% is indeed noisy.
- **Temporal Heatmaps**: In long videos, the model generates a timeline heatmap indicating the exact seconds where depressive cues peaked.

---

### Strengths & Limitations
- **Strengths**: Solves the black-box problem by providing exact temporal interpretability; removes noise by filtering out 80% of irrelevant video; achieves SOTA on AVEC 2014.
- **Limitations**: Only evaluated on video (no audio); highlighted segments reflect the AI's internal statistical confidence rather than certified medical biomarker labels; requires 3D VAT extraction which can be computationally heavy during stage 1.

---

## 4. Paper 2: Multi-Time Scale Feature Extraction and Attention Networks for Automatic Depression Level Prediction (Applied Soft Computing 2026)

### Citation & Authors
- **Title**: Multi-Time Scale Feature Extraction and Attention Networks for Automatic Depression Level Prediction
- **Authors**: Sarmad Al-Gawwam, Aleksandr Zaitcev, Mohammad R. Eissa, Noor Jassim, Mohammed Benaissa
- **Affiliations**: University of Sheffield, Leeds Beckett University, University of Surrey, UK
- **Venue**: Applied Soft Computing, Vol. 186, Article 114052, 2026
- **DOI**: [10.1016/j.asoc.2025.114052](https://doi.org/10.1016/j.asoc.2025.114052)

---

### Abstract in Simple Words
Depressive facial expressions show up on vastly different timescales: micro-expressions flicker in **milliseconds**, spoken sentences take **seconds**, and conversational fatigue or posture shifts unfold over **minutes**. Most deep learning models use massive 3D convolutional networks that consume huge amounts of computing power (tens of millions of parameters) and can only look at a couple of seconds at a time.

This paper proposes **MSFE-CTA**, an ultra-lightweight, end-to-end deep learning framework that:
1. Slices full-length interviews into overlapping 3-minute windows so no temporal context is ever lost.
2. Combines deep facial texture (from Inception-ResNet-V2) with behavioral signals (Action Units, head pose, eye gaze).
3. Uses stacked **Inception-TCN blocks with logarithmically increasing dilations** ($d = 1, 2, 4, 8$) to span milliseconds to minutes with virtually zero parameter explosion.
4. Uses a specialized **Channel-Temporal Attention (CTA)** mechanism that skips destructive dimensionality reduction.
5. Uses simple **median aggregation** across 3-minute windows, making it immune to sudden blinks or brief face occlusions.
6. Operates with only **0.85 Million parameters** and **1.85 GFLOPs**, beating models 90 times its size!

---

### The Problem & Motivation
1. **The Computational Bottleneck**: Previous models like MSN (77.7M parameters, 164.9 GFLOPs) or MemRank (62.6M parameters, 40 GFLOPs) require massive enterprise server GPUs. They cannot run on regular clinic computers, laptops, or mobile devices for telehealth.
2. **Short-Window Blindness**: Models that evaluate only 16 or 32 frames at a time miss the big picture. Depressed and healthy people can both smile or look sad for a split second; the true difference lies in how expressions evolve over minutes.
3. **Flaws in Standard Attention (Squeeze-and-Excitation)**: Traditional SE attention squashes channel dimensions down (e.g., $C \rightarrow C/16 \rightarrow C$). The authors show that this dimensionality reduction breaks direct channel-weight relationships needed for subtle facial signals.

---

### Architecture & Step-by-Step Methodology

```
+-------------------------------------------------------------------------------+
|                            MSFE-CTA ARCHITECTURE                              |
+-------------------------------------------------------------------------------+

[ Raw Video Recording (Full-Length Interview) ]
       |
       v (Overlapping 3-Minute Time Windows = 5,400 frames @ 30 fps)
[ Two-Stream Feature Extraction per Frame ]
       +------------------------------------+
       |                                    |
       v (Stream 1: Texture)                v (Stream 2: Behavior)
 [ Inception-ResNet-V2 ]             [ OpenFace Toolkit ]
 (Pre-trained facial appearance)     (17 AUs, 3 Head Pose, 2 Gaze)
       \                                    /
        \                                  /
         v                                v
     [ Concatenated Time-Series: X in R^(F x C) ]
                          |
                          v
+-------------------------------------------------------------+
|        MULTI-TIMESCALE FEATURE EXTRACTION (MSFE)            |
|                                                             |
|  [ Inception-TCN Block 1 ] ---> Millisecond scale dynamics  |
|         | MaxPool/2                                         |
|  [ Inception-TCN Block 2 ] ---> Second scale dynamics       |
|         | MaxPool/2                                         |
|  [ Inception-TCN Block 3 ] ---> Minute scale dynamics        |
|                                                             |
|  * Uses Dilated Conv1D with d = {1, 2, 4, 8} in parallel    |
+-------------------------------------------------------------+
                          |
                          v
+-------------------------------------------------------------+
|             CHANNEL-TEMPORAL ATTENTION (CTA)                |
|                                                             |
|  (A) Channel Attention:                                     |
|      AvgPool & MaxPool -> Dilated Conv1D (d=1,2,4)          |
|      (Avoids Squeeze-and-Excitation dimension bottleneck)   |
|                                                             |
|  (B) Temporal Attention:                                    |
|      Depthwise Separable Convs with kernels k={1, 3, 5}     |
|      (Multi-head temporal recalibration)                    |
+-------------------------------------------------------------+
                          |
                          v
        [ Fully Connected Regression Layer ]
                          |
                          v
   [ Window-Level Predictions: y_1, y_2, ..., y_W ]
                          |
                          v
     [ Final Score = Median(y_1, y_2, ..., y_W) ]
```

1. **Input & Two-Stream Extraction**:
   - Video is divided into 3-minute windows ($T = 5400$ frames at 30 fps).
   - Stream 1: Facial appearance extracted using pre-trained **Inception-ResNet-V2** (output from average pooling layer).
   - Stream 2: High-level behavioral cues extracted using **OpenFace** (Action Unit intensities, head pose sequence, and eye gaze vectors).
   - The two streams are concatenated along channels into a time-series matrix $X \in \mathbb{R}^{F \times C}$.
2. **Multi-Scale Feature Extraction (MSFE)**:
   - A hierarchical stack of **Inception-TCN blocks** separated by MaxPool/2 downsampling.
   - Each Inception block contains parallel 1D dilated convolutions with dilations $d \in \{1, 2, 4, 8\}$.
   - Layer 1 captures **millisecond-scale** twitches.
   - Layer 2 downsamples and captures **second-scale** sentence reactions.
   - Layer 3 downsamples and captures **minute-scale** behavioral shifts.
3. **Channel-Temporal Attention (CTA)**:
   - **Dilated Channel Attention**: Applies Global Average Pooling (GAP) and Max Average Pooling (MAP) across time, passes both through dilated 1D convolutions without reducing channel dimensions, adds them, and applies a Sigmoid gate.
   - **Temporal Attention**: Applies Depthwise Separable Convolutions (DSC) with multiple kernel sizes ($k \in \{1, 3, 5\}$) acting as parallel multi-head temporal attention.
4. **Window Prediction & Median Aggregation**:
   - The attended features pass through an FC layer to output a BDI-II score per 3-minute window.
   - Final video score = **Median of window scores**. The median acts as an outlier filter (if a person rubs their eye or looks away during one window, it does not skew the final prediction).

---

### Experimental Results

Evaluated across **four major benchmarks**: AVEC 2013, AVEC 2014, AVEC 2017 (DAIC-WOZ), and AVEC 2019 (E-DAIC):

| Dataset | Metric | Previous Benchmark / SOTA | MSFE-CTA (This Paper) |
| :--- | :--- | :--- | :--- |
| **AVEC 2013** | **MAE ↓**<br>**RMSE ↓** | 5.43 (DepressionMLP)<br>7.26 (AVA-DepressNet) | **5.75**<br>**6.23** (Lowest RMSE of all visual models!) |
| **AVEC 2014** | **MAE ↓**<br>**RMSE ↓** | 5.41 (PTN)<br>6.98 (VLDN) | **5.72**<br>**6.91** (Beats MDN, MSN, STA-DRN) |
| **AVEC 2017** | **MAE ↓**<br>**RMSE ↓** | 3.97 (DRSN-CW)<br>5.96 (AVA-DepressNet) | **4.85**<br>**5.20** |
| **AVEC 2019** | **MAE ↓**<br>**RMSE ↓** | 4.89 (Ste-Mamba)<br>5.99 (Two-Stage) | **5.30**<br>**6.44** |

#### Efficiency Comparison (The Star Metric!):
Look at the computational cost compared to famous prior architectures:

| Model | Parameters (Millions) ↓ | FLOPs (GigaFLOPs) ↓ | AVEC 2013 RMSE ↓ | AVEC 2014 RMSE ↓ |
| :--- | :--- | :--- | :--- | :--- |
| **MSN (2020)** | 77.70 M | 164.90 G | 7.90 | 7.61 |
| **MemRank (2025)** | 62.60 M | 40.00 G | 7.78 | 7.69 |
| **MDN-152 (2021)** | 52.00 M | 13.40 G | 7.55 | 7.65 |
| **STA (2024)** | 31.27 M | 19.27 G | 7.98 | 7.75 |
| **ResNet-50** | 25.56 M | 3.80 G | 8.25 | 8.23 |
| **DMSN (2024)** | 22.10 M | 11.29 G | 7.66 | 7.50 |
| **MSFE-CTA (Ours)** | **0.85 M** | **1.85 G** | **6.23** | **6.91** |

> **Key Takeaway**: MSFE-CTA achieves superior or matching accuracy with **90x fewer parameters** and **89x fewer FLOPs** than MSN!

#### Ablation Highlights:
- **Dilated Channel Attention vs. Squeeze-and-Excitation (SE)**:
  - Replacing standard SE blocks with Dilated 1D-Conv Channel Attention improved AVEC 2013 MAE/RMSE from 6.75/8.34 down to **5.75/6.23**.
- **Dilated TCN vs Plain 1D Conv in MSFE**:
  - Replacing plain 1D convs with dilated convolutions in the Inception blocks dropped AVEC 2013 RMSE from 8.23 down to **6.23** (a 24% error drop).

---

### Strengths & Limitations
- **Strengths**: Incredibly lightweight (runs on modest consumer hardware, laptops, and mobile NPUs); seamless multi-timescale coverage from milliseconds to minutes; robust against short occlusions due to median window pooling; language-agnostic (evaluated across German AVEC and English DAIC).
- **Limitations**: Visual-only (does not ingest audio speech recordings); evaluated primarily on structured interview settings rather than unconstrained free-living video.

---

## 5. Paper 3: Deep Multi-Modal Network Based Automated Depression Severity Estimation (IEEE TAFFC 2023)

### Citation & Authors
- **Title**: Deep Multi-Modal Network Based Automated Depression Severity Estimation
- **Authors**: Md Azher Uddin, Joolekha Bibi Joolee, Kyung-Ah Sohn
- **Affiliations**: Heriot-Watt University Dubai, Kyung Hee University (Korea), Ajou University (Korea)
- **Venue**: IEEE Transactions on Affective Computing (TAFFC), Vol. 14, No. 3, pp. 2153–2167, July-Sept 2023 (Accepted 2022)
- **DOI**: [10.1109/TAFFC.2022.3179478](https://doi.org/10.1109/TAFFC.2022.3179478)

---

### Abstract in Simple Words
Depression alters not only how someone looks (facial expressions) but also how they speak (vocal acoustic features). Most previous systems looked at only one modality, or fused audio and video simply by gluing feature numbers side-by-side (concatenation). Furthermore, audio models often converted sound into spectrogram images (losing fine-grained temporal patterns), and video models used optical flow (which is too crude for subtle facial texture changes).

This paper develops a unified **Deep Multi-Modal Framework** that:
1. Takes **raw 1D audio waveforms** directly (sampled at 2 kHz) and processes them through an attention-aware 1D ResNet + a BiLSTM Encoder-Decoder.
2. Ingests video using a newly invented descriptor called **Volume Local Directional Structural Pattern (VLDSP)** that captures dynamic facial micro-textures across 3 frames using Kirsch edge masks, paired with Inception-ResNet-V2 appearance features.
3. Summarizes long sequences using **Temporal Attentive Pooling (TAP)**, which preserves chronological order.
4. Fuses audio and video using **Multimodal Factorized Bilinear Pooling (MFB)**, allowing complex, non-linear interactions between voice and face.
5. Achieves top multimodal performance on AVEC 2013, AVEC 2014, and E-DAIC.

---

### The Problem & Motivation
1. **Audio Shortcomings**: Most prior audio ADA papers transformed speech into static spectrogram pictures and applied standard image CNNs. This ignores fine local acoustic transitions in raw speech.
2. **Facial Dynamics Shortcomings**: Optical flow tracks macro-movement (like turning your head), but completely fails to encode subtle structural skin texture changes (like tightening lips or micro-quivers).
3. **Temporal Summarization Blindness**: How do you combine 50 segment features into one video score? Prior papers used simple Max, Mean, or Median pooling, completely discarding the chronological sequence of events.
4. **Crude Multimodal Fusion**: Simply concatenating audio vector $A$ and video vector $V$ ($[A, V]$) assumes the two modalities don't interact. Complex psychiatric cues require bilinear (multiplicative) interaction.

---

### Architecture & Step-by-Step Methodology

```
+-------------------------------------------------------------------------------+
|                      DEEP MULTI-MODAL ARCHITECTURE                            |
+-------------------------------------------------------------------------------+

[ Raw Audio Signal ] (2 kHz, 50 segments)       [ Video Frames ] (7 fps, 60-frame segs)
         |                                                   |
         +-------------------+                               +-------------------+
         |                   |                               |                   |
         v                   v                               v                   v
  [ 1D ResNet ]       [ BiLSTM Enc-Dec ]              [ Inception-ResNet-V2 ] [ VLDSP + 2D ResNet ]
  (5 blocks + Attn)   (3 BiLSTM + Attn +              (Facial Appearance)     (Facial Dynamics via
  (Local features)     3 LSTM Decoder)                                         Kirsch Masks)
         \                   /                               \                   /
          \                 /                                 \                 /
           v               v                                   v               v
    [ Audio Joint Tuning (2 FC) ]                       [ Video Joint Tuning (2 FC) ]
                 |                                                   |
                 v                                                   v
   [ Audio Temporal Attentive Pooling ]               [ Video Temporal Attentive Pooling ]
                (TAP)                                               (TAP)
                 |                                                   |
                 v                                                   v
         Audio Vector a in R^m                               Video Vector v in R^n
                 \                                                   /
                  \                                                 /
                   v                                               v
        +-------------------------------------------------------------------+
        |       MULTIMODAL FACTORIZED BILINEAR POOLING (MFB) FUSION         |
        |                                                                   |
        |   z = SumPooling( (U^T a) (o) (V^T v), k=2 )                      |
        |   Followed by Dropout and L2-Normalization                        |
        +-------------------------------------------------------------------+
                                          |
                                          v
                              [ Fully Connected Layer ]
                                          |
                                          v
                             [ Predicted BDI-II Score ]
```

1. **Audio Spatio-Temporal Network**:
   - Audio is partitioned into 50 segments.
   - *Stream 1 (Local acoustic features)*: 1D Residual Network (5 residual blocks of $1 \times 3$ convs, Batch Normalization, ReLU) + Attention layer.
   - *Stream 2 (Temporal sequence features)*: Attention-aware BiLSTM Encoder-Decoder (3 BiLSTM encoder layers $\rightarrow$ temporal attention $\rightarrow$ 3 LSTM decoder layers).
   - Features from both streams are combined via two Joint Tuning fully connected layers (1024 and 512 units).
2. **Video Spatio-Temporal Network & VLDSP**:
   - Face cropped to $299 \times 299$, sampled at 7 fps, grouped into 60-frame segments.
   - *Stream 1 (Appearance)*: Pre-trained Inception-ResNet-V2.
   - *Stream 2 (Dynamics)*: **Volume Local Directional Structural Pattern (VLDSP)**.
     - Takes 3 consecutive frames ($t-1, t, t+1$).
     - Convolves them with 8 directional Kirsch edge masks ($KM_0$ to $KM_7$).
     - Finds the positions of the top-2 maximum positive edge responses ($P_1, P_2$) and top-2 minimum negative responses.
     - Calculates structural pattern $S$ based on whether $P_2$ is adjacent (edge), perpendicular (corner), or distant (noise).
     - Outputs rich 2D dynamic texture feature maps, which are processed by a 2D Residual Network with Attention.
   - Combined via two Joint Tuning FC layers (2048 and 1024 units) $\rightarrow$ BiLSTM Encoder-Decoder.
3. **Temporal Attentive Pooling (TAP)**:
   - Summarizes the sequence of segment-level features into a single fixed vector $V = \sum_{s=1}^S w_s F_s$.
   - Unlike simple average/median pooling, TAP uses a 1D convolution + 2 FC layers + Softmax + a residual skip connection to learn data-driven weights while **preserving chronological sequence order**.
4. **Multimodal Factorized Bilinear (MFB) Pooling**:
   - Instead of simple concatenation, project audio $a$ and video $v$ into a shared subspace via weights $U$ and $V$, perform element-wise multiplication ($\circ$), and apply sum-pooling with window $k=2$, dropout, and $L_2$ normalization:
     $$z = \text{SumPooling}(U^T a \circ V^T v, k=2)$$
   - Captures rich multi-modal cross-talk before predicting the final BDI-II score.

---

### Experimental Results

Evaluated on **AVEC 2013**, **AVEC 2014**, and **E-DAIC**:

| Modality | Method | AVEC 2013 MAE ↓ | AVEC 2013 RMSE ↓ | AVEC 2014 MAE ↓ | AVEC 2014 RMSE ↓ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Audio Only** | He et al. | 8.20 | 10.01 | 8.19 | 9.99 |
| | Niu et al. | 7.48 | 9.79 | 7.86 | 9.25 |
| | **Ours (Audio)** | **6.70** | **8.16** | **6.95** | **8.46** |
| **Video Only** | Al Jazaery et al. | 7.37 | 9.28 | 7.22 | 9.20 |
| | Melo et al. (MSN) | 5.98 | 7.90 | 5.82 | 7.61 |
| | **Ours (Video)** | **5.90** | **7.32** | **5.75** | **6.98** |
| **Audio + Video**| Meng et al. | 8.72 | 10.96 | — | — |
| | Kaya et al. | 7.68 | 9.44 | 7.69 | 9.61 |
| | Niu et al. (STA) | 6.14 | 8.16 | 5.21 | 7.03 |
| | **Ours (Multimodal MFB)** | **5.38** | **6.83** | **5.03** | **6.16** |

#### Language Independence on English E-DAIC (Audio):
- Audio-only models trained on German AVEC were tested on English E-DAIC:
  - Deep Spectrogram VGG + Transformer: RMSE = 8.72
  - GCNN-LSTM: RMSE = 6.71
  - **Proposed Audio Spatio-Temporal Network**: **RMSE = 5.78** (Superior acoustic feature extraction across languages!)

#### Ablation Highlights:
- **Texture Descriptor**: VLDSP achieved RMSE of 8.54, outperforming Optical Flow (9.89), MHH (10.98), and VLDN (9.08).
- **Pooling Strategy**: Temporal Attentive Pooling (TAP) beat Temporal Max Pooling, Temporal Median Pooling, Fisher Vectors, and Eigen Evolution Pooling.
- **Fusion Method**: MFB fusion achieved MAE 5.38 / RMSE 6.83, beating simple concatenation (5.76 / 7.18) and two FC layers (5.61 / 7.02).

---

### Strengths & Limitations
- **Strengths**: True multimodal integration (facial appearance + facial dynamics + acoustic local features + acoustic temporal sequences); statistical significance validated via Friedman and Nemenyi post-hoc tests ($p < 10^{-9}$); excellent Pearson correlation ($r = 0.905$ on AVEC 2014).
- **Limitations**: Fusion is performed at the very end of the network (late fusion); does not ingest conversational text transcripts; VLDSP requires hand-crafted Kirsch convolution calculations before passing to 2D ResNet.

---

## 6. Paper 4: ERBMA-Net: Enhanced Random Binary Multilevel Attention Network for Facial Depression Recognition

### Citation & Authors
- **Title**: ERBMA-Net: Enhanced Random Binary Multilevel Attention Network for Facial Depression Recognition
- **Authors**: Muhammad Turyalai Khan, Yin Cao, Faisal Shafait, Wu Jun
- **Affiliations**: Guangdong CAS Cogniser Information Technology Co. Ltd. (Guangzhou, China), Changzhou No. 2 People's Hospital of Nanjing Medical University (China), National University of Sciences and Technology (NUST, Islamabad, Pakistan)
- **Code**: [https://github.com/DrTuryalai/ERBMA-Net](https://github.com/DrTuryalai/ERBMA-Net)

---

### Abstract in Simple Words
Standard deep neural networks applied to facial depression detection have two major pitfalls: (1) they are massive and easily overfit on small clinical datasets, and (2) they look at the whole face without paying specialized attention to key emotional hotspots like the **eyes** and **mouth**. 

While Local Binary CNNs (LBCNN) were invented to make networks lightweight by using binary weights ($\pm 1$), their filters are fixed in advance and cannot adapt to messy real-world clinical data.

This paper proposes **ERBMA-Net**, which:
1. Slices the face into three separate regions: **Full Face** (global affect), **Eyes** (brow furrowing, gaze), and **Mouth** (drooping, flattened smile).
2. Uses **Customized EfficientNet** for the face and **Customized ResNet with Coordinate Attention** for eyes and mouth.
3. Invents **ERBCNN** (Enhanced Random Binary CNN), which replaces rigid fixed binary filters with **randomly initialized binary filters** and adds a **refinement layer** ($1 \times 1$ conv) to project features into a rich 512-channel space.
4. Applies **Multilevel Attention**: independent Spatial Attention on each region, Multi-Scale Feature Extraction (MSFE), and **Self-Attention** across the combined regions to learn how the eyes and mouth interact.
5. Evaluates not only on AVEC 2014, but on **CZ2024: a newly collected hospital dataset of 401 real Chinese psychiatric patients**.

---

### The Problem & Motivation
1. **The Overfitting & Heavy Compute Trap**: Deep CNNs like VGG or standard ResNet have 25M to 100M parameters. When trained on clinical datasets with only a few hundred patients, they quickly memorize patient identities instead of learning depressive biomarkers.
2. **The Rigidity of Fixed Binary Filters (LBCNN)**: LBCNN saves memory by freezing binary weights $\{-1, 0, 1\}$, but because the filters are hand-crafted and frozen, they cannot adapt to subtle facial textures.
3. **Regional Disregard**: A flat affect shows globally across the face, but micro-twitches occur around the eye corners (orbicularis oculi) and mouth corners (depressor anguli oris). Treating the face as a single un-cropped square dilutes these signals.
4. **Lack of Real Hospital Clinical Validation**: Almost all academic papers use the same university student challenge datasets (AVEC). Real clinical validation on hospital inpatients/outpatients with psychiatric diagnoses is desperately needed.

---

### Architecture & Step-by-Step Methodology

```
+-------------------------------------------------------------------------------+
|                            ERBMA-NET ARCHITECTURE                             |
+-------------------------------------------------------------------------------+

[ Raw Video Frames (Sampled every 5s) ]
                    |
                    v
[ MTCNN Keypoint Detection & Selective Cropping ]
       +--------------------+--------------------+
       |                    |                    |
       v                    v                    v
  [ Full Face ]          [ Eyes ]            [ Mouth ]
 (224 x 224 x 3)      (96 x 192 x 3)      (96 x 128 x 3)
       |                    |                    |
       v                    v                    v
[ Customized         [ Customized         [ Customized
  EfficientNet ]       ResNet + CAB ]       ResNet + CAB ]
 (FusedMBConv +       (Coordinate Attn     (Coordinate Attn
  Dual SE Attn)        Bottleneck)          Bottleneck)
  64 x 56 x 56        512 x 96 x 192       512 x 96 x 128
       |                    |                    |
       +--------------------+--------------------+
                            |
                            v
       [ ERBCNN (Random Binary Convolutions + Refinement Layer) ]
                            |
                            v
       [ Spatial Attention Mechanism (Independently per Region) ]
                            |
                            v
       [ Multi-Scale Feature Extraction (MSFE: 3x3, 5x5, 7x7) ]
                            |
                            v
       [ Residual Blocks with Identity Skip Connections ]
                            |
                            v
       [ Bilinear Interpolation to uniform 96 x 192 resolution ]
                            |
                            v
       [ Channel Concatenation: 512 x 3 = 1536 Channels ]
                     (1536 x 96 x 192)
                            |
                            v
       [ Self-Attention across Combined Regional Features ]
                            |
                            v
       [ 3-Stage Average Pooling (2x2 -> 4x4 -> 4x4) ]
                            |
                            v
       [ Flattening: Compact 27,648-dimensional Vector ]
                            |
                            v
       [ 3-Layer MLP Regressor (256 -> 64 -> 1) with Dropout ]
                            |
                            v
       [ Predicted BDI-II (AVEC) or HAMD Score (CZ2024) ]
```

1. **Preprocessing & Regional Cropping**:
   - Frames sampled every 5 seconds using OpenCV.
   - **MTCNN** detects face keypoints and crops three regions of interest (ROIs):
     - Face: $224 \times 224 \times 3$
     - Eyes: $96 \times 192 \times 3$ (wide aspect ratio to capture both eyes and inter-ocular furrowing)
     - Mouth: $96 \times 128 \times 3$
2. **Dedicated Backbone Feature Extractors**:
   - **Face (Global)**: Processed by a customized **EfficientNet** using FusedMBConv blocks (standard convolutions instead of depthwise to stabilize on small datasets) and dual Squeeze-and-Excitation (SE) modules. Output: $64 \times 56 \times 56$.
   - **Eyes & Mouth (Local)**: Processed by customized **ResNet** with **Coordinate Attention Bottlenecks (CAB)** that pool along height ($H$) and width ($W$) separately to retain precise spatial coordinates of facial landmarks. Output: $512 \times 96 \times 192$ (eyes) and $512 \times 96 \times 128$ (mouth).
3. **The ERBCNN Module**:
   - Applies **Random Binary Convolutions (RBC)**: 64 filters initialized randomly with weights in $\{-1, 0, 1\}$.
   - Followed by Batch Normalization, LeakyReLU, and a **Refinement Layer** ($1 \times 1$ conv) projecting to 512 channels.
4. **Multilevel Attention & MSFE**:
   - **Spatial Attention**: Applied independently to each of the three feature maps (AvgPool + MaxPool across channels $\rightarrow 7 \times 7$ conv $\rightarrow$ Sigmoid gate).
   - **MSFE**: Multi-Scale Feature Extraction using parallel $3 \times 3$, $5 \times 5$, and $7 \times 7$ filters to capture fine wrinkles, mid-level folds, and macro facial symmetry simultaneously.
   - **Residual Block**: Preserves input gradients via skip connections.
5. **Resampling, Fusion & Self-Attention**:
   - Bilinear interpolation resamples Face ($56 \times 56$) and Mouth ($96 \times 128$) up to $96 \times 192$.
   - The three streams are concatenated along the channel axis ($512 + 512 + 512 = 1536$ channels $\rightarrow 1536 \times 96 \times 192$).
   - **Self-Attention**: Uses Query, Key, and Value projections to calculate attention across all spatial locations, allowing the model to learn how an eye squint correlates with mouth tension across the entire face.
6. **Dimensionality Reduction & Regression**:
   - 3-stage Average Pooling:
     - Stage 1 ($2 \times 2$, stride 2) $\rightarrow 1536 \times 48 \times 96$
     - Stage 2 ($4 \times 4$, stride 4) $\rightarrow 1536 \times 12 \times 24$
     - Stage 3 ($4 \times 4$, stride 4) $\rightarrow 1536 \times 3 \times 6$
   - Flattened into a vector of 27,648 elements (9,216 from face, 9,216 from eyes, 9,216 from mouth).
   - 3-layer MLP Regressor ($256 \rightarrow 64 \rightarrow 1$) with LeakyReLU and 20% Dropout, trained via MSE loss.

---

### Experimental Results

Evaluated on **AVEC 2014** (German, BDI-II) and **CZ2024** (Chinese hospital clinical dataset, HAMD):

| Dataset | Metric | Baseline / Competitor | ERBMA-Net (This Paper) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **AVEC 2014** | **MAE ↓**<br>**RMSE ↓** | 10.03 / 12.56 (Baseline)<br>6.27 / 7.96 (CNN+ViT) | **6.80**<br>**8.18** | Fast inference: 0.85 ms per video on Tesla T4 GPU |
| **CZ2024 (Hospital)** | **MAE ↓**<br>**RMSE ↓** | 7.46 / 9.15 (Jiang et al. 2024)<br>6.84 / 8.77 (MCNN 2025) | **6.37**<br>**8.08** | **New State-of-the-Art on Clinical Dataset** |

#### Ablation Study (Proving Every Component):
- **Disabling ERBCNN completely**: Catastrophic failure! RMSE exploded to **18.61** and MAE to **13.10**.
- **Using fixed LBCNN (traditional)**: RMSE was **13.80**, MAE was **12.06**.
- **Using RBCNN (random binary weights without refinement)**: RMSE dropped to **11.14**, MAE to **8.51**.
- **Using ERBCNN (random binary + $1 \times 1$ refinement)**: RMSE dropped to **8.18**, MAE to **6.80**.
- **Importance of Regional Fusion**:
  - Face only: RMSE = 10.43, MAE = 8.68
  - Face + Mouth: RMSE = 9.82, MAE = 7.87
  - Face + Eyes: RMSE = 9.97, MAE = 7.91
  - **Face + Eyes + Mouth (Complete)**: **RMSE = 8.18, MAE = 6.80** (Combines global affect with local micro-cues).
- **Disabling Attention**:
  - Disabling Spatial Attention: RMSE worsened from 8.18 $\rightarrow$ 9.91 (+1.73).
  - Disabling Self-Attention: RMSE worsened from 8.18 $\rightarrow$ 9.62 (+1.44).

---

### Strengths & Limitations
- **Strengths**: Validated on real hospital clinical patients (CZ2024); highly efficient inference (0.85 ms per video); clear anatomical separation of Face, Eyes, and Mouth; proves that random binary filters with refinement outperform rigid fixed binary convolutions.
- **Limitations**: Operates on static frame crops sampled every 5 seconds, ignoring temporal video motion; single modality (visual only, no voice recordings); dataset class imbalance in CZ2024 (clustered around moderate depression) biased predictions away from mild/severe extremes.

---

## 7. Side-by-Side Comparative Matrix of All 4 Papers

To help you quickly compare and contrast how these four papers solve automated depression recognition, here is a master summary table:

| Dimension | Paper 1: ExpADA (AAAI 2026) | Paper 2: MSFE-CTA (Applied Soft Comp. 2026) | Paper 3: Deep Multi-Modal (IEEE TAFFC 2023) | Paper 4: ERBMA-Net |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Input** | Face Video | Full Face Video | Raw Audio (1D) + Face Video | Cropped Face, Eyes, and Mouth Images |
| **Input Modalities** | Visual only | Visual only (Deep texture + OpenFace AUs/Gaze/Pose) | **Multimodal** (Raw Audio waveform + Video frames) | Visual only (Multi-regional crops) |
| **Core Architecture** | Video Action Transformer (VAT) + Multi-Scale MLPs | Inception-TCN with logarithmic dilations | 1D/2D ResNets + BiLSTM Encoder-Decoders | Customized EfficientNet + ResNet-CAB + ERBCNN |
| **Temporal Modeling** | Multi-Scale Sliding Windows (32, 64, 96 frames) | Continuous 3-minute windows (5,400 frames) | 50 audio segments & 60-frame video segments via BiLSTM | Static frames sampled every 5 seconds |
| **Attention Mechanism** | Relevance branch scoring & Top-20% segment selection | Dilated Channel Attention + Multi-Kernel Depthwise Temporal Attention | Temporal Attention in BiLSTM + Attention in ResNets | Spatial Attention (per region) + Self-Attention (cross-region) |
| **Sequence Aggregation** | Average of Top 20% selected segments | **Median** across 3-minute windows | **Temporal Attentive Pooling (TAP)** (order-preserving) | 3-stage Average Pooling + Flattening |
| **Multimodal Fusion** | N/A (Single modality) | N/A (Feature concatenation of texture + AUs) | **Multimodal Factorized Bilinear (MFB)** pooling | N/A (Bilinear spatial interpolation + channel concat) |
| **Model Footprint** | Moderate | **Ultra-lightweight** (0.85M params, 1.85 GFLOPs) | Large (Dual spatio-temporal streams) | Compact backbone (23.3M params, 0.85 ms inference) |
| **Benchmarks Evaluated** | AVEC 2013, AVEC 2014 | AVEC 2013, 2014, 2017, 2019 | AVEC 2013, AVEC 2014, E-DAIC | AVEC 2014, **CZ2024 (Hospital Clinical)** |
| **Top Highlight** | Temporal explainability & heatmap evidence | SOTA accuracy with 90x fewer parameters | SOTA multimodal fusion & raw audio learning | Real clinical hospital dataset validation |

---

## 8. Evolutionary Timeline & Key Takeaways for Undergrad Researchers

As an undergraduate or researcher entering the affective computing and mental health AI domain, reading these four papers in sequence reveals how the entire research field has progressed over the last several years:

```
[ Traditional ADA Era ] ---> [ The Multimodal Era ] ---> [ The Multi-Timescale Era ] ---> [ The Explainable & Clinical Era ]
Handcrafted LPQ/LBP           Paper 3 (IEEE TAFFC 2023)    Paper 2 (App. Soft Comp 2026)   Paper 1 (AAAI 2026) & Paper 4
- Rigid hand features         - Combines Voice + Face      - Spans ms to minutes           - Weakly Supervised explainability
- Naive SVM / SVR             - Order-preserving TAP       - Ultra-lightweight (0.85M)     - Real hospital patient validation
- Misses deep patterns        - Bilinear MFB fusion        - Replaces heavy 3D CNNs        - Discards noisy segments
```

### Which Paper Should You Read First Depending on Your Goal?

1. **If your project involves Speech + Video (Multimodal AI)**:
   - **Study Paper 3 (Deep Multi-Modal Network)**. It teaches you how to handle raw audio waveforms without relying on clumsy spectrogram images, how to use Temporal Attentive Pooling (TAP) to preserve sequence order, and how to use Multimodal Factorized Bilinear (MFB) pooling to make audio and video interact.
2. **If you have limited GPU resources (e.g., a single student laptop or colab free tier)**:
   - **Study Paper 2 (MSFE-CTA)**. It is a masterclass in efficient network design. By using 1D dilated convolutions with logarithmic rates ($d=1, 2, 4, 8$) and skipping the heavy parameter bloat of 3D CNNs, it achieves state-of-the-art results with less than 1 million parameters.
3. **If you care about Clinical Trust, Explainability, or Action Localization**:
   - **Study Paper 1 (ExpADA)**. It addresses the real-world medical reality that depressed individuals do not exhibit symptoms constantly. Its weakly supervised selection algorithm acts as a filter that discards the 80% of normal/noisy frames and provides heatmaps showing doctors *when* depressive behaviors occurred.
4. **If your research is focused on Facial Landmarks, Region Decomposition, or Clinical Datasets**:
   - **Study Paper 4 (ERBMA-Net)**. It provides a great blueprint for dividing a face into anatomically meaningful regions (eyes, mouth, face), demonstrates that random binary filters outperform frozen binary filters, and provides rare insights into training AI on real hospital patients.

---

### Common Pitfalls to Avoid in Future Research
Across all four papers, the authors repeatedly emphasize several critical lessons:
- **Never treat all video frames equally**: Real depression data is noisy. Filtering uninformative segments or using median aggregation is essential.
- **Do not rely solely on facial appearance**: Micro-dynamics (how muscles twitch or relax over time) carry more diagnostic power than static skin appearance.
- **Beware of dataset imbalance**: In clinical datasets, mild or moderate cases often outnumber severe cases. Always inspect your score distributions and use metrics like MAE and RMSE in tandem.
- **A larger model is not always better**: Paper 2 proves that a well-designed 0.85M parameter network can outperform a 77M parameter 3D-CNN.
