"""
extract_opensmile_features.py

Extracts fixed-length acoustic feature vectors (eGeMAPSv02 functional
features) from every .wav file using openSMILE.

openSMILE does NOT understand depression. It measures raw acoustic
properties of the voice - pitch, loudness, spectral shape, jitter,
shimmer, voice quality, etc. The correlation between these measurements
and depression severity is learned later by our MLP/MoE model, from
labeled training data. openSMILE is a measuring instrument, not a
diagnostic tool.

Input:
    - config.LABELS_PATH        (video, split, BDI-II)
    - config.AUDIO_DIR / <video_id>.wav

Output:
    - config.OPENSMILE_BRANCH_DIR / opensmile_features.npy   shape (N, 88)
    - config.OPENSMILE_BRANCH_DIR / opensmile_sample_ids.npy shape (N,)
    - config.OPENSMILE_BRANCH_DIR / opensmile_features.csv   (human-readable)
"""

import numpy as np
import pandas as pd
import opensmile
from tqdm import tqdm

import config


def build_smile_extractor():
    """
    eGeMAPSv02 = a standard, compact (88-dim) voice-science feature set.
    Functionals = one aggregated vector per whole audio file (not per-frame),
    which is exactly what we want: one row per video.
    """
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def main():
    config.ensure_output_dirs()

    df = pd.read_csv(config.LABELS_PATH)
    print(f"Loaded {len(df)} rows from labels.csv")

    smile = build_smile_extractor()
    print(f"openSMILE feature set: eGeMAPSv02 (Functionals)")

    all_features = []
    all_sample_ids = []
    feature_names = None

    missing_audio = []
    errors = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting openSMILE features"):
        video_id = row["video"]
        audio_path = config.AUDIO_DIR / f"{video_id}.wav"

        if not audio_path.exists():
            missing_audio.append(video_id)
            continue

        try:
            result = smile.process_file(str(audio_path))
            # result is a pandas DataFrame with 1 row, 88 columns
            if feature_names is None:
                feature_names = list(result.columns)

            feature_vector = result.iloc[0].to_numpy(dtype=np.float32)
            all_features.append(feature_vector)
            all_sample_ids.append(video_id)

        except Exception as e:
            errors.append((video_id, str(e)))

    features_array = np.stack(all_features, axis=0)  # shape (N, 88)
    sample_ids_array = np.array(all_sample_ids)

    # ---- Save outputs ----
    np.save(config.OPENSMILE_BRANCH_DIR / "opensmile_features.npy", features_array)
    np.save(config.OPENSMILE_BRANCH_DIR / "opensmile_sample_ids.npy", sample_ids_array)

    # Human-readable CSV version too, with sample_id as first column
    csv_df = pd.DataFrame(features_array, columns=feature_names)
    csv_df.insert(0, "sample_id", sample_ids_array)
    csv_df.to_csv(config.OPENSMILE_BRANCH_DIR / "opensmile_features.csv", index=False)

    # ---- Summary ----
    print("\n================ SUMMARY ================")
    print(f"Successfully extracted : {len(all_sample_ids)}")
    print(f"Missing audio file     : {len(missing_audio)}")
    print(f"Errors                 : {len(errors)}")
    print(f"Feature vector shape   : {features_array.shape}  (samples x features)")

    if missing_audio:
        print(f"\nFirst 10 missing audio video_ids:")
        for vid in missing_audio[:10]:
            print(f"  {vid}")

    if errors:
        print(f"\nFirst 10 errors:")
        for vid, msg in errors[:10]:
            print(f"  {vid}: {msg}")

    # ---- Sanity checks ----
    nan_count = np.isnan(features_array).sum()
    inf_count = np.isinf(features_array).sum()
    print(f"\nNaN values in features : {nan_count}")
    print(f"Inf values in features  : {inf_count}")

    if nan_count > 0 or inf_count > 0:
        print("WARNING: NaN or Inf values detected. This can break training later.")
        print("We will handle this properly in the dataset-alignment module.")

    print(f"\nSaved to: {config.OPENSMILE_BRANCH_DIR}")


if __name__ == "__main__":
    main()