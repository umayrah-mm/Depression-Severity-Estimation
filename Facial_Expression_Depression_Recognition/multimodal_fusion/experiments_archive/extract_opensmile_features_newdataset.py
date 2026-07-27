"""
extract_opensmile_features_newdataset.py

Adapted from extract_opensmile_features.py for the 2 new videos'
audio files (0001.wav, 0006.wav). Same eGeMAPSv02 Functionals feature
set, same openSMILE configuration as your original pipeline.

Input:
    C:\\Users\\HP\\Desktop\\NewDataset_processed\\audio\\0001.wav
    C:\\Users\\HP\\Desktop\\NewDataset_processed\\audio\\0006.wav

Output:
    C:\\Users\\HP\\Desktop\\NewDataset_processed\\opensmile_features_new.npy    shape (2, 88)
    C:\\Users\\HP\\Desktop\\NewDataset_processed\\opensmile_sample_ids_new.npy  shape (2,)
    C:\\Users\\HP\\Desktop\\NewDataset_processed\\opensmile_features_new.csv    (human-readable)

Run:
    python extract_opensmile_features_newdataset.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import opensmile

AUDIO_DIR = Path(r"C:\Users\HP\Desktop\NewDataset_processed\audio")
OUTPUT_DIR = Path(r"C:\Users\HP\Desktop\NewDataset_processed")

VIDEOS = ["0001", "0006"]


def build_smile_extractor():
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def main():
    smile = build_smile_extractor()
    print(f"openSMILE feature set: eGeMAPSv02 (Functionals)")

    all_features = []
    all_ids = []
    feature_names = None

    for video_id in VIDEOS:
        audio_path = AUDIO_DIR / f"{video_id}.wav"
        print(f"\nProcessing {audio_path}")

        if not audio_path.exists():
            print(f"  MISSING audio file for {video_id}")
            continue

        result = smile.process_file(str(audio_path))
        if feature_names is None:
            feature_names = list(result.columns)

        feature_vector = result.iloc[0].to_numpy(dtype=np.float32)
        print(f"  Feature vector shape: {feature_vector.shape}  (expect (88,))")

        all_features.append(feature_vector)
        all_ids.append(video_id)

    features_array = np.stack(all_features, axis=0)
    ids_array = np.array(all_ids)

    np.save(OUTPUT_DIR / "opensmile_features_new.npy", features_array)
    np.save(OUTPUT_DIR / "opensmile_sample_ids_new.npy", ids_array)

    csv_df = pd.DataFrame(features_array, columns=feature_names)
    csv_df.insert(0, "sample_id", ids_array)
    csv_df.to_csv(OUTPUT_DIR / "opensmile_features_new.csv", index=False)

    print("\n================ SUMMARY ================")
    print(f"Feature array shape: {features_array.shape}  (expect (2, 88))")
    print(f"Sample IDs: {ids_array}")

    nan_count = np.isnan(features_array).sum()
    inf_count = np.isinf(features_array).sum()
    print(f"NaN values: {nan_count}")
    print(f"Inf values: {inf_count}")

    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()