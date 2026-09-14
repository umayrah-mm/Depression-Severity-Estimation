"""
evaluate_missing_modality_composite.py

Missing-modality stress test for the composite model - the analogue of the
baseline's evaluate_missing_modality_robustness.py, EXTENDED with
sub-branch scenarios that only the composite visual doctor supports
(static-only / temporal-only visual).

Every scenario runs the full 5-fold Testing ensemble with forced-missing
doctors / members: the corresponding doctor-mask entries are zeroed and the
raw features are zeroed, mirroring the baseline's convention. The model's
hard-floor suppressor (log_var = 8.0) then mutes the missing doctors.

Run AFTER train_cv_safe_composite.py:
    python evaluate_missing_modality_composite.py
"""

import numpy as np
import torch

import config
from composite_training_utils import metric_vector
from dataset_composite import dataset_to_arrays, load_split_dataset
from test_composite_model import ensemble_predict, load_fold_ensemble

# Reference Testing MAEs of the untouched baseline evidence model for the
# scenarios it also reported (docs/EVIDENCE_FUSION_ARCHITECTURE_EXPLAINED.md):
BASELINE_REFS = {
    "all present": 7.29,
    "missing rppg": 7.26,
    "missing clip": 8.47,
    "missing visual+clip": 10.59,
}

SCENARIOS = [
    # (name, drop_doctors, drop_static, drop_temporal)
    ("all present",             (),        False, False),
    ("missing rppg",            (2,),      False, False),
    ("missing smile",           (3,),      False, False),
    ("missing clip",            (1,),      False, False),
    ("missing visual doctor",   (0,),      False, False),
    ("visual static-only",      (),        False, True),
    ("visual temporal-only",    (),        True,  False),
    ("visual both members out", (),        True,  True),
    ("missing visual+clip",     (0, 1),    False, False),
    ("missing visual+rppg",     (0, 2),    False, False),
    ("missing clip+rppg",       (1, 2),    False, False),
    ("only visual",             (1, 2, 3), False, False),
    ("only clip",               (0, 2, 3), False, False),
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arrays = dataset_to_arrays(load_split_dataset("Testing"))
    print(f"Testing split: {len(arrays['y'])} samples   device: {device}")

    ensemble = load_fold_ensemble(arrays, device)

    header = (f"{'scenario':24s} {'MAE':>7s} {'RMSE':>7s} {'PCC':>6s} {'CCC':>6s}"
              f"  baseline_ref")
    lines = [header, "-" * len(header)]
    print("\n" + header)
    print("-" * len(header))

    for name, drop_doctors, drop_static, drop_temporal in SCENARIOS:
        mean_pred, _ = ensemble_predict(
            arrays, device,
            drop_doctors=drop_doctors,
            drop_static=drop_static,
            drop_temporal=drop_temporal,
            ensemble=ensemble,
        )
        m = metric_vector(arrays["y"], mean_pred)
        ref = BASELINE_REFS.get(name)
        ref_str = f"{ref:7.2f}" if ref is not None else "      -"
        row = (f"{name:24s} {m['mae']:7.3f} {m['rmse']:7.3f} "
               f"{m['pcc']:6.3f} {m['ccc']:6.3f}  {ref_str}")
        print(row)
        lines.append(row)

    note = ("\nbaseline_ref = untouched multimodal_fusion evidence model Testing MAE "
            "for the same scenario (docs/EVIDENCE_FUSION_ARCHITECTURE_EXPLAINED.md)."
            "\n'visual static-only' / 'visual temporal-only' / 'visual both members "
            "out' are composite-only scenarios (the baseline had a single visual "
            "doctor).")
    print(note)
    lines.append(note)

    out_path = config.COMPOSITE_BRANCH_DIR / "missing_modality_results.txt"
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
