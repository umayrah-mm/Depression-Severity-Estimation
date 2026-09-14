"""
test_composite_model.py

ONE-TIME evaluation of the trained 5-fold composite ensemble on the Testing
split - the baseline evidence pipeline's discipline: Testing is touched
here, and only here.

Loads per fold:
    composite_models/composite_fold_{f}.pt                 (best state dict)
    composite_models/composite_fold_{f}_preprocessor.pkl   (fold-local
        TemporalPreprocessor; the SAME one that fold was trained with, so
        each fold's Testing temporal features are preprocessed exactly as
        that fold's model expects)

Outputs:
    composite_models/test_predictions.csv
    composite_models/test_results.txt

Run AFTER train_cv_safe_composite.py has finished:
    python test_composite_model.py
"""

import csv

import numpy as np
import torch

import config
from composite_training_utils import TemporalPreprocessor, metric_vector
from dataset_composite import dataset_to_arrays, load_split_dataset
from models_composite import EvidenceFusionModelComposite

# Untouched multimodal_fusion controls (Testing MAE):
BASELINES = {
    "evidence fusion (control)": 7.29,
    "self-attention fusion": 7.3274,
    "MoE fusion": 7.73,
    "single-MLP baseline": 8.15,
}

BDI_BANDS = [
    ("minimal   (0-13)", 0, 13),
    ("mild     (14-19)", 14, 19),
    ("moderate (20-28)", 20, 28),
    ("severe   (29-63)", 29, 63),
]


def load_fold_ensemble(arrays, device, n_folds=5):
    """[(model, preprocessor)] with model dims inferred from the aligned arrays."""
    ensemble = []
    for fold in range(1, n_folds + 1):
        ckpt_path = config.COMPOSITE_BRANCH_DIR / f"composite_fold_{fold}.pt"
        prep_path = config.COMPOSITE_BRANCH_DIR / \
            f"composite_fold_{fold}_preprocessor.npz"
        if not ckpt_path.exists() or not prep_path.exists():
            raise FileNotFoundError(
                f"Fold {fold} artifacts missing ({ckpt_path.name} / {prep_path.name}). "
                "Run train_cv_safe_composite.py first."
            )
        model = EvidenceFusionModelComposite(
            static_dim=arrays["static_X"].shape[1],
            clip_dim=arrays["clip_X"].shape[1],
            rppg_dim=arrays["rppg_X"].shape[1],
            smile_dim=arrays["smile_X"].shape[1],
            embed_dim=config.EMBED_DIM,
        )
        model.load_state_dict(torch.load(
            ckpt_path, map_location=device, weights_only=True))
        model.to(device).eval()
        pre = TemporalPreprocessor.load_npz(prep_path)
        ensemble.append((model, pre))
    return ensemble


@torch.no_grad()
def ensemble_predict(arrays, device, n_folds=5, drop_doctors=(),
                     drop_static=False, drop_temporal=False, batch_size=64,
                     ensemble=None):
    """
    Fold-ensemble mean prediction with optional forced-missing modalities.

    drop_doctors  : doctor indices to force missing (0=visual, 1=clip,
                    2=rppg, 3=smile)
    drop_static   : force the static member missing
    drop_temporal : force the temporal member missing

    Returns (mean_pred (N,), per_fold_preds (F, N)).
    """
    if ensemble is None:
        ensemble = load_fold_ensemble(arrays, device, n_folds)
    n = len(arrays["y"])
    rows = np.arange(n)

    smask = arrays["static_valid"].astype(np.float32).copy()
    tm = arrays["frame_valid"].any(axis=1).astype(np.float32).copy()
    if drop_static:
        smask[:] = 0.0
    if drop_temporal:
        tm[:] = 0.0

    doctor_mask = np.ones((n, 4), np.float32)
    for d in drop_doctors:
        doctor_mask[:, d] = 0.0
    if (smask == 0).all() and (tm == 0).all():
        doctor_mask[:, 0] = 0.0  # both members gone -> visual doctor gone

    static = arrays["static_X"] * smask[:, None]
    clip = arrays["clip_X"] * doctor_mask[:, 1:2]
    rppg = arrays["rppg_X"] * doctor_mask[:, 2:3]
    smile = arrays["smile_X"] * doctor_mask[:, 3:4]

    per_fold = []
    for model, pre in ensemble:
        temporal = pre.transform(
            arrays["temporal_X"], arrays["frame_valid"], rows)
        temporal["seq"] = temporal["seq"] * \
            torch.from_numpy(tm).view(-1, 1, 1, 1)
        temporal["seq_mask"] = temporal["seq_mask"] * \
            torch.from_numpy(tm).view(-1, 1)
        temporal["motion"] = temporal["motion"] * \
            torch.from_numpy(tm).view(-1, 1, 1)

        preds = []
        for start in range(0, n, batch_size):
            sl = slice(start, min(start + batch_size, n))
            out = model(
                torch.as_tensor(
                    static[sl], dtype=torch.float32, device=device),
                torch.as_tensor(smask[sl], dtype=torch.float32, device=device),
                {k: v[sl].to(device) for k, v in temporal.items()},
                torch.as_tensor(tm[sl], dtype=torch.float32, device=device),
                torch.as_tensor(clip[sl], dtype=torch.float32, device=device),
                torch.as_tensor(rppg[sl], dtype=torch.float32, device=device),
                torch.as_tensor(smile[sl], dtype=torch.float32, device=device),
                mask=torch.as_tensor(
                    doctor_mask[sl], dtype=torch.float32, device=device),
            )
            preds.append(out["prediction"].squeeze(-1).float().cpu().numpy())
        per_fold.append(np.concatenate(preds))
    per_fold = np.stack(per_fold, axis=0)
    return per_fold.mean(axis=0), per_fold


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arrays = dataset_to_arrays(load_split_dataset("Testing"))
    n = len(arrays["y"])

    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    log("=" * 72)
    log("COMPOSITE VISUAL FUSION - ONE-TIME Testing evaluation (5-fold ensemble)")
    log("=" * 72)
    log(f"Testing samples: {n}   device: {device}")

    mean_pred, per_fold = ensemble_predict(arrays, device)

    overall = metric_vector(arrays["y"], mean_pred)
    log(f"\nEnsemble MAE={overall['mae']:.4f}  RMSE={overall['rmse']:.4f}  "
        f"PCC={overall['pcc']:.4f}  CCC={overall['ccc']:.4f}")

    log("\nPer-fold Testing MAE:")
    for f in range(per_fold.shape[0]):
        m = metric_vector(arrays["y"], per_fold[f])
        log(f"  fold {f + 1}: MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}")

    log("\nReference (untouched multimodal_fusion baselines, Testing MAE):")
    for name, mae in BASELINES.items():
        marker = "   <-- composite beats it" if overall["mae"] < mae else ""
        log(f"  {name:28s}: {mae:.4f}{marker}")

    # ---- range-squash audit (the failure mode that motivated this design) ----
    y = arrays["y"].astype(np.float64)
    log("\nPrediction-spread audit (see WIN FINDINGS.md):")
    log(f"  true std: {float(np.std(y)):.2f}   pred std: {float(np.std(mean_pred)):.2f}")
    resid = mean_pred - y
    if np.std(resid) > 1e-8 and np.std(y) > 1e-8:
        corr = float(np.corrcoef(resid, y)[0, 1])
    else:
        corr = 0.0
    log(f"  residual-vs-label correlation: {corr:+.3f} "
        "(strongly negative = systematic under-prediction of severe cases)")

    log("\nPer-band MAE (BDI-II):")
    for name, lo, hi in BDI_BANDS:
        sel = (y >= lo) & (y <= hi)
        if sel.any():
            band_mae = float(np.abs(mean_pred[sel] - y[sel]).mean())
            log(f"  {name}: n={int(sel.sum()):3d}  MAE={band_mae:7.3f}")
        else:
            log(f"  {name}: n=  0")

    csv_path = config.COMPOSITE_BRANCH_DIR / "test_predictions.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["video_id", "y_true", "y_pred_ensemble"]
                        + [f"y_pred_fold_{f + 1}" for f in range(per_fold.shape[0])])
        for i, vid in enumerate(arrays["video_ids"]):
            writer.writerow([vid, float(y[i]), float(mean_pred[i])]
                            + [float(per_fold[f, i]) for f in range(per_fold.shape[0])])
    log(f"\nSaved predictions: {csv_path}")

    txt_path = config.COMPOSITE_BRANCH_DIR / "test_results.txt"
    with open(txt_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log(f"Saved report: {txt_path}")


if __name__ == "__main__":
    main()
