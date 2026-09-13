"""
evaluate_missing_modality_robustness.py

Loads the 5 trained fold checkpoints from train_cv_safe_evidence.py and
evaluates the ensemble on the REAL Testing split under every 0/1/2-modality-
missing scenario (11 total), to see how gracefully the model degrades when
modalities are unavailable at inference time.

This is a READ-ONLY evaluation script - it does not train anything, and it
reuses the exact same held-out Testing split already evaluated (once) in
train_cv_safe_evidence.py, so there is no additional test exposure beyond
what already happened there.

VERIFY BEFORE RUNNING: batch key names must match your dataset.py, same as
train_cv_safe_evidence.py.

Run:
    python evaluate_missing_modality_robustness.py
"""

import itertools
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

import config
from dataset import load_split_dataset
from models_evidence import EvidenceFusionModel

MODALITY_NAMES = ["visual", "clip", "rppg", "smile"]


def concordance_ccc(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)


def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    try:
        pcc = pearsonr(y_true, y_pred)[0]
    except Exception:
        pcc = 0.0
    return {"MAE": mae, "RMSE": rmse, "PCC": pcc, "CCC": concordance_ccc(y_true, y_pred)}


def describe(missing_indices):
    if not missing_indices:
        return "none missing (baseline)"
    return "missing: " + "+".join(MODALITY_NAMES[i] for i in missing_indices)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    fold_paths = [
        config.EVIDENCE_BRANCH_DIR / f"cv_safe_evidence_fold{i + 1}.pt" for i in range(5)
    ]
    for p in fold_paths:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing checkpoint: {p}\n"
                "Run train_cv_safe_evidence.py first - this script only "
                "evaluates already-trained fold models, it does not train anything."
            )

    models = []
    for p in fold_paths:
        m = EvidenceFusionModel().to(device)
        m.load_state_dict(torch.load(p, map_location=device))
        m.eval()
        models.append(m)
    print(f"Loaded {len(models)} fold checkpoints.")

    test_ds = load_split_dataset("Testing")
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"Testing samples: {len(test_ds)}")

    combos = (
        [()]
        + list(itertools.combinations(range(4), 1))
        + list(itertools.combinations(range(4), 2))
    )

    results_lines = []
    header = f"{'Scenario':<30}{'MAE':<10}{'RMSE':<10}{'PCC':<10}{'CCC':<10}"
    print("\n" + header)
    print("-" * len(header))
    results_lines.append(header)
    results_lines.append("-" * len(header))

    baseline_mae = None

    for combo in combos:
        all_fold_preds = []
        y_true_final = None

        for model in models:
            preds, trues = [], []
            with torch.no_grad():
                for batch in test_loader:
                    visual = batch["visual"].clone().to(device)
                    clip = batch["clip"].clone().to(device)
                    rppg = batch["rppg"].clone().to(device)
                    smile = batch["smile"].clone().to(device)
                    batch_size = visual.shape[0]

                    mask = torch.ones(batch_size, 4, device=device)
                    tensors = [visual, clip, rppg, smile]
                    for idx in combo:
                        mask[:, idx] = 0
                        tensors[idx].zero_()

                    outputs = model(tensors[0], tensors[1], tensors[2], tensors[3], mask)
                    preds.extend(outputs["prediction"].squeeze(-1).cpu().numpy().tolist())
                    trues.extend(batch["y"].numpy().tolist())

            all_fold_preds.append(preds)
            y_true_final = trues

        all_fold_preds = np.array(all_fold_preds)
        ensemble_preds = all_fold_preds.mean(axis=0)
        metrics = compute_metrics(y_true_final, ensemble_preds)

        if not combo:
            baseline_mae = metrics["MAE"]

        line = (f"{describe(combo):<30}{metrics['MAE']:<10.4f}{metrics['RMSE']:<10.4f}"
                f"{metrics['PCC']:<10.4f}{metrics['CCC']:<10.4f}")
        print(line)
        results_lines.append(line)

    print(f"\nBaseline (all present) MAE: {baseline_mae:.4f}")
    print("Compare each scenario's MAE above to this baseline to see how much")
    print("performance degrades as modalities go missing.")

    results_path = config.EVIDENCE_BRANCH_DIR / "results_missing_modality_robustness.txt"
    with open(results_path, "w") as f:
        f.write("Missing-modality robustness evaluation (5-fold ensemble, real Testing split)\n\n")
        f.write("\n".join(results_lines))
        f.write(f"\n\nBaseline (all present) MAE: {baseline_mae:.4f}\n")
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()