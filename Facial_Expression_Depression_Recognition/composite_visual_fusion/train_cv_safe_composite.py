"""
train_cv_safe_composite.py

Leak-free subject-grouped 5-fold CV trainer for the composite visual fusion
model (evidence panel whose visual doctor = WIN static + temporal members).

Mirrors the discipline of multimodal_fusion/train_cv_safe_evidence.py:

    * concat Training + Development splits
    * GroupKFold(5) grouped by subject id (video_id.split("_")[0])
    * explicit per-fold subject-leakage check (RuntimeError on overlap)
    * early stopping on validation MAE (patience 15) with best-state restore
    * per-fold checkpoints saved for the ensemble evaluation scripts
    * the Testing split is NEVER touched here

Composite-specific pieces:

    * fold-local temporal preprocessing: TemporalPreprocessor (per-region
      z-score + PCA-48 + motion tokens, WIN recipe) is fit on the TRAIN rows
      of each fold only; the SAME fitted preprocessor transforms that fold's
      val rows (and later the Testing rows in the eval scripts)
    * two-level modality dropout (sub-branch / doctor / outer), honoring
      true base availability
    * parameter groups:
        - temporal member + its head : AdamW lr 3e-4, wd 0.05  (WIN recipe)
        - everything else            : AdamW lr 1e-3, wd 1e-2  (evidence recipe)
      gradients clipped at global norm 1.0

Run:
    python train_cv_safe_composite.py
"""

import csv
import random
import time
from copy import deepcopy

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset

import config
from composite_training_utils import (
    TemporalPreprocessor,
    apply_two_level_modality_dropout,
    build_member_mask,
    combined_loss_composite,
    metric_vector,
)
from dataset_composite import dataset_to_arrays, load_split_dataset
from models_composite import EvidenceFusionModelComposite

N_FOLDS = 5
PATIENCE = 15
MIN_DELTA = 1e-4
GRAD_CLIP = 1.0
EVAL_BATCH_SIZE = 64

# Untouched multimodal_fusion controls (Testing MAE), reference only:
BASELINE_EVIDENCE_TEST_MAE = 7.29
BASELINE_ATTENTION_TEST_MAE = 7.3274


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_tensor_dataset(arrays: dict, rows: np.ndarray, temporal: dict) -> TensorDataset:
    """One row per sample; temporal tensors are already row-aligned to `rows`."""
    return TensorDataset(
        torch.as_tensor(arrays["static_X"][rows], dtype=torch.float32),
        torch.as_tensor(arrays["static_valid"][rows], dtype=torch.float32),
        temporal["seq"],
        temporal["seq_mask"],
        temporal["motion"],
        torch.as_tensor(arrays["clip_X"][rows], dtype=torch.float32),
        torch.as_tensor(arrays["rppg_X"][rows], dtype=torch.float32),
        torch.as_tensor(arrays["smile_X"][rows], dtype=torch.float32),
        torch.as_tensor(arrays["y"][rows], dtype=torch.float32),
    )


@torch.no_grad()
def evaluate_fold(model, loader, device):
    """True-availability prediction (no modality dropout, model.eval())."""
    model.eval()
    preds, ys = [], []
    for static, smask, seq, seq_mask, motion, clip, rppg, smile, y in loader:
        static = static.to(device)
        smask = smask.to(device)
        seq, seq_mask, motion = seq.to(
            device), seq_mask.to(device), motion.to(device)
        clip, rppg, smile = clip.to(device), rppg.to(device), smile.to(device)
        tm = (seq_mask.sum(dim=1) > 0).float()
        static = static * smask.unsqueeze(-1)
        out = model(
            static, smask,
            {"seq": seq, "seq_mask": seq_mask, "motion": motion}, tm,
            clip, rppg, smile, mask=None,
        )
        preds.append(out["prediction"].squeeze(-1).float().cpu())
        ys.append(y)
    return torch.cat(preds).numpy(), torch.cat(ys).numpy()


def train_one_fold(fold_idx, arrays, rows_train, rows_val, device, log):
    set_seed(config.RANDOM_SEED + fold_idx)

    # ---- fold-local temporal preprocessing: fit on TRAIN rows only ----
    pre = TemporalPreprocessor().fit(
        arrays["temporal_X"], arrays["frame_valid"], rows_train)
    t_train = pre.transform(
        arrays["temporal_X"], arrays["frame_valid"], rows_train)
    t_val = pre.transform(arrays["temporal_X"],
                          arrays["frame_valid"], rows_val)

    train_loader = DataLoader(
        make_tensor_dataset(arrays, rows_train, t_train),
        batch_size=config.BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        make_tensor_dataset(arrays, rows_val, t_val),
        batch_size=EVAL_BATCH_SIZE, shuffle=False,
    )

    model = EvidenceFusionModelComposite(
        static_dim=arrays["static_X"].shape[1],
        clip_dim=arrays["clip_X"].shape[1],
        rppg_dim=arrays["rppg_X"].shape[1],
        smile_dim=arrays["smile_X"].shape[1],
        embed_dim=config.EMBED_DIM,
    ).to(device)

    # ---- parameter groups: WIN recipe for the temporal member ----
    win_params = (
        list(model.visual.temporal.parameters())
        + list(model.visual.temporal_head.parameters())
    )
    win_param_ids = {id(p) for p in win_params}
    other_params = [p for p in model.parameters() if id(p)
                    not in win_param_ids]
    optimizer = torch.optim.AdamW([
        {"params": other_params, "lr": config.LEARNING_RATE,
         "weight_decay": config.WEIGHT_DECAY},
        {"params": win_params, "lr": config.TEMPORAL_LEARNING_RATE,
         "weight_decay": config.TEMPORAL_WEIGHT_DECAY},
    ])

    best_mae = float("inf")
    best_epoch = -1
    best_state = None
    stale = 0
    started = time.time()

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        main_sum, aux_sum, batches = 0.0, 0.0, 0
        for static, smask, seq, seq_mask, motion, clip, rppg, smile, y in train_loader:
            static, smask = static.to(device), smask.to(device)
            seq, seq_mask, motion = seq.to(
                device), seq_mask.to(device), motion.to(device)
            clip, rppg, smile = clip.to(device), rppg.to(
                device), smile.to(device)
            y = y.to(device)
            temporal = {"seq": seq, "seq_mask": seq_mask, "motion": motion}

            s_out, sm, t_out, tm, c_out, r_out, smi_out, dm = apply_two_level_modality_dropout(
                static, smask, temporal,
                (seq_mask.sum(dim=1) > 0).float(),
                clip, rppg, smile,
                subbranch_prob=config.SUBBRANCH_DROPOUT_PROB,
                doctor_prob=config.DOCTOR_DROPOUT_PROB,
                outer_prob=config.OUTER_DROPOUT_PROB,
            )
            outputs = model(s_out, sm, t_out, tm, c_out,
                            r_out, smi_out, mask=dm)
            member_mask = build_member_mask(sm, tm, outputs["doctor_mask"])
            total, main_loss, aux_loss = combined_loss_composite(
                outputs, y, member_mask, config.AUX_LOSS_WEIGHT
            )

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            main_sum += main_loss.item()
            aux_sum += aux_loss.item()
            batches += 1

        val_preds, val_ys = evaluate_fold(model, val_loader, device)
        val_mae = metric_vector(val_ys, val_preds)["mae"]
        log(f"    epoch {epoch:3d}  train_main={main_sum / batches:8.4f}  "
            f"train_aux={aux_sum / batches:8.4f}  val_mae={val_mae:8.4f}")

        if val_mae < best_mae - MIN_DELTA:
            best_mae = val_mae
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                log(
                    f"    early stop (best epoch {best_epoch}, val MAE {best_mae:.4f})")
                break

    model.load_state_dict(best_state)
    val_preds, val_ys = evaluate_fold(model, val_loader, device)
    metrics = metric_vector(val_ys, val_preds)
    elapsed = time.time() - started

    torch.save(best_state, config.COMPOSITE_BRANCH_DIR /
               f"composite_fold_{fold_idx + 1}.pt")
    pre.save_npz(config.COMPOSITE_BRANCH_DIR /
                 f"composite_fold_{fold_idx + 1}_preprocessor.npz")

    log(f"    fold {fold_idx + 1} done in {elapsed / 60:.1f} min  "
        f"best_epoch={best_epoch}  val MAE={metrics['mae']:.4f}  "
        f"RMSE={metrics['rmse']:.4f}  PCC={metrics['pcc']:.4f}  CCC={metrics['ccc']:.4f}")

    return {
        "fold": fold_idx + 1,
        "best_epoch": best_epoch,
        "metrics": metrics,
        "val_preds": val_preds,
        "rows_val": rows_val,
        "elapsed_s": elapsed,
    }


def main():
    config.ensure_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    def write_results():
        with open(config.COMPOSITE_BRANCH_DIR / "cv_results.txt", "w") as fh:
            fh.write("\n".join(lines) + "\n")

    log("=" * 72)
    log("COMPOSITE VISUAL FUSION - subject-grouped 5-fold CV (Training + Development)")
    log("=" * 72)
    log(f"device      : {device}")
    log(f"seed        : {config.RANDOM_SEED}")
    log(f"epochs      : {config.EPOCHS} (patience {PATIENCE})")
    log(f"batch size  : {config.BATCH_SIZE}")
    log(f"lr          : panel {config.LEARNING_RATE} / temporal {config.TEMPORAL_LEARNING_RATE}")
    log(f"wd          : panel {config.WEIGHT_DECAY} / temporal {config.TEMPORAL_WEIGHT_DECAY}")
    log(f"dropout p   : sub-branch {config.SUBBRANCH_DROPOUT_PROB}, "
        f"doctor {config.DOCTOR_DROPOUT_PROB}, outer {config.OUTER_DROPOUT_PROB}")

    train_arrays = dataset_to_arrays(load_split_dataset("Training"))
    dev_arrays = dataset_to_arrays(load_split_dataset("Development"))
    arrays = {k: np.concatenate([train_arrays[k], dev_arrays[k]], axis=0)
              for k in train_arrays.keys()}

    n = len(arrays["y"])
    subjects = sorted({str(v).split("_")[0] for v in arrays["video_ids"]})
    log(f"samples     : {n} ({len(train_arrays['y'])} Training + "
        f"{len(dev_arrays['y'])} Development), {len(subjects)} subjects")
    log(f"availability: static invalid {int((arrays['static_valid'] == 0).sum())}, "
        f"temporal invalid {int((~arrays['frame_valid'].any(axis=1)).sum())} (kept with mask 0)")

    groups = np.asarray([str(v).split("_")[0] for v in arrays["video_ids"]])
    splitter = GroupKFold(n_splits=N_FOLDS)
    splits = list(splitter.split(np.arange(n), arrays["y"], groups))

    for fold_idx, (rows_train, rows_val) in enumerate(splits):
        overlap = set(groups[rows_train]) & set(groups[rows_val])
        if overlap:
            raise RuntimeError(
                f"subject leakage in fold {fold_idx + 1}: {sorted(overlap)}")
    log("leakage check PASSED (no subject in both train and val of any fold).")
    write_results()

    oof_pred = np.zeros(n, dtype=np.float64)
    oof_fold = np.zeros(n, dtype=np.int64)
    fold_rows = []

    for fold_idx, (rows_train, rows_val) in enumerate(splits):
        log("")
        log(f"----- fold {fold_idx + 1}/{N_FOLDS}: "
            f"train {len(rows_train)} / val {len(rows_val)}  "
            f"val subjects {sorted(set(groups[rows_val]))} -----")
        result = train_one_fold(
            fold_idx, arrays, rows_train, rows_val, device, log)
        fold_rows.append(result)
        oof_pred[rows_val] = result["val_preds"]
        oof_fold[rows_val] = result["fold"]
        write_results()

    oof = metric_vector(arrays["y"], oof_pred)
    log("")
    log("=" * 72)
    log("OUT-OF-FOLD SUMMARY (Training + Development, all folds pooled)")
    log("=" * 72)
    log(f"OOF MAE={oof['mae']:.4f}  RMSE={oof['rmse']:.4f}  "
        f"PCC={oof['pcc']:.4f}  CCC={oof['ccc']:.4f}")
    fold_maes = [r["metrics"]["mae"] for r in fold_rows]
    log(f"per-fold MAE: {[f'{m:.4f}' for m in fold_maes]}  "
        f"(std {float(np.std(fold_maes)):.4f})")
    log("")
    log("Reference (untouched multimodal_fusion, Testing MAE - NOT comparable to OOF,")
    log(f"just the bar to beat): evidence {BASELINE_EVIDENCE_TEST_MAE}, "
        f"attention {BASELINE_ATTENTION_TEST_MAE}")
    log("")
    log("Next step: python test_composite_model.py   (one-time Testing evaluation)")

    csv_path = config.COMPOSITE_BRANCH_DIR / "oof_predictions.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["video_id", "fold", "y_true", "y_pred_oof"])
        for i, vid in enumerate(arrays["video_ids"]):
            writer.writerow([vid, int(oof_fold[i]),
                             float(arrays["y"][i]), float(oof_pred[i])])
    log(f"Saved OOF predictions: {csv_path}")
    write_results()


if __name__ == "__main__":
    main()
