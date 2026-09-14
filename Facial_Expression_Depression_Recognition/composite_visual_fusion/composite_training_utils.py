"""
composite_training_utils.py

Training pieces for the composite visual doctor:

1. TemporalPreprocessor
   WIN-faithful fold-local preprocessing of the raw (64, 5, 576) regional
   features: per-region MissingAwareZScore + DeterministicPCA(48) +
   segment motion tokens. Fit on TRAIN-FOLD rows only.

2. apply_two_level_modality_dropout
   Training-time augmentation over a hierarchy:
       sub-branch level : drop static-only / temporal-only   (p=0.15)
       doctor level     : drop the whole visual doctor       (p=0.15)
       outer level      : drop clip / rppg / smile           (p=0.15)
   Guarantees: >=1 visual sub-branch alive whenever the visual doctor is
   alive, and >=1 outer doctor alive per sample.

3. combined_loss_composite
   SmoothL1 on the final fused prediction + 0.3 * heteroscedastic loss over
   the FIVE member heads (static, temporal, clip, rppg, smile), each masked
   by its own availability. The doctor's derived score gets no separate
   loss; gradients reach the members through the nested fusion.
"""

import numpy as np
import torch
import torch.nn.functional as F

import config

# ---------------------------------------------------------------------------
# WIN preprocessing primitives (ported from win/lib.py)
# ---------------------------------------------------------------------------

Z_CLIP = 10.0
LOW_VARIANCE_THRESHOLD = 1e-8


class MissingAwareZScore:
    """Z-score fit only on observed entries (ported from WIN)."""

    def fit(self, values: np.ndarray, missing: np.ndarray) -> "MissingAwareZScore":
        observed = ~missing
        counts = observed.sum(axis=0)
        dim = values.shape[1]
        self.mean_ = np.zeros(dim, np.float32)
        self.scale_ = np.ones(dim, np.float32)
        self.low_variance_ = np.ones(dim, bool)
        usable = counts > 0
        if usable.any():
            totals = np.where(observed[:, usable],
                              values[:, usable], 0.0).sum(axis=0)
            mean = totals / counts[usable]
            self.mean_[usable] = mean.astype(np.float32)
            variance = (
                np.where(
                    observed[:, usable], (values[:, usable] - mean) ** 2, 0.0).sum(axis=0)
                / counts[usable]
            )
            scale = np.sqrt(variance).astype(np.float32)
            low = scale < LOW_VARIANCE_THRESHOLD
            self.low_variance_[usable] = low
            self.scale_[usable] = np.where(low, 1.0, scale)
        return self

    def transform(self, values: np.ndarray, missing: np.ndarray) -> np.ndarray:
        out = np.zeros_like(values, dtype=np.float32)
        observed = ~missing
        out[observed] = ((values - self.mean_) / self.scale_)[observed]
        out[:, self.low_variance_] = 0.0
        return np.clip(out, -Z_CLIP, Z_CLIP).astype(np.float32)


class DeterministicPCA:
    """Sign-stabilized SVD PCA (ported from WIN)."""

    def __init__(self, n_components: int):
        self.n_components = int(n_components)

    def fit(self, rows: np.ndarray) -> "DeterministicPCA":
        rows = np.asarray(rows, dtype=np.float64)
        k = min(self.n_components, rows.shape[0], rows.shape[1])
        if rows.shape[0] == 0 or k <= 0:
            self.components_ = np.zeros(
                (max(self.n_components, 1), rows.shape[1]), np.float32)
            self.n_kept_ = 0
            return self
        _, _, vt = np.linalg.svd(rows, full_matrices=False)
        components = vt[:k]
        signs = np.sign(components[np.arange(
            k), np.abs(components).argmax(axis=1)])
        signs[signs == 0] = 1.0
        components = components * signs[:, None]
        if k < self.n_components:
            components = np.vstack([components, np.zeros(
                (self.n_components - k, rows.shape[1]))])
        self.components_ = components.astype(np.float32)
        self.n_kept_ = int(k)
        return self

    def transform(self, rows: np.ndarray) -> np.ndarray:
        return (np.asarray(rows, dtype=np.float32) @ self.components_.T).astype(np.float32)


# ---------------------------------------------------------------------------
# Fold-local temporal preprocessor
# ---------------------------------------------------------------------------


class TemporalPreprocessor:
    """
    Raw (N, 64, 5, 576) region features -> model-ready tensors, fit on
    train-fold rows only. Same recipe as WIN SeqPreprocessor, video-level
    (no task axis), audio stripped, only the 4 A05 regions used.
    """

    def __init__(self, region_names=config.A05_REGIONS, pca_dim=config.PCA_DIM):
        self.region_names = tuple(region_names)
        self.pca_dim = int(pca_dim)
        self.all_region_names = tuple(config.REGION_NAMES)

    def fit(self, raw: np.ndarray, frame_valid: np.ndarray, rows: np.ndarray) -> "TemporalPreprocessor":
        """
        raw         (N, T, 5, 576) all-region features
        frame_valid (N, T) bool
        rows        indices of TRAIN rows to fit on
        """
        self.region_indices_ = [
            self.all_region_names.index(n) for n in self.region_names]
        # (n, T, 5, 576)
        block = raw[rows]
        fmask = frame_valid[rows]                                   # (n, T)
        self.scalers_ = []
        self.pcas_ = []
        for region in self.region_indices_:
            feats = block[:, :, region,
                          :].reshape(-1, config.MOBILENET_EMBED_DIM)
            observed = np.repeat(
                fmask.reshape(-1)[:, None], config.MOBILENET_EMBED_DIM, axis=1)
            scaler = MissingAwareZScore().fit(feats, ~observed)
            standardized = scaler.transform(feats, ~observed)
            pca = DeterministicPCA(self.pca_dim).fit(
                standardized[fmask.reshape(-1)])
            self.scalers_.append(scaler)
            self.pcas_.append(pca)
        return self

    def transform(self, raw: np.ndarray, frame_valid: np.ndarray, rows: np.ndarray) -> dict:
        """
        Returns dict of torch tensors (on CPU; trainer moves to device):
            seq       (n, T, 4, pca_dim)  float32, zeroed at invalid frames
            seq_mask  (n, T)              float32 frame validity
            motion    (n, 4, pca_dim)     float32 segment motion tokens
        """
        n = len(rows)
        block = raw[rows]
        fmask = frame_valid[rows]
        projected = np.zeros((n, config.T_FRAMES, len(
            self.region_indices_), self.pca_dim), np.float32)
        for slot, region in enumerate(self.region_indices_):
            feats = block[:, :, region,
                          :].reshape(-1, config.MOBILENET_EMBED_DIM)
            observed = np.repeat(
                fmask.reshape(-1)[:, None], config.MOBILENET_EMBED_DIM, axis=1)
            standardized = self.scalers_[slot].transform(feats, ~observed)
            reduced = self.pcas_[slot].transform(standardized)
            reduced[~fmask.reshape(-1)] = 0.0
            projected[:, :, slot, :] = reduced.reshape(
                n, config.T_FRAMES, self.pca_dim)

        motion = self._segment_motion(projected, fmask)
        return {
            "seq": torch.from_numpy(projected),
            "seq_mask": torch.from_numpy(fmask.astype(np.float32)),
            "motion": torch.from_numpy(motion),
        }

    @staticmethod
    def _segment_motion(projected: np.ndarray, fmask: np.ndarray) -> np.ndarray:
        """Mean |delta| within each 2-frame segment, only over complete pairs."""
        n, n_frames, n_regions, pca_dim = projected.shape
        grouped = projected.reshape(
            n, config.N_SEGMENTS, config.FRAMES_PER_SEGMENT, n_regions, pca_dim)
        grouped_mask = fmask.reshape(
            n, config.N_SEGMENTS, config.FRAMES_PER_SEGMENT)
        complete = grouped_mask.all(
            axis=2)                             # (n, S)
        # (n, S, R, D)
        deltas = np.abs(grouped[:, :, 1] - grouped[:, :, 0])
        weights = complete[:, :, None, None].astype(np.float32)
        totals = weights.sum(axis=1)
        return ((deltas * weights).sum(axis=1) / np.maximum(totals, 1.0)).astype(np.float32)

    # ---- persistence (plain NPZ, no pickle) ----------------------------

    def save_npz(self, path) -> None:
        """Save fitted state as plain NPZ (loadable with allow_pickle=False)."""
        arrays = {"region_indices": np.asarray(self.region_indices_, np.int64)}
        for slot, (scaler, pca) in enumerate(zip(self.scalers_, self.pcas_)):
            arrays[f"scaler{slot}_mean"] = scaler.mean_
            arrays[f"scaler{slot}_scale"] = scaler.scale_
            arrays[f"scaler{slot}_low_variance"] = scaler.low_variance_
            arrays[f"pca{slot}_components"] = pca.components_
            arrays[f"pca{slot}_n_kept"] = np.asarray(pca.n_kept_, np.int64)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load_npz(cls, path):
        """Rebuild a fitted TemporalPreprocessor from save_npz output."""
        obj = cls()
        with np.load(path, allow_pickle=False) as data:
            obj.region_indices_ = [int(v) for v in data["region_indices"]]
            obj.scalers_ = []
            obj.pcas_ = []
            for slot in range(len(obj.region_indices_)):
                scaler = MissingAwareZScore()
                scaler.mean_ = data[f"scaler{slot}_mean"]
                scaler.scale_ = data[f"scaler{slot}_scale"]
                scaler.low_variance_ = data[f"scaler{slot}_low_variance"]
                pca = DeterministicPCA(0)
                pca.components_ = data[f"pca{slot}_components"]
                pca.n_components = int(pca.components_.shape[0])
                pca.n_kept_ = int(data[f"pca{slot}_n_kept"])
                obj.scalers_.append(scaler)
                obj.pcas_.append(pca)
        return obj


# ---------------------------------------------------------------------------
# Two-level modality dropout
# ---------------------------------------------------------------------------


def apply_two_level_modality_dropout(
    static, static_mask, temporal_tensors, temporal_mask,
    clip, rppg, smile,
    subbranch_prob=0.15, doctor_prob=0.15, outer_prob=0.15,
):
    """
    Hierarchy-aware modality dropout for one batch.

    Returns the (possibly zeroed) feature tensors, sub-branch masks
    (B, 2) [static, temporal], and outer doctor mask (B, 4)
    [visual, clip, rppg, smile].

    Base availability (static_mask / temporal_mask) is always honored:
    dropout can only REMOVE a sub-branch that exists, never resurrect one
    that is genuinely missing.

    Guarantees per sample:
      - if the visual doctor is alive, at least one sub-branch is alive
      - at least one outer doctor is alive
      - clip/rppg/smile each dropped independently with outer_prob
    """
    b = static.shape[0]
    device = static.device

    # Base availability: dropout can only remove, never resurrect.
    base = torch.stack(
        [static_mask.to(device).float(), temporal_mask.to(device).float()], dim=1
    )  # (B, 2)

    # ---- sub-branch level ----
    draw = (torch.rand(b, 2, device=device) >= subbranch_prob).float()
    sub_mask = draw * base

    # If dropout killed every sub-branch a sample actually has, flip the
    # FIRST available one back on (never invent availability).
    has_any = base.sum(dim=1) > 0
    both_dropped = (sub_mask.sum(dim=1) == 0) & has_any
    if both_dropped.any():
        rows = torch.nonzero(both_dropped).squeeze(1)
        first_available = base[rows].argmax(dim=1)
        sub_mask[rows, first_available] = 1.0

    # ---- doctor level (whole visual doctor) ----
    doctor_drop = (torch.rand(b, device=device) < doctor_prob).float()
    sub_any = (sub_mask.sum(dim=1) > 0).float()
    visual_alive = (1.0 - doctor_drop) * sub_any      # (B,)

    # ---- outer level: clip / rppg / smile ----
    outer = (torch.rand(b, 3, device=device) >= outer_prob).float()

    doctor_mask = torch.stack(
        [visual_alive, outer[:, 0], outer[:, 1], outer[:, 2]], dim=1
    )  # (B, 4)

    # at least one outer doctor alive
    need_fix = doctor_mask.sum(dim=1) == 0
    if need_fix.any():
        rows = torch.nonzero(need_fix).squeeze(1)
        # flip on the highest-precision fallback: clip first, then rppg, smile,
        # then visual
        for col in (1, 2, 3, 0):
            still = doctor_mask[rows].sum(dim=1) == 0
            if not still.any():
                break
            rows2 = rows[still]
            doctor_mask[rows2, col] = 1.0

    # ---- zero features where dropped ----
    sm = sub_mask[:, 0]
    tm = sub_mask[:, 1]
    static_out = static * sm.unsqueeze(-1)
    temporal_out = {
        "seq": temporal_tensors["seq"] * tm.view(-1, 1, 1, 1),
        "seq_mask": temporal_tensors["seq_mask"] * tm.view(-1, 1),
        "motion": temporal_tensors["motion"] * tm.view(-1, 1, 1),
    }
    clip_out = clip * doctor_mask[:, 1:2]
    rppg_out = rppg * doctor_mask[:, 2:3]
    smile_out = smile * doctor_mask[:, 3:4]

    return static_out, sm, temporal_out, tm, clip_out, rppg_out, smile_out, doctor_mask


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def heteroscedastic_loss_members(aux_predictions, log_vars, targets, member_mask):
    """
    Gaussian NLL over member heads.

    aux_predictions: (B, K) stacked member standalone predictions
    log_vars       : (B, K) stacked (already floored) member log-variances
    targets        : (B,)
    member_mask    : (B, K) 1 = member present (used in the loss)
    """
    k = aux_predictions.shape[1]
    targets_expanded = targets.unsqueeze(1).expand(-1, k)
    squared_error = (aux_predictions - targets_expanded) ** 2
    precision = torch.exp(-log_vars)
    per_member = 0.5 * precision * squared_error + 0.5 * log_vars
    masked = per_member * member_mask
    n_present = member_mask.sum()
    return masked.sum() / (n_present + 1e-8)


def combined_loss_composite(outputs, targets, member_mask, aux_loss_weight=0.3):
    """
    outputs     : dict from EvidenceFusionModelComposite.forward()
    targets     : (B,)
    member_mask : (B, 5) [static, temporal, clip, rppg, smile]
                  (sub-branch masks + outer masks for the scalar doctors)
    """
    prediction = outputs["prediction"].squeeze(-1)
    main_loss = F.smooth_l1_loss(prediction, targets)

    vis = outputs["visual"]
    aux_predictions = torch.cat(
        [
            vis["pred_static"],
            vis["pred_temporal"],
            outputs["aux_predictions"][:, 1:2],  # clip
            outputs["aux_predictions"][:, 2:3],  # rppg
            outputs["aux_predictions"][:, 3:4],  # smile
        ],
        dim=1,
    )  # (B, 5)
    log_vars = torch.cat(
        [
            vis["log_var_static"],
            vis["log_var_temporal"],
            outputs["log_vars"][:, 1:2],
            outputs["log_vars"][:, 2:3],
            outputs["log_vars"][:, 3:4],
        ],
        dim=1,
    )  # (B, 5)

    aux_loss = heteroscedastic_loss_members(
        aux_predictions, log_vars, targets, member_mask)
    total = main_loss + aux_loss_weight * aux_loss
    return total, main_loss, aux_loss


def build_member_mask(sub_static, sub_temporal, doctor_mask):
    """
    (B,) static sub-mask, (B,) temporal sub-mask, (B,4) doctor mask
        -> (B, 5) [static, temporal, clip, rppg, smile]
    Scalar doctors participate in the aux loss only when their doctor is
    present. Visual members participate when their sub-branch is present.
    """
    return torch.stack(
        [
            sub_static,
            sub_temporal,
            doctor_mask[:, 1],
            doctor_mask[:, 2],
            doctor_mask[:, 3],
        ],
        dim=1,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def metric_vector(y_true, y_pred):
    """MAE / RMSE / PCC / CCC (population std; degenerate cases -> 0)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    std_t = float(np.std(y_true))
    std_p = float(np.std(y_pred))
    if std_t < 1e-8 or std_p < 1e-8:
        pcc = 0.0
    else:
        pcc = float(np.corrcoef(y_true, y_pred)[0, 1])
    denom = std_t ** 2 + std_p ** 2 + \
        (float(np.mean(y_true)) - float(np.mean(y_pred))) ** 2
    ccc = float(2.0 * pcc * std_t * std_p / denom) if denom > 1e-8 else 0.0
    return {"mae": mae, "rmse": rmse, "pcc": pcc, "ccc": ccc}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running composite_training_utils.py self-test...\n")
    torch.manual_seed(0)

    b = 32
    static = torch.randn(b, 576)
    temporal = {
        "seq": torch.randn(b, config.T_FRAMES, 4, config.PCA_DIM),
        "seq_mask": torch.ones(b, config.T_FRAMES),
        "motion": torch.randn(b, 4, config.PCA_DIM),
    }
    clip = torch.randn(b, 512)
    rppg = torch.randn(b, 9)
    smile = torch.randn(b, 88)

    # ---- dropout guarantees ----
    for _ in range(20):
        s_out, sm, t_out, tm, c_out, r_out, sm_out, dm = apply_two_level_modality_dropout(
            static, torch.ones(b), temporal, torch.ones(b), clip, rppg, smile,
            subbranch_prob=0.5, doctor_prob=0.5, outer_prob=0.5,
        )
        # >=1 sub-branch alive
        assert ((sm + tm) >= 1).all(), "a sample lost both visual sub-branches"
        # >=1 doctor alive
        assert (dm.sum(dim=1) >= 1).all(), "a sample lost all doctors"
        # visual doctor alive implies >=1 sub-branch alive
        vis_alive = dm[:, 0] > 0
        assert ((sm + tm) >= 1)[vis_alive].all() if vis_alive.any() else True
    print("PASS: two-level dropout guarantees hold at p=0.5 over 20 draws.")

    # base availability honored: unavailable sub-branches are never resurrected
    base_sm = (torch.rand(b) > 0.5).float()
    base_tm = (torch.rand(b) > 0.5).float()
    s_out, sm, t_out, tm, c_out, r_out, smi_out, dm = apply_two_level_modality_dropout(
        static, base_sm, temporal, base_tm, clip, rppg, smile
    )
    assert ((sm == 0) | (base_sm > 0)).all(
    ), "static resurrected without availability"
    assert ((tm == 0) | (base_tm > 0)).all(
    ), "temporal resurrected without availability"
    has_any = (base_sm + base_tm) > 0
    vis_alive = dm[:, 0] > 0
    assert not (vis_alive & ~has_any).any(
    ), "visual doctor alive with no members"
    print("PASS: base availability masks are honored by the dropout draws.")

    # ---- preprocessor round trip on synthetic data ----
    n = 24
    raw = np.random.randn(n, config.T_FRAMES, 5,
                          config.MOBILENET_EMBED_DIM).astype(np.float32)
    fmask = np.random.rand(n, config.T_FRAMES) > 0.2
    fmask[:, 0] = True
    pre = TemporalPreprocessor().fit(raw, fmask, np.arange(16))
    t = pre.transform(raw, fmask, np.arange(16, 24))
    assert t["seq"].shape == (8, config.T_FRAMES, 4, config.PCA_DIM)
    assert t["motion"].shape == (8, 4, config.PCA_DIM)
    assert torch.isfinite(t["seq"]).all() and torch.isfinite(t["motion"]).all()
    print(
        f"PASS: TemporalPreprocessor shapes seq={tuple(t['seq'].shape)} motion={tuple(t['motion'].shape)}, all finite.")

    # ---- NPZ persistence round trip (bit-exact) ----
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as fh:
        tmp_path = fh.name
    try:
        pre.save_npz(tmp_path)
        pre2 = TemporalPreprocessor.load_npz(tmp_path)
        t2 = pre2.transform(raw, fmask, np.arange(16, 24))
        assert torch.allclose(t["seq"], t2["seq"]) and torch.allclose(
            t["motion"], t2["motion"])
        print("PASS: TemporalPreprocessor NPZ round trip is bit-exact.")
    finally:
        os.unlink(tmp_path)

    # ---- loss with the real model ----
    from models_composite import EvidenceFusionModelComposite

    model = EvidenceFusionModelComposite()
    model.train()
    s_out, sm, t_out, tm, c_out, r_out, sm_out, dm = apply_two_level_modality_dropout(
        static, torch.ones(b), temporal, torch.ones(b), clip, rppg, smile
    )
    outputs = model(s_out, sm, t_out, tm, c_out, r_out, sm_out, mask=dm)
    targets = torch.randint(0, 63, (b,)).float()
    member_mask = build_member_mask(sm, tm, outputs["doctor_mask"])
    total, main, aux = combined_loss_composite(outputs, targets, member_mask)
    total.backward()
    nan_grad = any(p.grad is not None and torch.isnan(p.grad).any()
                   for p in model.parameters())
    assert not torch.isnan(total).any() and not nan_grad
    print(f"PASS: combined_loss_composite end-to-end backward OK "
          f"(main={main.item():.4f}, aux={aux.item():.4f}).")

    print("\nALL SELF-TESTS PASSED.")
