"""
models_composite.py

Composite Visual Doctor: replaces the single time-collapsed visual doctor of
the evidence-fusion model with a doctor whose evidence comes from two
members with complementary errors (WIN finding: static whole-face and
temporal regional make different mistakes; blending them beat either alone).

Architecture
------------

  static   (576,)  --> StaticMember:
                         ModalityProjector(576->128) --> MemberHead(128->2)
                         = (y_hat_static, log_var_static)

  temporal seq     --> TemporalMember (WIN regional recipe, audio stripped):
    (B, 64, 4, 48)     AttentionPool over frames per region
    + motion (B,4,48)  Linear(48->64) projections + region/stream embeddings
                       1-layer pre-norm Transformer (d=64, 4 heads)
                       mean-pool over tokens --> Linear(64->128) --> MemberHead
                         = (y_hat_temporal, log_var_temporal)

  NESTED inverse-variance fusion (inside the doctor):
      p_m = exp(-log_var_m)                      m in {static, temporal}
      w_m = p_m / (p_static + p_temporal)        (hard-floor at +8 for missing)
      e_vis     = sum_m w_m * e_m * mask_m       (128-dim doctor embedding)
      y_hat_vis = sum_m w_m * y_hat_m
      log_var_vis = -logaddexp(-s_static, -s_temporal)
                  = log(sigma_static^2 * sigma_temporal^2 / (sum of precisions))
        -> var_vis = 1 / (p_static + p_temporal) <= min(var_s, var_t)
        The doctor is SHARPER than either member when both are confident
        and automatically defers when one is unsure or missing.

  Outer panel (unchanged recipe from models_evidence.py):
      doctors = [composite_visual, clip, rppg, smile]
      outer inverse-variance fusion of 128-dim doctor embeddings
      --> pred head (128->64->1) --> BDI-II

This file defines the forward pass ONLY. Training logic lives in
composite_training_utils.py / train_cv_safe_composite.py.
"""

import torch
import torch.nn as nn

import config

# ---------------------------------------------------------------------------
# Shared blocks (same recipes as multimodal_fusion/models_evidence.py)
# ---------------------------------------------------------------------------


class ModalityProjector(nn.Module):
    """Raw modality vector -> shared 128-dim embedding."""

    def __init__(self, input_dim, output_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MemberHead(nn.Module):
    """
    128-dim embedding -> (score, log_variance) with WIN-style head shaping:
    LayerNorm -> small GELU MLP -> Linear(hidden, 2).
    """

    def __init__(self, embed_dim=128, hidden=32, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        out = self.net(x)
        return out[:, 0:1], out[:, 1:2]


# ---------------------------------------------------------------------------
# Temporal member: WIN RegionalAttentionFusion, audio stripped, video-level
# ---------------------------------------------------------------------------


class AttentionPool(nn.Module):
    """WIN temporal attention pooling over the frame axis."""

    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # seq:  (B, T, R, D)   mask: (B, T) float
        logits = self.score(seq).squeeze(-1)            # (B, T, R)
        keep = mask.unsqueeze(-1) > 0                    # (B, T, 1)
        logits = logits.masked_fill(~keep, float("-inf"))
        # (B, 1, 1) no valid frame
        dead = ~keep.any(dim=1, keepdim=True)
        logits = torch.where(dead.expand_as(
            logits), torch.zeros_like(logits), logits)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)   # (B, T, R, 1)
        # (B, R, D)
        return torch.nan_to_num((seq * weights).sum(dim=1), nan=0.0)


class TemporalMember(nn.Module):
    """
    WIN regional sequence model for a single video.

    Input (already fold-local preprocessed):
        seq        (B, T_FRAMES, n_regions, pca_dim)  appearance tokens
        seq_mask   (B, T_FRAMES) float                frame validity
        motion     (B, n_regions, pca_dim)            segment motion tokens

    Output: 128-dim embedding for the MemberHead.
    """

    def __init__(self, n_regions=4, region_dim=48, d_model=64, n_heads=4,
                 n_layers=1, dim_feedforward=64, dropout=0.5,
                 token_dropout=0.15, embed_dim=128, head_hidden=32):
        super().__init__()
        self.token_dropout = float(token_dropout)
        self.pool = AttentionPool(region_dim)
        self.region_projection = nn.Linear(region_dim, d_model)
        self.motion_projection = nn.Linear(region_dim, d_model)
        self.region_embedding = nn.Parameter(torch.zeros(n_regions, d_model))
        self.stream_embedding = nn.Parameter(
            torch.zeros(2, d_model))  # appearance / motion
        for p in (self.region_embedding, self.stream_embedding):
            nn.init.normal_(p, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

    def forward(self, seq, seq_mask, motion):
        pooled = self.pool(seq, seq_mask)                       # (B, R, D)
        b = pooled.shape[0]
        appearance = self.region_projection(pooled) + \
            self.region_embedding[None, :, :] + self.stream_embedding[0]
        motion_tok = self.motion_projection(motion) + \
            self.region_embedding[None, :, :] + self.stream_embedding[1]

        frames_present = (seq_mask.sum(dim=1) > 0).float()      # (B,)
        region_avail = frames_present.unsqueeze(
            -1).expand(-1, self.region_embedding.shape[0])
        motion_avail = (motion.abs().sum(dim=-1) > 0).float()    # (B, R)

        tokens = torch.cat([appearance, motion_tok],
                           dim=1)      # (B, 2R, d_model)
        availability = torch.cat(
            [region_avail, motion_avail], dim=1)  # (B, 2R)

        if self.training and self.token_dropout > 0:
            draw = torch.rand(availability.shape,
                              device=availability.device) < self.token_dropout
            kept = availability * (~draw).float()
            empty = (kept.sum(dim=1) == 0) & (availability.sum(dim=1) > 0)
            if empty.any():
                first = availability[empty].argmax(dim=1)
                kept[torch.nonzero(empty).squeeze(1), first] = 1.0
            availability = kept
        pad_mask = availability <= 0
        all_pad = pad_mask.all(dim=1)
        if all_pad.any():
            pad_mask = pad_mask.clone()
            pad_mask[all_pad, 0] = False
        encoded = self.encoder(
            tokens * availability.unsqueeze(-1), src_key_padding_mask=pad_mask)
        weights = (~pad_mask).float().unsqueeze(-1)
        pooled_out = (encoded * weights).sum(dim=1) / \
            weights.sum(dim=1).clamp_min(1.0)
        # (B, embed_dim)
        return self.out_proj(pooled_out)


# ---------------------------------------------------------------------------
# The composite visual doctor
# ---------------------------------------------------------------------------


class CompositeVisualDoctor(nn.Module):
    """
    Static + temporal members fused by NESTED inverse-variance weighting.

    forward inputs:
        static_feat  (B, 576)   z-scored static embedding (zeros if invalid)
        static_mask  (B,) float 1 = static member present
        temporal dict with keys:
            seq       (B, T, 4, pca_dim)
            seq_mask  (B, T) float
            motion    (B, 4, pca_dim)
        temporal_mask (B,) float  1 = temporal member present

    Returns dict with doctor embedding, doctor (score, log_var),
    member embeddings / scores / log_vars / weights.
    """

    LOG_VAR_MIN = -8.0
    LOG_VAR_MAX = 8.0
    MISSING_LOG_VAR = 8.0

    def __init__(self, static_dim=576, embed_dim=128,
                 temporal_kwargs=None, static_dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        temporal_kwargs = temporal_kwargs or {}
        self.static_proj = ModalityProjector(
            static_dim, embed_dim, static_dropout)
        self.static_head = MemberHead(embed_dim)
        self.temporal = TemporalMember(embed_dim=embed_dim, **temporal_kwargs)
        self.temporal_head = MemberHead(embed_dim)

    def forward(self, static_feat, static_mask, temporal, temporal_mask):
        device = static_feat.device
        b = static_feat.shape[0]

        # ---- member embeddings ----
        emb_static = self.static_proj(
            static_feat)                       # (B, 128)
        emb_temporal = self.temporal(
            temporal["seq"], temporal["seq_mask"], temporal["motion"]
            # (B, 128)
        )

        # ---- member standalone predictions + log-variances ----
        pred_s, lv_s = self.static_head(emb_static)
        pred_t, lv_t = self.temporal_head(emb_temporal)

        # ---- clamps and hard floors ----
        lv_s = torch.clamp(lv_s, self.LOG_VAR_MIN, self.LOG_VAR_MAX)
        lv_t = torch.clamp(lv_t, self.LOG_VAR_MIN, self.LOG_VAR_MAX)

        mask_s = static_mask.to(
            device).unsqueeze(-1)                     # (B, 1)
        mask_t = temporal_mask.to(
            device).unsqueeze(-1)                   # (B, 1)
        missing_fill = torch.full_like(lv_s, self.MISSING_LOG_VAR)
        lv_s_f = torch.where(mask_s > 0, lv_s, missing_fill)
        lv_t_f = torch.where(mask_t > 0, lv_t, missing_fill)

        # ---- nested inverse-variance fusion ----
        # (B, 1)
        p_s = torch.exp(-lv_s_f)
        p_t = torch.exp(-lv_t_f)
        denom = p_s + p_t
        w_s = p_s / denom
        w_t = p_t / denom

        emb_static_z = emb_static * mask_s
        emb_temporal_z = emb_temporal * mask_t
        emb_vis = w_s * emb_static_z + w_t * \
            emb_temporal_z               # (B, 128)
        pred_vis = w_s * pred_s * mask_s + w_t * \
            pred_t * mask_t          # (B, 1)

        # variance of the precision-weighted mean:
        # var_vis = 1 / (p_s + p_t); in log space:
        # log_var_vis = -logaddexp(-lv_s_f, -lv_t_f)
        # (B, 1)
        lv_vis = -torch.logaddexp(-lv_s_f, -lv_t_f)
        lv_vis = torch.where(
            (mask_s > 0) | (mask_t > 0), lv_vis, missing_fill
        )  # both missing -> doctor missing

        return {
            "emb_vis": emb_vis,
            "pred_vis": pred_vis,
            "log_var_vis": lv_vis,
            "emb_static": emb_static, "emb_temporal": emb_temporal,
            "pred_static": pred_s, "pred_temporal": pred_t,
            "log_var_static": lv_s_f, "log_var_temporal": lv_t_f,
            "w_static": w_s, "w_temporal": w_t,
        }


# ---------------------------------------------------------------------------
# Full model: composite visual doctor + CLIP + rPPG + openSMILE panel
# ---------------------------------------------------------------------------


class EvidenceFusionModelComposite(nn.Module):
    """
    Outer panel identical in spirit to EvidenceFusionModel, but doctor 0 is
    the CompositeVisualDoctor instead of a single projector.

    Doctor order (FIXED):
        0 = composite visual (static + temporal)
        1 = clip
        2 = rppg
        3 = smile
    """

    LOG_VAR_MIN = -8.0
    LOG_VAR_MAX = 8.0
    MISSING_LOG_VAR = 8.0

    def __init__(self, static_dim=576, clip_dim=512, rppg_dim=9, smile_dim=88,
                 embed_dim=128, dropout=0.3, temporal_kwargs=None):
        super().__init__()
        self.embed_dim = embed_dim

        self.visual = CompositeVisualDoctor(
            static_dim, embed_dim, temporal_kwargs=temporal_kwargs
        )
        self.clip_proj = ModalityProjector(clip_dim, embed_dim, dropout)
        self.rppg_proj = ModalityProjector(rppg_dim, embed_dim, dropout)
        self.smile_proj = ModalityProjector(smile_dim, embed_dim, dropout)

        self.clip_head = MemberHead(embed_dim)
        self.rppg_head = MemberHead(embed_dim)
        self.smile_head = MemberHead(embed_dim)

        self.pred_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, static_feat, static_mask, temporal, temporal_mask,
                clip, rppg, smile, mask=None):
        """
        mask: (B, 4) doctor-level availability, order
              [composite_visual, clip, rppg, smile]. If None, all present.
              mask[:, 0] is combined with the sub-branch masks internally:
              the visual doctor is present iff mask[0]==1 AND at least one
              sub-branch is present.
        """
        device = static_feat.device
        b = static_feat.shape[0]

        # ---- doctor 0: composite visual ----
        vis = self.visual(static_feat, static_mask, temporal, temporal_mask)
        emb_vis = vis["emb_vis"]
        lv_vis = vis["log_var_vis"]
        pred_vis = vis["pred_vis"]

        # ---- doctors 1..3 ----
        emb_clip = self.clip_proj(clip)
        emb_rppg = self.rppg_proj(rppg)
        emb_smile = self.smile_proj(smile)

        pred_clip, lv_clip = self.clip_head(emb_clip)
        pred_rppg, lv_rppg = self.rppg_head(emb_rppg)
        pred_smile, lv_smile = self.smile_head(emb_smile)

        lv_clip = torch.clamp(lv_clip, self.LOG_VAR_MIN, self.LOG_VAR_MAX)
        lv_rppg = torch.clamp(lv_rppg, self.LOG_VAR_MIN, self.LOG_VAR_MAX)
        lv_smile = torch.clamp(lv_smile, self.LOG_VAR_MIN, self.LOG_VAR_MAX)

        # ---- outer mask ----
        if mask is None:
            mask = torch.ones(b, 4, device=device)
        else:
            mask = mask.to(device).float()

        # doctor presence = explicit mask AND internal sub-branch availability
        internal_vis = ((static_mask > 0) | (
            temporal_mask > 0)).float()   # (B,)
        doctor_mask = mask.clone()
        doctor_mask[:, 0] = doctor_mask[:, 0] * internal_vis

        missing_fill = torch.full_like(lv_vis, self.MISSING_LOG_VAR)
        lv_vis_f = torch.where(doctor_mask[:, 0:1] > 0, lv_vis, missing_fill)
        lv_clip_f = torch.where(doctor_mask[:, 1:2] > 0, lv_clip, missing_fill)
        lv_rppg_f = torch.where(doctor_mask[:, 2:3] > 0, lv_rppg, missing_fill)
        lv_smile_f = torch.where(
            doctor_mask[:, 3:4] > 0, lv_smile, missing_fill)

        # ---- outer inverse-variance fusion ----
        log_vars = torch.cat(
            [lv_vis_f, lv_clip_f, lv_rppg_f, lv_smile_f], dim=1)  # (B, 4)
        precision = torch.exp(-log_vars)
        weights = precision / precision.sum(dim=1, keepdim=True)

        embeddings = torch.stack(
            [emb_vis, emb_clip, emb_rppg, emb_smile], dim=1
        )  # (B, 4, 128)
        dm = doctor_mask.unsqueeze(-1)
        embeddings_for_fusion = embeddings * dm

        weights_expanded = weights.unsqueeze(-1)
        fused = (embeddings_for_fusion *
                 weights_expanded).sum(dim=1)     # (B, 128)

        prediction = self.pred_head(fused)

        aux_predictions = torch.cat(
            [pred_vis, pred_clip, pred_rppg, pred_smile], dim=1
        )  # (B, 4) doctor-level standalone predictions

        return {
            "prediction": prediction,
            "fused": fused,
            "log_vars": log_vars,
            "weights": weights,
            "aux_predictions": aux_predictions,
            "doctor_mask": doctor_mask,
            "visual": vis,   # nested detail: member preds / log_vars / weights
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running models_composite.py smoke test...")
    torch.manual_seed(0)

    b = 5
    model = EvidenceFusionModelComposite()
    model.eval()

    static = torch.randn(b, 576)
    clip = torch.randn(b, 512)
    rppg = torch.randn(b, 9)
    smile = torch.randn(b, 88)
    temporal = {
        "seq": torch.randn(b, config.T_FRAMES, 4, config.PCA_DIM),
        "seq_mask": torch.ones(b, config.T_FRAMES),
        "motion": torch.randn(b, 4, config.PCA_DIM),
    }
    static_mask = torch.ones(b)
    temporal_mask = torch.ones(b)

    with torch.no_grad():
        out = model(static, static_mask, temporal, temporal_mask,
                    clip, rppg, smile, mask=None)

    print(
        f"prediction shape:      {tuple(out['prediction'].shape)}  (expect ({b}, 1))")
    print(
        f"fused shape:           {tuple(out['fused'].shape)}   (expect ({b}, 128))")
    print(
        f"weights shape:         {tuple(out['weights'].shape)}    (expect ({b}, 4))")
    print(f"outer weight row sums: {out['weights'].sum(dim=1)}  (expect 1.0)")
    print(
        f"member w sums:         {out['visual']['w_static'].squeeze(-1) + out['visual']['w_temporal'].squeeze(-1)}  (expect 1.0)")

    # both visual sub-branches missing -> visual doctor weight ~ 0
    sm = torch.zeros(b)
    tm = torch.zeros(b)
    with torch.no_grad():
        out2 = model(torch.zeros(b, 576), sm,
                     {**temporal, "seq_mask": torch.zeros(b, config.T_FRAMES)},
                     tm, clip, rppg, smile, mask=None)
    print(
        f"\nboth-visual-missing doctor weight: {out2['weights'][:, 0]}  (expect ~0.0003)")
    assert (out2["weights"][:, 0] < 1e-3).all()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal trainable parameters: {n_params:,}")
    print("Smoke test finished with no errors.")
