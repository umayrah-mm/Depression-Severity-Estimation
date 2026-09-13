"""
models_evidence.py

Evidence-based (uncertainty-weighted) multimodal fusion model.

Purpose
-------
Each of the four modality branches (visual, clip, rppg, smile) produces:
    1. a 128-dim embedding
    2. its OWN standalone prediction of the severity score, from just
       that one modality ("aux_prediction")
    3. a predicted uncertainty (log-variance) for that standalone
       prediction

The aux_prediction + log_variance pair is what makes this "evidence
based": in Module 4b we will train each modality to predict how
confident it should be in ITS OWN guess, using a heteroscedastic loss.
That is what gives the log-variance heads a real, meaningful training
signal, instead of them just being free parameters with no direct
supervision.

The four embeddings are then combined for the FINAL prediction using
INVERSE-VARIANCE WEIGHTING: modalities the model is confident about get
more say in the final prediction, modalities it is unsure about (or
that are explicitly flagged as MISSING) get automatically down-weighted.

This file defines the architecture (forward pass) ONLY. No training
logic lives here. Training happens in train_cv_safe_evidence.py, a
later module.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Small building block 1: projects a raw modality feature vector to 128-dim
# ---------------------------------------------------------------------------
class ModalityProjector(nn.Module):
    """
    Takes a raw modality feature vector (e.g. 2304-dim visual features)
    and projects it down to a shared 128-dim embedding space, so that
    all four modalities can be compared/fused on equal footing.

    Input shape:  (batch_size, input_dim)
    Output shape: (batch_size, 128)
    """
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


# ---------------------------------------------------------------------------
# Small building block 2: standalone prediction + confidence, per modality
# ---------------------------------------------------------------------------
class ModalityHead(nn.Module):
    """
    Takes a 128-dim embedding and outputs TWO numbers from ONE small
    linear layer:
        - aux_prediction: "if I only had this modality, what severity
          score would I guess?"
        - log_variance:   "how much do I trust that guess?"

    Why bundle them in one Linear(embed_dim, 2) instead of two separate
    layers? It's slightly cheaper (one matrix instead of two) and it's
    a common, well-tested pattern for this kind of "heteroscedastic"
    (uncertainty-aware) head.

    Why log-variance instead of variance directly? Variance must always
    be positive. If the network directly output variance, we'd have to
    clip it to stop it going negative, which creates messy gradients.
    By predicting the LOG of the variance instead, the network can
    output any real number, and we exponentiate it later to get a
    guaranteed-positive variance.

    Input shape:  (batch_size, 128)
    Output shape: two tensors, each (batch_size, 1)
    """
    def __init__(self, embed_dim=128):
        super().__init__()
        self.net = nn.Linear(embed_dim, 2)

    def forward(self, x):
        out = self.net(x)              # (batch_size, 2)
        aux_prediction = out[:, 0:1]   # (batch_size, 1)
        log_variance = out[:, 1:2]     # (batch_size, 1)
        return aux_prediction, log_variance


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class EvidenceFusionModel(nn.Module):
    """
    Four modality branches -> inverse-variance-weighted fusion -> prediction.

    Modality order is FIXED and must always be:
        0 = visual   (2304-dim, MobileNetV3-Small)
        1 = clip     (512-dim,  frozen CLIP ViT-B-32)
        2 = rppg     (9-dim,    green-channel rPPG)
        3 = smile    (88-dim,   openSMILE eGeMAPSv02)

    This order is used consistently for the `mask` tensor, and for every
    (batch, 4)-shaped output below.
    """

    # Hard safety limits so exp() never explodes or underflows.
    LOG_VAR_MIN = -8.0
    LOG_VAR_MAX = 8.0

    # If a modality is explicitly marked MISSING, we override its
    # predicted log-variance with this large constant, forcing its
    # fusion weight down close to zero regardless of what the
    # uncertainty head guessed. This is a "hard floor" safety net that
    # works correctly even BEFORE the model has been trained on
    # examples of missing modalities. Training later refines the
    # PRESENT-modality confidence estimates; it does not need to learn
    # the missing-modality behavior from scratch, because this override
    # guarantees it.
    MISSING_LOG_VAR = 8.0

    def __init__(
        self,
        visual_dim=2304,
        clip_dim=512,
        rppg_dim=9,
        smile_dim=88,
        embed_dim=128,
        dropout=0.3,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # One projector per modality (input_dim differs, output_dim is shared)
        self.visual_proj = ModalityProjector(visual_dim, embed_dim, dropout)
        self.clip_proj   = ModalityProjector(clip_dim, embed_dim, dropout)
        self.rppg_proj   = ModalityProjector(rppg_dim, embed_dim, dropout)
        self.smile_proj  = ModalityProjector(smile_dim, embed_dim, dropout)

        # One (aux_prediction, log_variance) head per modality
        self.visual_head = ModalityHead(embed_dim)
        self.clip_head   = ModalityHead(embed_dim)
        self.rppg_head   = ModalityHead(embed_dim)
        self.smile_head  = ModalityHead(embed_dim)

        # Final prediction head, applied to the FUSED 128-dim vector.
        # Linear(128->64) -> ReLU -> Linear(64->1)
        self.pred_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, visual, clip, rppg, smile, mask=None):
        """
        Parameters
        ----------
        visual : Tensor, shape (batch_size, 2304)
        clip   : Tensor, shape (batch_size, 512)
        rppg   : Tensor, shape (batch_size, 9)
        smile  : Tensor, shape (batch_size, 88)
        mask   : Tensor or None, shape (batch_size, 4), values in {0, 1}.
                 1 = modality is PRESENT for this sample.
                 0 = modality is MISSING for this sample (its raw
                     features are assumed to already be zero-filled
                     by the data loader).
                 Column order must be [visual, clip, rppg, smile].
                 If None, all modalities are assumed present.

        Returns
        -------
        A dict with keys:
            "prediction"      : (batch_size, 1)   final fused prediction
            "fused"           : (batch_size, 128) fused embedding
            "log_vars"        : (batch_size, 4)   per-modality log-variance
                                 (AFTER the missing-modality override)
            "weights"         : (batch_size, 4)   per-modality fusion weight
                                 (sums to 1.0 per sample)
            "aux_predictions" : (batch_size, 4)   each modality's own
                                 standalone prediction, BEFORE fusion.
                                 IMPORTANT: for modalities marked missing
                                 in `mask`, these values are computed from
                                 zero-filled input and are NOT meaningful -
                                 the training loss must exclude them using
                                 the same mask (handled in the loss
                                 function, not here).
        """
        batch_size = visual.shape[0]
        device = visual.device

        # Step 1: project each modality to a 128-dim embedding
        emb_visual = self.visual_proj(visual)   # (batch, 128)
        emb_clip   = self.clip_proj(clip)       # (batch, 128)
        emb_rppg   = self.rppg_proj(rppg)       # (batch, 128)
        emb_smile  = self.smile_proj(smile)     # (batch, 128)

        # Step 2: each modality's own standalone prediction + confidence
        pred_visual, lv_visual = self.visual_head(emb_visual)
        pred_clip,   lv_clip   = self.clip_head(emb_clip)
        pred_rppg,   lv_rppg   = self.rppg_head(emb_rppg)
        pred_smile,  lv_smile  = self.smile_head(emb_smile)

        # Stack into (batch, 4) / (batch, 4, 128) tensors.
        # Fixed order: [visual, clip, rppg, smile]
        embeddings = torch.stack([emb_visual, emb_clip, emb_rppg, emb_smile], dim=1)
        # embeddings shape: (batch, 4, 128)

        aux_predictions = torch.cat([pred_visual, pred_clip, pred_rppg, pred_smile], dim=1)
        # aux_predictions shape: (batch, 4)

        log_vars = torch.cat([lv_visual, lv_clip, lv_rppg, lv_smile], dim=1)
        # log_vars shape: (batch, 4)

        # Clamp for numerical safety before any exponentiation.
        log_vars = torch.clamp(log_vars, self.LOG_VAR_MIN, self.LOG_VAR_MAX)

        # Step 3: apply the "hard floor" for explicitly missing modalities.
        if mask is None:
            mask = torch.ones(batch_size, 4, device=device)
        else:
            mask = mask.to(device)

        # Where mask == 0 (missing), force log-variance to the large
        # MISSING_LOG_VAR constant, overriding whatever the network guessed.
        missing_fill = torch.full_like(log_vars, self.MISSING_LOG_VAR)
        log_vars_for_fusion = torch.where(mask.bool(), log_vars, missing_fill)

        # Also zero out the embedding itself for missing modalities, so
        # even the tiny leftover weight it gets cannot inject content
        # into the fused vector.
        mask_expanded = mask.unsqueeze(-1)  # (batch, 4, 1) for broadcasting
        embeddings_for_fusion = embeddings * mask_expanded

        # Step 4: inverse-variance fusion weights.
        # precision = 1 / variance = exp(-log_variance)
        precision = torch.exp(-log_vars_for_fusion)  # (batch, 4)
        weights = precision / precision.sum(dim=1, keepdim=True)  # (batch, 4)

        # Step 5: weighted sum of embeddings -> single fused 128-dim vector
        weights_expanded = weights.unsqueeze(-1)  # (batch, 4, 1)
        fused = (embeddings_for_fusion * weights_expanded).sum(dim=1)  # (batch, 128)

        # Step 6: predict the depression severity score from the fused vector
        prediction = self.pred_head(fused)  # (batch, 1)

        return {
            "prediction": prediction,
            "fused": fused,
            "log_vars": log_vars_for_fusion,
            "weights": weights,
            "aux_predictions": aux_predictions,
        }


# ---------------------------------------------------------------------------
# Minimal smoke test: run this file directly to check nothing crashes and
# all shapes are correct.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running models_evidence.py smoke test...")

    batch_size = 5
    model = EvidenceFusionModel()
    model.eval()

    visual = torch.randn(batch_size, 2304)
    clip   = torch.randn(batch_size, 512)
    rppg   = torch.randn(batch_size, 9)
    smile  = torch.randn(batch_size, 88)

    # Test 1: all modalities present (mask=None)
    with torch.no_grad():
        outputs = model(visual, clip, rppg, smile, mask=None)

    print("\n--- Test 1: all modalities present ---")
    print(f"prediction shape:      {tuple(outputs['prediction'].shape)}  (expected: ({batch_size}, 1))")
    print(f"fused shape:           {tuple(outputs['fused'].shape)}  (expected: ({batch_size}, 128))")
    print(f"log_vars shape:        {tuple(outputs['log_vars'].shape)}  (expected: ({batch_size}, 4))")
    print(f"weights shape:         {tuple(outputs['weights'].shape)}  (expected: ({batch_size}, 4))")
    print(f"aux_predictions shape: {tuple(outputs['aux_predictions'].shape)}  (expected: ({batch_size}, 4))")
    print(f"weights row sums (should all be ~1.0): {outputs['weights'].sum(dim=1)}")

    # Test 2: mark rppg (index 2) as missing for every sample in the batch
    mask = torch.ones(batch_size, 4)
    mask[:, 2] = 0  # rppg missing
    rppg_zeroed = torch.zeros(batch_size, 9)

    with torch.no_grad():
        outputs2 = model(visual, clip, rppg_zeroed, smile, mask=mask)

    print("\n--- Test 2: rppg marked missing ---")
    print(f"rppg fusion weight (should be near 0): {outputs2['weights'][:, 2]}")
    print(f"rppg log-variance (should be {EvidenceFusionModel.MISSING_LOG_VAR}): {outputs2['log_vars'][:, 2]}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal trainable parameters: {num_params}")
    print("(Slightly more than before, since each modality now also has its")
    print(" own tiny standalone-prediction output. Still far smaller than")
    print(" the attention model's 614,213.)")

    print("\nSmoke test finished with no errors. Model builds and runs correctly.")