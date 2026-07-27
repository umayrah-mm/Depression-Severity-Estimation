"""
models_attention_multitask.py

Adds MULTI-TASK LEARNING to the attention fusion architecture: the model
now predicts BOTH the continuous BDI-II score (regression) AND the
severity band (None/Minimal, Mild, Moderate, Severe - classification),
sharing the same fused representation for both.

Why: forcing the model to also correctly classify the coarser severity
band is a form of regularization that's harder to "cheat" via
memorization than a single regression number - it should push the
model toward learning genuinely generalizable patterns.

    visual (2304) --\
    rppg   (9)     --\
    clip   (512)   ---+--> projectors --> attention fusion --> fused (128) --+--> regression head --> score
    smile  (88)    --/                                                       +--> classification head --> band (4-way)

This is a SEPARATE file - models.py and models_attention.py are untouched.
"""

import torch
import torch.nn as nn

import config
from models_attention import AttentionFusion, MODALITY_ORDER, MODALITY_INPUT_DIMS


def score_to_band(score):
    """Converts a continuous BDI-II score into one of 4 severity band indices."""
    if score <= 13:
        return 0  # None/Minimal
    elif score <= 19:
        return 1  # Mild
    elif score <= 28:
        return 2  # Moderate
    else:
        return 3  # Severe


BAND_NAMES = ["None/Minimal", "Mild", "Moderate", "Severe"]


class DepressionPredictionModelMultitask(nn.Module):
    """
    Attention fusion -> TWO heads: regression (score) and classification
    (severity band). Both heads share the same fused representation.
    """

    def __init__(self, embed_dim=config.EMBED_DIM, num_bands=4, dropout=0.6):
        super().__init__()
        self.fusion = AttentionFusion(embed_dim, dropout=dropout)

        self.regression_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

        self.classification_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_bands),
        )

    def forward(self, batch):
        fused, attn_weights = self.fusion(batch)

        score_pred = self.regression_head(fused).squeeze(1)     # (batch,)
        band_logits = self.classification_head(fused)            # (batch, 4)

        return score_pred, band_logits, attn_weights


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Building multi-task attention model...")
    model = DepressionPredictionModelMultitask()
    model.eval()

    batch_size = 4
    fake_batch = {
        "visual": torch.randn(batch_size, MODALITY_INPUT_DIMS["visual"]),
        "rppg": torch.randn(batch_size, MODALITY_INPUT_DIMS["rppg"]),
        "clip": torch.randn(batch_size, MODALITY_INPUT_DIMS["clip"]),
        "smile": torch.randn(batch_size, MODALITY_INPUT_DIMS["smile"]),
    }

    with torch.no_grad():
        score_pred, band_logits, attn_weights = model(fake_batch)

    print(f"\nScore prediction shape: {tuple(score_pred.shape)}  (expect ({batch_size},))")
    print(f"Band logits shape:      {tuple(band_logits.shape)}  (expect ({batch_size}, 4) - 4 severity bands)")

    n_params = count_trainable_parameters(model)
    print(f"\nTotal trainable parameters: {n_params:,} ({n_params/1e6:.3f} M)")
    print("(Expect slightly MORE than the single-task attention model's 449,025,")
    print(" since we added one new classification head - should still be small.)")

    print("\nIf both output shapes match expectations, the multi-task architecture is correct.")