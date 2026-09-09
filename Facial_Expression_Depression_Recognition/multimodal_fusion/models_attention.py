"""
models_attention.py

A small, heavily-regularaized ATTENTION-based fusion architecture, as an
alternative to the weighted-average MoE gate in models.py.

Idea: instead of blending modalities with a simple weighted average,
let a small self-attention layer look at all 4 modality embeddings
TOGETHER and adjust each one based on the others - so modalities that
agree can reinforce each other, and modalities that disagree can be
partly discounted. This is a much smaller, safer version of the
"transformer fusion" idea, kept intentionally tiny and regularized
to avoid the overfitting risk of a full transformer on ~240 samples.

This is a completely SEPARATE file from models.py - your original
working model is untouched.

    visual (2304) --\
    rppg   (9)     --\
    clip   (512)   ---+--> projectors (128 each) --> [4 tokens] --> self-attention --> pool --> prediction head --> score
    smile  (88)    --/
"""

import torch
import torch.nn as nn

import config


MODALITY_INPUT_DIMS = {
    "visual": 2304,
    "rppg": 9,
    "clip": 512,
    "smile": 88,
}

MODALITY_ORDER = ["visual", "rppg", "clip", "smile"]


class ModalityProjector(nn.Module):
    """
    Same idea as in models.py: raw modality features -> shared embedding
    space. Dropout kept a bit HIGHER here (0.4 vs 0.3) since attention
    fusion has more room to overfit than a simple weighted average.

    Input shape:  (batch, input_dim)
    Output shape: (batch, EMBED_DIM)
    """

    def __init__(self, input_dim, embed_dim=config.EMBED_DIM, dropout=0.6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class AttentionFusion(nn.Module):
    """
    Projects each modality, stacks them as 4 "tokens", and runs ONE
    small self-attention layer over them so modalities can adjust each
    other. Then mean-pools the 4 (now attention-adjusted) tokens into a
    single fused representation.

    Input:  dict of raw modality tensors, each (batch, raw_dim)
    Output: fused (batch, EMBED_DIM), attention_weights (batch, num_heads, 4, 4)
            for inspection/explainability
    """

    def __init__(self, embed_dim=config.EMBED_DIM, num_heads=2, dropout=0.4):
        super().__init__()
        self.embed_dim = embed_dim

        self.projectors = nn.ModuleDict({
            name: ModalityProjector(MODALITY_INPUT_DIMS[name], embed_dim, dropout=dropout)
            for name in MODALITY_ORDER
        })

        # A single, small self-attention layer over the 4 modality tokens.
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        projected = [self.projectors[name](batch[name]) for name in MODALITY_ORDER]
        tokens = torch.stack(projected, dim=1)  # (batch, 4, embed_dim)

        attn_out, attn_weights = self.attention(tokens, tokens, tokens)  # self-attention
        tokens = self.norm(tokens + self.dropout(attn_out))  # residual connection + norm

        fused = tokens.mean(dim=1)  # (batch, embed_dim) - average the 4 adjusted tokens
        return fused, attn_weights


class DepressionPredictionModelAttention(nn.Module):
    """
    Full model: AttentionFusion -> small prediction head -> severity score.
    Mirrors DepressionPredictionModel in models.py, but with attention
    fusion instead of the MoE gate.
    """

    def __init__(self, embed_dim=config.EMBED_DIM, task_type=config.TASK_TYPE, num_classes=4, dropout=0.6):
        super().__init__()
        self.task_type = task_type
        self.fusion = AttentionFusion(embed_dim, dropout=dropout)

        output_dim = 1 if task_type == "regression" else num_classes
        self.prediction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, output_dim),
        )

    def forward(self, batch):
        fused, attn_weights = self.fusion(batch)
        output = self.prediction_head(fused)

        if self.task_type == "regression":
            output = output.squeeze(1)

        return output, attn_weights


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Building attention fusion model...")
    model = DepressionPredictionModelAttention()
    model.eval()

    batch_size = 4
    fake_batch = {
        "visual": torch.randn(batch_size, MODALITY_INPUT_DIMS["visual"]),
        "rppg": torch.randn(batch_size, MODALITY_INPUT_DIMS["rppg"]),
        "clip": torch.randn(batch_size, MODALITY_INPUT_DIMS["clip"]),
        "smile": torch.randn(batch_size, MODALITY_INPUT_DIMS["smile"]),
    }

    with torch.no_grad():
        output, attn_weights = model(fake_batch)

    print(f"\nOutput (prediction) shape: {tuple(output.shape)}  (expect ({batch_size},))")
    print(f"Attention weights shape:   {tuple(attn_weights.shape)}  (expect ({batch_size}, 4, 4))")

    n_params = count_trainable_parameters(model)
    print(f"\nTotal trainable parameters: {n_params:,} ({n_params/1e6:.3f} M)")
    print("(For comparison: your MoE model has 547,909 - expect this to be similar,")
    print(" maybe slightly higher due to the attention layer's extra weights.)")

    print("\nIf the output shape and attention weights shape match expectations,")
    print("the attention fusion architecture is correct.")