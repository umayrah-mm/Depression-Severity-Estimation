"""
Multi-Modal Fusion with Integrated Self-Attention and Mixture-of-Experts (MoE)

This script implements a unified neural network module that merges multi-head 
self-attention with a modality-adaptive MoE gating network. By combining these 
two strategies sequentially, they perform synergistic operations rather than 
functioning as standalone, competing architectures:

  - Inter-Modality Attention: Modeled after the approach in 'models_attention.py', 
    this phase enables the four distinct projected modality tokens to interact, 
    updating their representations via cross-modal attention mechanisms prior 
    to the routing stage.
  - Context-Aware MoE Routing: Borrowing the core components from 'models.py', 
    the gating framework evaluates the contextualized embeddings to compute sample-specific 
    routing coefficients for each modality expert.

Note: This implementation leaves 'models.py' and 'models_attention.py' untouched. 
It preserves a singular source of truth by importing verified primitives 
(ModalityProjector, ModalityExpert, GateNetwork, MODALITY_INPUT_DIMS, 
and MODALITY_ORDER) directly from 'models.py'.

    visual (2304) --\
    rppg   (9)     --\                                    [gate: 4 weights]
    clip   (512)   ---+--> projectors (128 each) --> self-attention --+--> experts --> weighted sum --> prediction head --> score
    smile  (88)    --/       (4 tokens, B x 4 x 128)   (context-enriched)
"""

import torch
import torch.nn as nn

import config
from models import (
    MODALITY_INPUT_DIMS,
    MODALITY_ORDER,
    ModalityProjector,
    ModalityExpert,
    GateNetwork,
)


class AttentionMoEFusion(nn.Module):
    """
    Sequential modality integration architecture.
    
    Processes inputs by first applying self-attention across modalities to capture 
    cross-modal dynamics, followed by conditional MoE routing to dynamically scale 
    the resulting representations.
    
    Expected Inputs:
        batch (dict): Dictionary mapping modality identifiers ("visual", "rppg", 
                      "clip", "smile") to tensors of shape (batch_size, feature_dim).
                      
    Expected Outputs:
        fused (Tensor): Aggregated embedding tensor of shape (batch_size, EMBED_DIM).
        gate_weights (Tensor): Routing probability distributions of shape (batch_size, 4).
        attn_weights (Tensor): Extracted attention map tensor of shape (batch_size, 4, 4).
    """

    def __init__(self, embed_dim=config.EMBED_DIM, num_heads=2,
                 projector_dropout=0.6, attn_dropout=0.6):
        super().__init__()
        self.embed_dim = embed_dim

        # Step 1: Modality projection initializers (reused from models.py)
        self.projectors = nn.ModuleDict({
            name: ModalityProjector(MODALITY_INPUT_DIMS[name], embed_dim, dropout=projector_dropout)
            for name in MODALITY_ORDER
        })

        # Steps 2-3: Core multi-head self-attention module over modality items
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

        # Steps 4-5: Routing gate and dedicated sub-network expert layers
        self.gate = GateNetwork(embed_dim, num_modalities=len(MODALITY_ORDER))
        self.experts = nn.ModuleDict({
            name: ModalityExpert(embed_dim)
            for name in MODALITY_ORDER
        })

    def forward(self, batch):
        # ---- Phase 1: Modality Projection ----
        projected = [self.projectors[name](batch[name]) for name in MODALITY_ORDER]
        tokens = torch.stack(projected, dim=1)  # Shape: (B, 4, embed_dim)

        # ---- Phase 2: Cross-Modal Self-Attention ----
        attn_out, attn_weights = self.attention(tokens, tokens, tokens)
        tokens = self.attn_norm(tokens + self.attn_dropout(attn_out))  # Shape: (B, 4, embed_dim)

        # Unpack the sequence back into individual modality components 
        # to ensure compatibility with downstream expert layers.
        enriched = {name: tokens[:, i, :] for i, name in enumerate(MODALITY_ORDER)}

        # ---- Phase 3: Gating Network Processing ----
        z_all = torch.cat([enriched[name] for name in MODALITY_ORDER], dim=1)  # Shape: (B, embed_dim * 4)
        gate_weights = self.gate(z_all)  # Shape: (B, 4)

        # ---- Phase 4: Conditional Expert Computation ----
        expert_outputs = {
            name: self.experts[name](enriched[name])
            for name in MODALITY_ORDER
        }

        # ---- Phase 5: Linear Combination / Gated Aggregation ----
        fused = torch.zeros_like(expert_outputs["visual"])
        for i, name in enumerate(MODALITY_ORDER):
            weight = gate_weights[:, i].unsqueeze(1)
            fused = fused + weight * expert_outputs[name]

        return fused, gate_weights, attn_weights


class DepressionPredictionModelAttentionMoE(nn.Module):
    """
    Downstream prediction framework incorporating AttentionMoEFusion.
    
    Passes multi-modal integrated representations through a multi-layer 
    perceptron (MLP) head to predict final severity targets. Complements the 
    interfaces of standard models in 'models.py' and 'models_attention.py'.
    """

    def __init__(self, embed_dim=config.EMBED_DIM, task_type=config.TASK_TYPE,
                 num_classes=4, dropout=0.6):
        super().__init__()
        self.task_type = task_type
        self.fusion = AttentionMoEFusion(
            embed_dim, projector_dropout=dropout, attn_dropout=dropout
        )

        output_dim = 1 if task_type == "regression" else num_classes
        self.prediction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, output_dim),
        )

    def forward(self, batch):
        fused, gate_weights, attn_weights = self.fusion(batch)
        output = self.prediction_head(fused)

        if self.task_type == "regression":
            output = output.squeeze(1)

        return output, gate_weights, attn_weights


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Instantiating combined attention and MoE architecture...")
    model = DepressionPredictionModelAttentionMoE()
    model.eval()

    batch_size = 4
    fake_batch = {
        "visual": torch.randn(batch_size, MODALITY_INPUT_DIMS["visual"]),
        "rppg": torch.randn(batch_size, MODALITY_INPUT_DIMS["rppg"]),
        "clip": torch.randn(batch_size, MODALITY_INPUT_DIMS["clip"]),
        "smile": torch.randn(batch_size, MODALITY_INPUT_DIMS["smile"]),
    }

    with torch.no_grad():
        output, gate_weights, attn_weights = model(fake_batch)

    print(f"\nDimensions of input features:")
    for name in MODALITY_ORDER:
        print(f"  {name}: {tuple(fake_batch[name].shape)}")

    print(f"\nOutput tensor dimensions: {tuple(output.shape)}  (Target: ({batch_size},) for continuous regression tasks)")
    print(f"Gate routing weights dimension: {tuple(gate_weights.shape)}  (Target: ({batch_size}, 4))")
    print(f"Attention map tensor dimension: {tuple(attn_weights.shape)}  (Target: ({batch_size}, 4, 4))")

    print(f"\nVerification of gating weights normalization (sums should equal ~1.0):")
    print(gate_weights.sum(dim=1))

    print(f"\nSample routing distributions (initial index):")
    for i, name in enumerate(MODALITY_ORDER):
        print(f"  {name}: {gate_weights[0, i].item():.4f}")

    n_params = count_trainable_parameters(model)
    print(f"\nTotal learnable parameters: {n_params:,}  ({n_params/1e6:.3f} M)")
    print("(Reference parameters: Baseline MoE model contains 547,909 elements.")
    print(" This specific variant appends multi-head attention weights,")
    print(" remaining well below the 1.0M threshold to maintain efficiency.)")

    print("\nIf all tensor structures align with target criteria and routing vectors")
    print("sum up to unity, the joint attention-gated mechanism is verified.")
