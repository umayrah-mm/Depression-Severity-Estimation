"""
models.py

Defines the modality-adaptive Mixture-of-Experts (MoE) fusion architecture:

    visual (2304) --\
    rppg   (9)     --\
    clip   (512)   ---+--> projectors (128 each) --> gate + experts --> fusion --> prediction head --> severity score
    smile  (88)    --/

All projector/expert/gate/head parameters are trainable and small.
Nothing here loads or trains CLIP, MobileNet, or openSMILE - those are
already-frozen, already-extracted features (config.EMBED_DIM-sized inputs
below refer to the SHARED embedding space we project into, not the raw
feature sizes).
"""

import torch
import torch.nn as nn

import config


# Raw input dimensionality of each modality, as confirmed by our
# extraction + alignment modules (Module 4 and Module 5).
MODALITY_INPUT_DIMS = {
    "visual": 2304,
    "rppg": 9,
    "clip": 512,
    "smile": 88,
}

MODALITY_ORDER = ["visual", "rppg", "clip", "smile"]  # fixed order used everywhere


class ModalityProjector(nn.Module):
    """
    Small MLP: raw modality features -> shared embedding space (EMBED_DIM).

    Input shape:  (batch, input_dim)
    Output shape: (batch, EMBED_DIM)
    """

    def __init__(self, input_dim, embed_dim=config.EMBED_DIM, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ModalityExpert(nn.Module):
    """
    Small MLP "expert" that further transforms a modality's projected
    embedding before fusion. Kept intentionally small (Green AI).

    Input shape:  (batch, EMBED_DIM)
    Output shape: (batch, EMBED_DIM)
    """

    def __init__(self, embed_dim=config.EMBED_DIM, hidden_dim=None, dropout=0.3):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class GateNetwork(nn.Module):
    """
    Looks at all 4 projected embeddings concatenated together, and outputs
    a weight per modality (summing to 1 via softmax).

    Input shape:  (batch, EMBED_DIM * num_modalities)
    Output shape: (batch, num_modalities)
    """

    def __init__(self, embed_dim=config.EMBED_DIM, num_modalities=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modalities),
        )

    def forward(self, z_all):
        raw_scores = self.net(z_all)          # (batch, num_modalities)
        gate_weights = torch.softmax(raw_scores, dim=1)  # sums to 1 per sample
        return gate_weights


class AdaptiveMoEFusion(nn.Module):
    """
    Combines projectors, experts, and the gate network into one fusion
    module.

    Forward pass:
        1. Project each raw modality into shared embedding space.
        2. Concatenate all projected embeddings -> feed to gate.
        3. Pass each projected embedding through its own expert.
        4. Weighted sum of expert outputs, using gate weights.

    Returns:
        fused:        (batch, EMBED_DIM)      - the combined representation
        gate_weights: (batch, num_modalities)  - for explainability/logging
    """

    def __init__(self, embed_dim=config.EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim

        self.projectors = nn.ModuleDict({
            name: ModalityProjector(MODALITY_INPUT_DIMS[name], embed_dim)
            for name in MODALITY_ORDER
        })
        self.experts = nn.ModuleDict({
            name: ModalityExpert(embed_dim)
            for name in MODALITY_ORDER
        })
        self.gate = GateNetwork(embed_dim, num_modalities=len(MODALITY_ORDER))

    def forward(self, batch):
        """
        batch: dict with keys "visual", "rppg", "clip", "smile", each a
        tensor of shape (batch_size, raw_input_dim_for_that_modality).
        """
        projected = {
            name: self.projectors[name](batch[name])  # (B, embed_dim)
            for name in MODALITY_ORDER
        }

        z_all = torch.cat([projected[name] for name in MODALITY_ORDER], dim=1)  # (B, embed_dim*4)
        gate_weights = self.gate(z_all)  # (B, 4)

        expert_outputs = {
            name: self.experts[name](projected[name])  # (B, embed_dim)
            for name in MODALITY_ORDER
        }

        # Weighted sum: gate_weights[:, i] scales expert_outputs[modality_i]
        fused = torch.zeros_like(expert_outputs["visual"])
        for i, name in enumerate(MODALITY_ORDER):
            weight = gate_weights[:, i].unsqueeze(1)  # (B, 1), broadcasts over embed_dim
            fused = fused + weight * expert_outputs[name]

        return fused, gate_weights


class DepressionPredictionModel(nn.Module):
    """
    Full model: AdaptiveMoEFusion -> small prediction head -> severity score.

    For TASK_TYPE == "regression": output shape (batch, 1), a continuous
    predicted BDI-II score.
    For TASK_TYPE == "classification": output shape (batch, num_classes).
    """

    def __init__(self, embed_dim=config.EMBED_DIM, task_type=config.TASK_TYPE, num_classes=4):
        super().__init__()
        self.task_type = task_type
        self.fusion = AdaptiveMoEFusion(embed_dim)

        output_dim = 1 if task_type == "regression" else num_classes

        self.prediction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, output_dim),
        )

    def forward(self, batch):
        fused, gate_weights = self.fusion(batch)
        output = self.prediction_head(fused)

        if self.task_type == "regression":
            output = output.squeeze(1)  # (B, 1) -> (B,)

        return output, gate_weights


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Self-test: verify the model runs end-to-end on FAKE data with the
    # correct shapes, and report how many trainable parameters it has
    # (should be small - Green AI check).
    print("Building model...")
    model = DepressionPredictionModel()
    model.eval()

    batch_size = 4
    fake_batch = {
        "visual": torch.randn(batch_size, MODALITY_INPUT_DIMS["visual"]),
        "rppg": torch.randn(batch_size, MODALITY_INPUT_DIMS["rppg"]),
        "clip": torch.randn(batch_size, MODALITY_INPUT_DIMS["clip"]),
        "smile": torch.randn(batch_size, MODALITY_INPUT_DIMS["smile"]),
    }

    with torch.no_grad():
        output, gate_weights = model(fake_batch)

    print(f"\nInput shapes:")
    for name in MODALITY_ORDER:
        print(f"  {name}: {tuple(fake_batch[name].shape)}")

    print(f"\nOutput (prediction) shape: {tuple(output.shape)}  (expect ({batch_size},) for regression)")
    print(f"Gate weights shape:        {tuple(gate_weights.shape)}  (expect ({batch_size}, 4))")
    print(f"\nGate weights sum per sample (should all be ~1.0):")
    print(gate_weights.sum(dim=1))

    print(f"\nGate weights for first sample:")
    for i, name in enumerate(MODALITY_ORDER):
        print(f"  {name}: {gate_weights[0, i].item():.4f}")

    n_params = count_trainable_parameters(model)
    print(f"\nTotal trainable parameters: {n_params:,}  ({n_params/1e6:.3f} M)")
    print("\nIf output shape and gate weights shape match expectations, and")
    print("gate weights sum to ~1.0 per sample, the model architecture is correct.")