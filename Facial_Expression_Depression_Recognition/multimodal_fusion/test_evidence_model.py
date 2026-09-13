"""
test_evidence_model.py

Thorough self-test for EvidenceFusionModel (from models_evidence.py).

This checks, on fake random data (no real features needed yet):
    1. All single- and double-missing-modality combinations behave correctly
       (weights sum to 1, missing modalities are suppressed, no NaNs).
    2. Gradients flow correctly through a backward pass, including when
       some modalities are missing.

This does NOT test the model's real accuracy (it hasn't been trained on
real data yet) - it only tests that the MECHANICS of the architecture are
correct before we build a training loop on top of it.
"""

import itertools
import torch

from models_evidence import EvidenceFusionModel

torch.manual_seed(42)

# Modality feature dimensions and names, in the model's fixed order.
VISUAL_DIM, CLIP_DIM, RPPG_DIM, SMILE_DIM = 2304, 512, 9, 88
MODALITY_NAMES = ["visual", "clip", "rppg", "smile"]


def make_fake_batch(batch_size=6):
    """Creates one fake batch of random features for all 4 modalities."""
    return {
        "visual": torch.randn(batch_size, VISUAL_DIM),
        "clip": torch.randn(batch_size, CLIP_DIM),
        "rppg": torch.randn(batch_size, RPPG_DIM),
        "smile": torch.randn(batch_size, SMILE_DIM),
    }


def apply_missing(batch, missing_indices):
    """
    Given a fake batch and a tuple of modality indices to mark as missing,
    returns (visual, clip, rppg, smile, mask) with:
      - the missing modalities' raw features zeroed out (as the real data
        loader will eventually do)
      - a mask tensor of shape (batch_size, 4), 0 at missing positions
    """
    batch_size = batch["visual"].shape[0]
    mask = torch.ones(batch_size, 4)

    visual = batch["visual"].clone()
    clip = batch["clip"].clone()
    rppg = batch["rppg"].clone()
    smile = batch["smile"].clone()
    tensors = [visual, clip, rppg, smile]

    for idx in missing_indices:
        mask[:, idx] = 0
        tensors[idx].zero_()

    return tensors[0], tensors[1], tensors[2], tensors[3], mask


def describe(missing_indices):
    if not missing_indices:
        return "none missing"
    return "missing: " + ", ".join(MODALITY_NAMES[i] for i in missing_indices)


def run_missing_modality_checks():
    print("=" * 70)
    print("PART 1: checking every 0/1/2-missing-modality combination")
    print("=" * 70)

    model = EvidenceFusionModel()
    model.eval()
    batch = make_fake_batch(batch_size=6)

    # All combinations: 0 missing (1 case), 1 missing (4 cases), 2 missing (6 cases)
    combos = (
        [()]
        + list(itertools.combinations(range(4), 1))
        + list(itertools.combinations(range(4), 2))
    )
    assert len(combos) == 11, f"expected 11 combinations, got {len(combos)}"

    for combo in combos:
        visual, clip, rppg, smile, mask = apply_missing(batch, combo)

        with torch.no_grad():
            outputs = model(visual, clip, rppg, smile, mask)

        prediction = outputs["prediction"]
        weights = outputs["weights"]
        log_vars = outputs["log_vars"]
        aux_predictions = outputs["aux_predictions"]

        # Check 1: no NaNs anywhere
        assert not torch.isnan(prediction).any(), f"NaN prediction for {describe(combo)}"
        assert not torch.isnan(weights).any(), f"NaN weights for {describe(combo)}"
        assert not torch.isnan(aux_predictions).any(), f"NaN aux_predictions for {describe(combo)}"

        # Check 2: weights sum to 1 per sample
        weight_sums = weights.sum(dim=1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-4), (
            f"weights do not sum to 1 for {describe(combo)}: {weight_sums}"
        )

        # Check 3: missing modalities have near-zero weight and exact hard-floor log-var
        for idx in combo:
            assert torch.all(weights[:, idx] < 0.01), (
                f"missing modality '{MODALITY_NAMES[idx]}' weight too high "
                f"for {describe(combo)}: {weights[:, idx]}"
            )
            expected = torch.full_like(log_vars[:, idx], EvidenceFusionModel.MISSING_LOG_VAR)
            assert torch.allclose(log_vars[:, idx], expected), (
                f"log-variance override failed for '{MODALITY_NAMES[idx]}' "
                f"in {describe(combo)}"
            )

        # Check 4: present modalities' weights still sum to ~1 (since missing ~0)
        present = [i for i in range(4) if i not in combo]
        present_weight_sum = weights[:, present].sum(dim=1)[0].item()

        print(f"{describe(combo):32s} | present-weight-sum: {present_weight_sum:.4f} "
              f"| sample-0 weights: {[round(w, 4) for w in weights[0].tolist()]}")

    print("\nAll 11 combinations passed: correct weights, correct log-variance "
          "override, correct aux_predictions shape, no NaNs.\n")


def run_gradient_check():
    print("=" * 70)
    print("PART 2: checking gradients flow correctly through backward()")
    print("=" * 70)

    model = EvidenceFusionModel()
    model.train()
    batch = make_fake_batch(batch_size=6)

    # Use a two-missing-modality scenario, since that matches the
    # robustness test we'll run later at evaluation time.
    missing_indices = (0, 2)  # visual + rppg missing
    visual, clip, rppg, smile, mask = apply_missing(batch, missing_indices)

    outputs = model(visual, clip, rppg, smile, mask)
    prediction = outputs["prediction"]

    fake_target = torch.randn_like(prediction)
    loss = ((prediction - fake_target) ** 2).mean()
    loss.backward()

    print(f"Scenario: {describe(missing_indices)}")
    print(f"Loss value: {loss.item():.4f}")

    nan_found = False
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"  NaN GRADIENT FOUND in: {name}")
            nan_found = True

    assert not nan_found, "NaN gradients detected - stop and report this before continuing."

    print("No NaN gradients found anywhere in the model.")
    print("Backward pass completed successfully.\n")


if __name__ == "__main__":
    run_missing_modality_checks()
    run_gradient_check()
    print("=" * 70)
    print("ALL CHECKS PASSED. The architecture's mechanics are verified.")
    print("=" * 70)