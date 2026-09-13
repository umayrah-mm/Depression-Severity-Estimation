"""
evidence_training_utils.py

Two pieces needed to actually TRAIN the EvidenceFusionModel:

1. apply_modality_dropout(...)
   A training-time data augmentation. Randomly hides a modality for a
   sample (zeroes its raw features, flags it missing in the mask) so
   the model gets practice handling missing modalities during training,
   not just at evaluation time. At least one modality is always kept
   per sample.

2. heteroscedastic_loss(...) and combined_loss(...)
   The loss functions. heteroscedastic_loss trains each modality's own
   standalone prediction + confidence pair (see ModalityHead in
   models_evidence.py). combined_loss adds that to a plain loss on the
   FINAL fused prediction, which is what we actually care about at the
   end of the day.

Nothing in this file trains anything by itself - it just provides the
pieces the training script (a later module) will call every batch.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Modality dropout augmentation
# ---------------------------------------------------------------------------
def apply_modality_dropout(visual, clip, rppg, smile, dropout_prob=0.15, min_present=1):
    """
    Randomly hides modalities for training, so the model practices
    handling missing data instead of only ever seeing all 4 present.

    Parameters
    ----------
    visual, clip, rppg, smile : Tensors for one batch, all modalities
                                 assumed genuinely present (this is what
                                 comes straight out of your real dataset).
    dropout_prob : float, probability that ANY GIVEN modality is hidden
                   for ANY GIVEN sample, independently. 0.15 means each
                   modality has a 15% chance of being hidden per sample.
    min_present : int, minimum number of modalities that must remain
                  visible for every sample (default 1 - never hide all 4,
                  since the model would have literally nothing to learn
                  from on that sample).

    Returns
    -------
    visual, clip, rppg, smile : same shapes as input, but with some
                                 samples' modalities zeroed out.
    mask : Tensor, shape (batch_size, 4), 1 = kept, 0 = hidden.
           Column order: [visual, clip, rppg, smile] - matches the order
           EvidenceFusionModel.forward() expects.
    """
    batch_size = visual.shape[0]
    device = visual.device

    # Start with a random 0/1 mask: 1 with probability (1 - dropout_prob)
    mask = (torch.rand(batch_size, 4, device=device) >= dropout_prob).float()

    # Safety pass: if any sample ended up with fewer than min_present
    # modalities kept, randomly flip some hidden ones back on for that
    # sample until it meets the minimum. This loop runs at most a few
    # times per sample and batch sizes here are small, so the cost is
    # negligible.
    num_present = mask.sum(dim=1)
    for i in range(batch_size):
        while num_present[i] < min_present:
            zero_positions = (mask[i] == 0).nonzero(as_tuple=True)[0]
            flip_choice = zero_positions[torch.randint(len(zero_positions), (1,))]
            mask[i, flip_choice] = 1
            num_present[i] = mask[i].sum()

    # Zero out the raw features wherever the mask says "hidden"
    visual_out = visual.clone()
    clip_out = clip.clone()
    rppg_out = rppg.clone()
    smile_out = smile.clone()
    tensors = [visual_out, clip_out, rppg_out, smile_out]

    for modality_idx in range(4):
        hidden_rows = (mask[:, modality_idx] == 0)
        tensors[modality_idx][hidden_rows] = 0

    return tensors[0], tensors[1], tensors[2], tensors[3], mask


# ---------------------------------------------------------------------------
# 2. Heteroscedastic loss (trains the per-modality confidence estimates)
# ---------------------------------------------------------------------------
def heteroscedastic_loss(aux_predictions, log_vars, targets, mask):
    """
    Trains each modality's standalone (aux_prediction, log_variance) pair
    to be well-calibrated: confident when right, appropriately unsure
    when wrong. Modalities marked missing in `mask` are excluded
    entirely - their aux_prediction was computed from zero-filled input
    and is meaningless, so it must not influence training.

    Parameters
    ----------
    aux_predictions : Tensor, shape (batch_size, 4)
    log_vars        : Tensor, shape (batch_size, 4)
    targets         : Tensor, shape (batch_size,) - the true BDI-II scores
    mask            : Tensor, shape (batch_size, 4), 1 = present, 0 = missing

    Returns
    -------
    A single scalar loss (averaged only over present modality-entries).
    """
    targets_expanded = targets.unsqueeze(1).expand(-1, 4)  # (batch, 4)

    squared_error = (aux_predictions - targets_expanded) ** 2  # (batch, 4)
    precision = torch.exp(-log_vars)  # (batch, 4)

    # The two competing terms described above.
    per_modality_loss = 0.5 * precision * squared_error + 0.5 * log_vars  # (batch, 4)

    # Zero out any entries where the modality was missing, then average
    # only over the entries that were actually present.
    masked_loss = per_modality_loss * mask
    num_present_entries = mask.sum()

    # Guard against dividing by zero in the (extremely unlikely) case a
    # whole batch had zero present entries for a modality.
    loss = masked_loss.sum() / (num_present_entries + 1e-8)
    return loss


# ---------------------------------------------------------------------------
# 3. Combined loss (what the training script will actually call)
# ---------------------------------------------------------------------------
def combined_loss(outputs, targets, mask, aux_loss_weight=0.3):
    """
    The full training objective for one batch.

    main_loss: plain MAE (L1 loss) between the FINAL fused prediction
               and the true score. This is the number we actually
               report and care about (matches how every other model in
               this project is evaluated).
    aux_loss:  the heteroscedastic loss described above, which teaches
               each modality to predict calibrated confidence.

    aux_loss_weight controls how much the auxiliary objective is allowed
    to influence training relative to the main objective. 0.3 is a
    reasonable starting point - strong enough that the log-variance
    heads actually learn something, not so strong that it distracts
    from the main prediction task. This is a hyperparameter we may tune
    later if training behaves oddly.

    Parameters
    ----------
    outputs : the dict returned by EvidenceFusionModel.forward()
    targets : Tensor, shape (batch_size,) - true BDI-II scores
    mask    : Tensor, shape (batch_size, 4) - same mask passed into the model
    aux_loss_weight : float

    Returns
    -------
    total_loss : scalar Tensor - call .backward() on this one
    main_loss  : scalar Tensor - for logging/monitoring only
    aux_loss   : scalar Tensor - for logging/monitoring only
    """
    prediction = outputs["prediction"].squeeze(-1)  # (batch,) <- (batch, 1)
    main_loss = F.smooth_l1_loss(prediction, targets)

    aux_loss = heteroscedastic_loss(
        outputs["aux_predictions"], outputs["log_vars"], targets, mask
    )

    total_loss = main_loss + aux_loss_weight * aux_loss
    return total_loss, main_loss, aux_loss


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running evidence_training_utils.py self-test...\n")

    torch.manual_seed(0)

    # -----------------------------------------------------------------
    # Test A: modality dropout never drops all 4 modalities for any sample
    # -----------------------------------------------------------------
    print("--- Test A: modality dropout respects min_present ---")
    batch_size = 200  # large batch to make it very likely all-4-hidden would occur by chance if unguarded
    visual = torch.randn(batch_size, 2304)
    clip = torch.randn(batch_size, 512)
    rppg = torch.randn(batch_size, 9)
    smile = torch.randn(batch_size, 88)

    v, c, r, s, mask = apply_modality_dropout(visual, clip, rppg, smile, dropout_prob=0.5)
    num_present_per_sample = mask.sum(dim=1)
    assert num_present_per_sample.min().item() >= 1, "Found a sample with 0 modalities present!"
    print(f"Min modalities present across {batch_size} samples: {num_present_per_sample.min().item()} (expected >= 1)")
    print(f"Average modalities present: {num_present_per_sample.mean().item():.2f} (expected roughly 2.0 at dropout_prob=0.5)")

    # -----------------------------------------------------------------
    # Test B: heteroscedastic loss rewards honest uncertainty when wrong
    # -----------------------------------------------------------------
    print("\n--- Test B: heteroscedastic loss sanity checks ---")
    targets = torch.tensor([20.0])  # a single fake true BDI-II score

    # Scenario 1: prediction is very wrong, but model is falsely CONFIDENT (log_var = 0)
    aux_pred_wrong_confident = torch.tensor([[5.0, 0.0, 0.0, 0.0]])
    log_var_confident = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    mask_only_first = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    loss_wrong_confident = heteroscedastic_loss(
        aux_pred_wrong_confident, log_var_confident, targets, mask_only_first
    )

    # Scenario 2: same wrong prediction, but model HONESTLY says it's unsure (large log_var)
    log_var_uncertain = torch.tensor([[4.0, 0.0, 0.0, 0.0]])
    loss_wrong_uncertain = heteroscedastic_loss(
        aux_pred_wrong_confident, log_var_uncertain, targets, mask_only_first
    )

    print(f"Loss when WRONG + falsely confident:  {loss_wrong_confident.item():.4f}")
    print(f"Loss when WRONG + honestly uncertain:  {loss_wrong_uncertain.item():.4f}")
    assert loss_wrong_uncertain.item() < loss_wrong_confident.item(), (
        "Uncertainty should reduce the loss when the prediction is wrong!"
    )
    print("PASS: being honestly uncertain when wrong costs less than being falsely confident.\n")

    # Scenario 3: prediction is CORRECT and confident - should be the cheapest of all
    aux_pred_correct_confident = torch.tensor([[20.0, 0.0, 0.0, 0.0]])
    loss_correct_confident = heteroscedastic_loss(
        aux_pred_correct_confident, log_var_confident, targets, mask_only_first
    )
    print(f"Loss when CORRECT + confident:        {loss_correct_confident.item():.4f}")
    assert loss_correct_confident.item() < loss_wrong_uncertain.item(), (
        "Being right and confident should always beat being wrong, even honestly uncertain!"
    )
    print("PASS: correct + confident is cheaper than wrong + uncertain, as expected.\n")

    # -----------------------------------------------------------------
    # Test C: masked-out modalities contribute exactly zero to the loss
    # -----------------------------------------------------------------
    print("--- Test C: missing modalities are fully excluded from the loss ---")
    aux_pred_with_garbage = torch.tensor([[20.0, 999.0, -999.0, 500.0]])  # 3 huge, wrong garbage values
    log_var_for_garbage = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    mask_first_only = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # only modality 0 counts

    loss_with_garbage_masked = heteroscedastic_loss(
        aux_pred_with_garbage, log_var_for_garbage, targets, mask_first_only
    )
    print(f"Loss with 3 garbage predictions correctly masked out: {loss_with_garbage_masked.item():.4f}")
    print(f"(Should match the 'CORRECT + confident' loss above: {loss_correct_confident.item():.4f})")
    assert abs(loss_with_garbage_masked.item() - loss_correct_confident.item()) < 1e-4, (
        "Masked-out garbage values are leaking into the loss!"
    )
    print("PASS: garbage values from missing modalities do not affect the loss.\n")

    # -----------------------------------------------------------------
    # Test D: combined_loss runs end-to-end through the real model + backward()
    # -----------------------------------------------------------------
    print("--- Test D: combined_loss works with the real model end-to-end ---")
    from models_evidence import EvidenceFusionModel

    model = EvidenceFusionModel()
    model.train()

    batch_size = 8
    visual = torch.randn(batch_size, 2304)
    clip = torch.randn(batch_size, 512)
    rppg = torch.randn(batch_size, 9)
    smile = torch.randn(batch_size, 88)
    fake_targets = torch.randint(0, 63, (batch_size,)).float()

    v, c, r, s, mask = apply_modality_dropout(visual, clip, rppg, smile, dropout_prob=0.15)
    outputs = model(v, c, r, s, mask)

    total_loss, main_loss, aux_loss = combined_loss(outputs, fake_targets, mask, aux_loss_weight=0.3)
    assert not torch.isnan(total_loss).any(), "total_loss is NaN!"

    total_loss.backward()
    nan_grad_found = any(
        p.grad is not None and torch.isnan(p.grad).any() for p in model.parameters()
    )
    assert not nan_grad_found, "NaN gradient found after combined_loss.backward()!"

    print(f"main_loss:  {main_loss.item():.4f}")
    print(f"aux_loss:   {aux_loss.item():.4f}")
    print(f"total_loss: {total_loss.item():.4f}")
    print("PASS: combined_loss runs end-to-end with the real model, backward() works, no NaNs.\n")

    print("=" * 70)
    print("ALL SELF-TESTS PASSED.")
    print("Loss functions and modality dropout are ready for the training script.")
    print("=" * 70)