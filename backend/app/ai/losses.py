"""
losses.py
---------
Self-supervised masked loss and the training step, adapted for the
streaming pipeline. The key adaptation from the original prototype:

  - The original masked_mse_loss took a single mask meaning "this item was
    rated at all" and trained the model to reconstruct its own input
    (autoencoding, not prediction).
  - The streaming pipeline's SparseUserVectorDataset instead produces a
    `hidden_mask` -- a *subset* of the known ratings, randomly held out per
    draw -- specifically so the model is scored only on ratings it did NOT
    see in its input. This is what makes it a genuinely predictive
    network rather than an identity-reconstructing autoencoder: the model
    must infer a hidden rating from the *other*, still-visible ratings,
    not from itself.

masked_mse_loss itself is generic over "which mask defines the loss
region" -- it is called with hidden_mask during training (predictive,
self-supervised) and could equally be called with a full known-ratings
mask if a caller ever wants classic reconstruction-style evaluation. The
mask semantics are the training loop's responsibility, not the loss
function's.
"""

from __future__ import annotations

import torch

from app.ai.model import DualModeHCAIAutoEncoder


def masked_mse_loss(
    predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Mean squared error computed strictly over positions where `mask` is
    True, completely ignoring every other position (including all unrated
    items, which are zero in `targets` but must never contribute to the
    loss -- otherwise the model would be trained to predict "0 stars" for
    every movie a user simply hasn't watched).

    Args:
        predictions: (batch, num_movies) float32.
        targets: (batch, num_movies) float32.
        mask: (batch, num_movies) bool or float -- True/1.0 marks positions
              to include in the loss.

    Returns:
        Scalar loss tensor. Returns 0.0 (via the epsilon-stabilized
        denominator) rather than NaN if mask has no True positions at all,
        which can legitimately happen for a batch where every user's
        held-out fraction rounds down to zero hidden ratings.
    """
    squared_error = (predictions - targets) ** 2
    mask_float = mask.float()
    masked_error = squared_error * mask_float
    num_elements = mask_float.sum()
    return masked_error.sum() / (num_elements + 1e-8)


def train_step(
    model: DualModeHCAIAutoEncoder,
    optimizer: torch.optim.Optimizer,
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    hidden_mask_batch: torch.Tensor,
    lambda_reg: float = 0.05,
) -> tuple:
    """
    One optimized HCAI training step against streamed, masked mini-batches.

    Args:
        model: the DualModeHCAIAutoEncoder being trained.
        optimizer: its optimizer.
        input_batch: (batch, num_movies) -- visible ratings only; hidden
                     ratings are zeroed out (this is what the model sees).
        target_batch: (batch, num_movies) -- ground-truth ratings,
                       including the ones that were hidden from the input.
        hidden_mask_batch: (batch, num_movies) bool -- True exactly at the
                            positions that were hidden this draw; loss is
                            computed strictly here.
        lambda_reg: weight of the residual-pathway L2 regularization term
                    (see DualModeHCAIAutoEncoder.get_semantic_loss).

    Returns:
        (total_loss, pred_loss, semantic_loss) as Python floats.
    """
    optimizer.zero_grad()

    predictions, _ = model.forward_standard(input_batch)

    pred_loss = masked_mse_loss(predictions, target_batch, hidden_mask_batch)
    semantic_loss = model.get_semantic_loss()
    total_loss = pred_loss + (lambda_reg * semantic_loss)

    total_loss.backward()
    optimizer.step()

    return total_loss.item(), pred_loss.item(), semantic_loss.item()
