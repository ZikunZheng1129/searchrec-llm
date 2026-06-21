"""Loss helpers for sequential recommendation models."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_negative_item_ids(
    positive_ids: torch.Tensor,
    num_items: int,
    num_negatives: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample negative item IDs in [1, num_items], avoiding positives when practical."""
    if num_items <= 0 or num_negatives <= 0:
        return torch.empty((positive_ids.shape[0], 0), dtype=torch.long, device=positive_ids.device)

    negatives = torch.randint(
        low=1,
        high=num_items + 1,
        size=(positive_ids.shape[0], num_negatives),
        generator=generator,
        device=positive_ids.device,
    )
    if num_items > 1:
        positive_matrix = positive_ids.view(-1, 1).expand_as(negatives)
        collision_mask = negatives == positive_matrix
        while collision_mask.any():
            replacement = torch.randint(
                low=1,
                high=num_items + 1,
                size=negatives.shape,
                generator=generator,
                device=positive_ids.device,
            )
            negatives = torch.where(collision_mask, replacement, negatives)
            collision_mask = negatives == positive_matrix
    return negatives


def sampled_bce_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy over positive and sampled negative scores."""
    positive_loss = F.binary_cross_entropy_with_logits(
        positive_scores,
        torch.ones_like(positive_scores),
    )
    negative_loss = F.binary_cross_entropy_with_logits(
        negative_scores,
        torch.zeros_like(negative_scores),
    )
    return positive_loss + negative_loss
