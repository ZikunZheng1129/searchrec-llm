"""GRU4Rec sequence model."""

from __future__ import annotations

import torch
from torch import nn


class GRU4Rec(nn.Module):
    """Small GRU4Rec-style next-item model."""

    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.num_items = int(num_items)
        self.padding_idx = int(padding_idx)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=padding_idx)
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.output_projection = (
            nn.Identity() if hidden_dim == embedding_dim else nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return final non-padding hidden representation."""
        embeddings = self.item_embedding(input_ids)
        outputs, _ = self.gru(embeddings)
        lengths = (input_ids != self.padding_idx).sum(dim=1).clamp(min=1)
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        return outputs[batch_indices, lengths - 1]

    def score_items(
        self,
        user_repr: torch.Tensor,
        item_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score all or selected items by dot product."""
        projected = self.output_projection(user_repr)
        if item_ids is None:
            return projected @ self.item_embedding.weight.T
        item_embeddings = self.item_embedding(item_ids)
        if item_ids.dim() == 1:
            return projected @ item_embeddings.T
        return (projected.unsqueeze(1) * item_embeddings).sum(dim=-1)

    def recommend(
        self,
        input_ids: torch.Tensor,
        top_k: int = 10,
        exclude_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recommend top-k encoded item IDs."""
        user_repr = self.forward(input_ids)
        scores = self.score_items(user_repr)
        scores[:, self.padding_idx] = float("-inf")
        if exclude_ids is not None:
            for row_index in range(scores.shape[0]):
                valid_ids = exclude_ids[row_index][exclude_ids[row_index] > 0]
                scores[row_index, valid_ids] = float("-inf")
        limit = min(int(top_k), self.num_items)
        top_scores, top_ids = torch.topk(scores, k=limit, dim=1)
        return top_scores, top_ids
