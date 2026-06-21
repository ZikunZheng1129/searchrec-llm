"""SASRec-style Transformer sequence model."""

from __future__ import annotations

import torch
from torch import nn


class SASRec(nn.Module):
    """Small SASRec-style next-item model."""

    def __init__(
        self,
        num_items: int,
        max_seq_len: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.num_items = int(num_items)
        self.max_seq_len = int(max_seq_len)
        self.padding_idx = int(padding_idx)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=padding_idx)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return final non-padding Transformer representation."""
        batch_size, seq_len = input_ids.shape
        positions = (
            torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        )
        embeddings = self.item_embedding(input_ids) + self.position_embedding(positions)
        embeddings = self.layer_norm(embeddings)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        padding_mask = input_ids == self.padding_idx
        outputs = self.encoder(
            embeddings,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        lengths = (input_ids != self.padding_idx).sum(dim=1).clamp(min=1)
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        return outputs[batch_indices, lengths - 1]

    def score_items(
        self,
        user_repr: torch.Tensor,
        item_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score all or selected items by dot product."""
        if item_ids is None:
            return user_repr @ self.item_embedding.weight.T
        item_embeddings = self.item_embedding(item_ids)
        if item_ids.dim() == 1:
            return user_repr @ item_embeddings.T
        return (user_repr.unsqueeze(1) * item_embeddings).sum(dim=-1)

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
