"""Sequence dataset utilities for next-item prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SequenceMappings:
    """Item ID mappings for sequence models."""

    item_id_to_index: dict[str, int]
    index_to_item_id: dict[int, str]
    padding_idx: int = 0

    @property
    def num_items(self) -> int:
        """Number of real items, excluding padding."""
        return len(self.item_id_to_index)


def build_item_id_mapping(items: pd.DataFrame, interactions: pd.DataFrame) -> SequenceMappings:
    """Build item mappings with 0 reserved for padding."""
    item_ids = set()
    if "item_id" in items.columns:
        item_ids.update(items["item_id"].dropna().astype(str).tolist())
    if "item_id" in interactions.columns:
        item_ids.update(interactions["item_id"].dropna().astype(str).tolist())

    sorted_item_ids = sorted(item_ids)
    item_id_to_index = {item_id: index for index, item_id in enumerate(sorted_item_ids, start=1)}
    index_to_item_id = {index: item_id for item_id, index in item_id_to_index.items()}
    return SequenceMappings(item_id_to_index=item_id_to_index, index_to_item_id=index_to_item_id)


def encode_interactions(interactions: pd.DataFrame, mappings: SequenceMappings) -> pd.DataFrame:
    """Add integer item IDs for sequence models."""
    encoded = interactions.copy()
    if encoded.empty:
        encoded["item_idx"] = []
        return encoded
    encoded["user_id"] = encoded["user_id"].astype(str)
    encoded["item_id"] = encoded["item_id"].astype(str)
    encoded["timestamp"] = pd.to_datetime(encoded["timestamp"], errors="coerce")
    encoded["item_idx"] = encoded["item_id"].map(mappings.item_id_to_index)
    encoded = encoded.dropna(subset=["item_idx"]).copy()
    encoded["item_idx"] = encoded["item_idx"].astype(int)
    return encoded.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)


def _truncate(sequence: list[int], max_seq_len: int) -> list[int]:
    return sequence[-max_seq_len:]


def _left_pad(sequence: list[int], max_seq_len: int, padding_idx: int = 0) -> list[int]:
    truncated = _truncate(sequence, max_seq_len)
    return [padding_idx] * (max_seq_len - len(truncated)) + truncated


def build_train_examples(
    train_interactions: pd.DataFrame,
    mappings: SequenceMappings,
    max_seq_len: int,
    min_seq_len: int = 2,
) -> list[dict[str, Any]]:
    """Create prefix-to-next-item training examples from chronological histories."""
    encoded = encode_interactions(train_interactions, mappings)
    examples: list[dict[str, Any]] = []
    for user_id, group in encoded.groupby("user_id", sort=True):
        sequence = group["item_idx"].tolist()
        if len(sequence) < min_seq_len:
            continue
        for target_position in range(1, len(sequence)):
            prefix = sequence[:target_position]
            examples.append(
                {
                    "user_id": str(user_id),
                    "input_ids": _left_pad(prefix, max_seq_len, mappings.padding_idx),
                    "target_id": int(sequence[target_position]),
                    "seen_ids": sorted(set(prefix)),
                }
            )
    return examples


def build_eval_examples(
    train_interactions: pd.DataFrame,
    eval_interactions: pd.DataFrame,
    mappings: SequenceMappings,
    max_seq_len: int,
) -> list[dict[str, Any]]:
    """Build one evaluation example per user using train history and eval targets."""
    train_encoded = encode_interactions(train_interactions, mappings)
    eval_encoded = encode_interactions(eval_interactions, mappings)
    history_by_user = {
        str(user_id): group["item_idx"].tolist()
        for user_id, group in train_encoded.groupby("user_id", sort=True)
    }
    examples: list[dict[str, Any]] = []
    for user_id, group in eval_encoded.groupby("user_id", sort=True):
        user_id = str(user_id)
        history = history_by_user.get(user_id, [])
        if not history:
            continue
        target_ids = sorted(set(int(value) for value in group["item_idx"].tolist()))
        if not target_ids:
            continue
        examples.append(
            {
                "user_id": user_id,
                "input_ids": _left_pad(history, max_seq_len, mappings.padding_idx),
                "target_ids": target_ids,
                "seen_ids": sorted(set(history)),
            }
        )
    return examples


class SequenceTrainDataset(Dataset):
    """Torch dataset for sequence training examples."""

    def __init__(self, examples: list[dict[str, Any]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


class SequenceEvalDataset(Dataset):
    """Torch dataset for sequence evaluation examples."""

    def __init__(self, examples: list[dict[str, Any]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


def sequence_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate sequence examples into tensors."""
    input_ids = torch.tensor([row["input_ids"] for row in batch], dtype=torch.long)
    output: dict[str, Any] = {
        "input_ids": input_ids,
        "user_ids": [str(row["user_id"]) for row in batch],
        "seen_ids": [list(row.get("seen_ids", [])) for row in batch],
    }
    if "target_id" in batch[0]:
        output["target_ids"] = torch.tensor([row["target_id"] for row in batch], dtype=torch.long)
    else:
        output["target_ids"] = [list(row["target_ids"]) for row in batch]
    return output
