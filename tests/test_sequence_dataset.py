import pandas as pd
import torch

from src.sequence_models.dataset import (
    SequenceEvalDataset,
    SequenceTrainDataset,
    build_eval_examples,
    build_item_id_mapping,
    build_train_examples,
    sequence_collate_fn,
)


def _items() -> pd.DataFrame:
    return pd.DataFrame({"item_id": ["item_a", "item_b", "item_c"]})


def _train() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "item_a", "timestamp": "2024-01-01"},
            {"user_id": "u1", "item_id": "item_b", "timestamp": "2024-01-02"},
            {"user_id": "u1", "item_id": "item_c", "timestamp": "2024-01-03"},
        ]
    )


def test_item_mapping_reserves_zero_for_padding() -> None:
    mappings = build_item_id_mapping(_items(), _train())

    assert mappings.padding_idx == 0
    assert 0 not in mappings.index_to_item_id
    assert min(mappings.item_id_to_index.values()) == 1


def test_train_examples_are_prefix_targets() -> None:
    mappings = build_item_id_mapping(_items(), _train())
    examples = build_train_examples(_train(), mappings, max_seq_len=4, min_seq_len=2)

    assert len(examples) == 2
    assert examples[0]["target_id"] == mappings.item_id_to_index["item_b"]
    assert examples[1]["target_id"] == mappings.item_id_to_index["item_c"]


def test_sequences_are_left_padded_and_truncated() -> None:
    mappings = build_item_id_mapping(_items(), _train())
    examples = build_train_examples(_train(), mappings, max_seq_len=2, min_seq_len=2)

    assert len(examples[1]["input_ids"]) == 2
    assert examples[0]["input_ids"][0] == 0


def test_collate_returns_expected_tensor_shapes() -> None:
    mappings = build_item_id_mapping(_items(), _train())
    examples = build_train_examples(_train(), mappings, max_seq_len=4, min_seq_len=2)
    batch = sequence_collate_fn([examples[0], examples[1]])

    assert isinstance(batch["input_ids"], torch.Tensor)
    assert batch["input_ids"].shape == (2, 4)
    assert batch["target_ids"].shape == (2,)


def test_eval_examples_use_train_history_and_eval_target() -> None:
    mappings = build_item_id_mapping(_items(), _train())
    eval_interactions = pd.DataFrame(
        [{"user_id": "u1", "item_id": "item_c", "timestamp": "2024-01-04"}]
    )

    examples = build_eval_examples(_train(), eval_interactions, mappings, max_seq_len=4)

    assert len(SequenceEvalDataset(examples)) == 1
    assert examples[0]["target_ids"] == [mappings.item_id_to_index["item_c"]]
    assert len(SequenceTrainDataset(build_train_examples(_train(), mappings, 4))) == 2
