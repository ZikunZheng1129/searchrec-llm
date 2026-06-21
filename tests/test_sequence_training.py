from pathlib import Path

import pandas as pd
import torch

from src.sequence_models.dataset import (
    SequenceTrainDataset,
    build_item_id_mapping,
    build_train_examples,
    sequence_collate_fn,
)
from src.sequence_models.evaluate import load_sequence_checkpoint
from src.sequence_models.gru4rec import GRU4Rec
from src.sequence_models.losses import sample_negative_item_ids, sampled_bce_loss
from src.sequence_models.sasrec import SASRec
from src.sequence_models.trainer import train_one_epoch, train_sequence_model


def _items() -> pd.DataFrame:
    return pd.DataFrame({"item_id": ["item_a", "item_b", "item_c", "item_d"]})


def _train() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "item_a", "timestamp": "2024-01-01"},
            {"user_id": "u1", "item_id": "item_b", "timestamp": "2024-01-02"},
            {"user_id": "u1", "item_id": "item_c", "timestamp": "2024-01-03"},
            {"user_id": "u2", "item_id": "item_b", "timestamp": "2024-01-01"},
            {"user_id": "u2", "item_id": "item_d", "timestamp": "2024-01-02"},
        ]
    )


def test_sample_negative_ids_shape_and_range() -> None:
    positive = torch.tensor([1, 2])
    negatives = sample_negative_item_ids(
        positive,
        num_items=4,
        num_negatives=3,
        generator=torch.Generator().manual_seed(7),
    )

    assert negatives.shape == (2, 3)
    assert int(negatives.min()) >= 1
    assert int(negatives.max()) <= 4


def test_sampled_bce_loss_is_finite_scalar() -> None:
    loss = sampled_bce_loss(torch.tensor([1.0, 2.0]), torch.tensor([[0.0, -1.0], [0.5, -0.5]]))

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def _one_training_step(model: torch.nn.Module) -> float:
    mappings = build_item_id_mapping(_items(), _train())
    examples = build_train_examples(_train(), mappings, max_seq_len=4)
    loader = torch.utils.data.DataLoader(
        SequenceTrainDataset(examples),
        batch_size=2,
        collate_fn=sequence_collate_fn,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return train_one_epoch(
        model=model,
        data_loader=loader,
        optimizer=optimizer,
        num_items=mappings.num_items,
        num_negatives=2,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(3),
    )


def test_one_tiny_gru4rec_training_step_runs() -> None:
    loss = _one_training_step(GRU4Rec(num_items=4, embedding_dim=8, hidden_dim=8))

    assert loss > 0


def test_one_tiny_sasrec_training_step_runs() -> None:
    loss = _one_training_step(
        SASRec(num_items=4, max_seq_len=4, embedding_dim=8, num_heads=2, num_layers=1)
    )

    assert loss > 0


def test_checkpoint_save_and_load(tmp_path: Path) -> None:
    train_path = tmp_path / "train.parquet"
    items_path = tmp_path / "items.parquet"
    checkpoint_path = tmp_path / "model.pt"
    _train().to_parquet(train_path, index=False)
    _items().to_parquet(items_path, index=False)
    config = {
        "seed": 11,
        "input": {"train_path": str(train_path), "item_metadata_path": str(items_path)},
        "output": {"checkpoint_path": str(checkpoint_path)},
        "sequence": {"method": "gru4rec", "max_seq_len": 4, "min_seq_len": 2},
        "model": {"embedding_dim": 8, "hidden_dim": 8, "num_layers": 1, "dropout": 0.0},
        "training": {
            "device": "cpu",
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "num_negatives": 2,
        },
    }

    summary = train_sequence_model(config)
    model, mappings, loaded_config = load_sequence_checkpoint(checkpoint_path)

    assert Path(summary["checkpoint_path"]).exists()
    assert mappings.num_items == 4
    assert loaded_config["sequence"]["method"] == "gru4rec"
    assert isinstance(model, GRU4Rec)
