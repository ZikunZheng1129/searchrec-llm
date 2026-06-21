"""Training utilities for sequential recommendation models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.sequence_models.dataset import (
    SequenceTrainDataset,
    build_item_id_mapping,
    build_train_examples,
    sequence_collate_fn,
)
from src.sequence_models.gru4rec import GRU4Rec
from src.sequence_models.losses import sample_negative_item_ids, sampled_bce_loss
from src.sequence_models.sasrec import SASRec
from src.utils.config import resolve_project_path
from src.utils.io import read_parquet
from src.utils.seed import set_seed


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolve_project_path(str(candidate))


def choose_device(device_config: str) -> torch.device:
    """Choose CPU or CUDA from config."""
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)


def create_sequence_model(method: str, num_items: int, config: dict[str, Any]) -> torch.nn.Module:
    """Create a GRU4Rec or SASRec model from config."""
    model_config = config.get("model", {})
    sequence_config = config.get("sequence", {})
    if method == "gru4rec":
        return GRU4Rec(
            num_items=num_items,
            embedding_dim=int(model_config.get("embedding_dim", 32)),
            hidden_dim=int(model_config.get("hidden_dim", model_config.get("embedding_dim", 32))),
            num_layers=int(model_config.get("num_layers", 1)),
            dropout=float(model_config.get("dropout", 0.1)),
        )
    if method == "sasrec":
        return SASRec(
            num_items=num_items,
            max_seq_len=int(sequence_config.get("max_seq_len", 20)),
            embedding_dim=int(model_config.get("embedding_dim", 32)),
            num_heads=int(model_config.get("num_heads", 2)),
            num_layers=int(model_config.get("num_layers", 1)),
            dropout=float(model_config.get("dropout", 0.1)),
        )
    raise ValueError(f"Unsupported sequence method: {method}")


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_items: int,
    num_negatives: int,
    device: torch.device,
    generator: torch.Generator,
) -> float:
    """Train one epoch with sampled BCE loss."""
    model.train()
    total_loss = 0.0
    total_batches = 0
    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        user_repr = model(input_ids)
        positive_scores = model.score_items(user_repr, target_ids.unsqueeze(1)).squeeze(1)
        negative_ids = sample_negative_item_ids(
            positive_ids=target_ids,
            num_items=num_items,
            num_negatives=num_negatives,
            generator=generator,
        )
        negative_scores = model.score_items(user_repr, negative_ids)
        loss = sampled_bce_loss(positive_scores, negative_scores)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_batches += 1
    return total_loss / max(1, total_batches)


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    data_loader: DataLoader,
    num_items: int,
    num_negatives: int,
    device: torch.device,
    generator: torch.Generator,
) -> float:
    """Compute sampled BCE loss on a data loader."""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        user_repr = model(input_ids)
        positive_scores = model.score_items(user_repr, target_ids.unsqueeze(1)).squeeze(1)
        negative_ids = sample_negative_item_ids(
            positive_ids=target_ids,
            num_items=num_items,
            num_negatives=num_negatives,
            generator=generator,
        )
        negative_scores = model.score_items(user_repr, negative_ids)
        total_loss += float(sampled_bce_loss(positive_scores, negative_scores).cpu())
        total_batches += 1
    return total_loss / max(1, total_batches)


def train_sequence_model(config: dict[str, Any]) -> dict[str, Any]:
    """Train a configured sequence model and save a checkpoint."""
    seed = int(config.get("seed", 42))
    set_seed(seed)
    torch.manual_seed(seed)

    input_config = config.get("input", {})
    train = read_parquet(resolve_path(input_config["train_path"]))
    items = read_parquet(resolve_path(input_config["item_metadata_path"]))
    mappings = build_item_id_mapping(items=items, interactions=train)

    sequence_config = config.get("sequence", {})
    max_seq_len = int(sequence_config.get("max_seq_len", 20))
    min_seq_len = int(sequence_config.get("min_seq_len", 2))
    train_examples = build_train_examples(
        train_interactions=train,
        mappings=mappings,
        max_seq_len=max_seq_len,
        min_seq_len=min_seq_len,
    )
    if not train_examples:
        raise ValueError("No sequence training examples were generated")

    training_config = config.get("training", {})
    device = choose_device(str(training_config.get("device", "cpu")))
    method = str(sequence_config.get("method"))
    model = create_sequence_model(
        method=method,
        num_items=mappings.num_items,
        config=config,
    ).to(device)
    data_loader = DataLoader(
        SequenceTrainDataset(train_examples),
        batch_size=int(training_config.get("batch_size", 16)),
        shuffle=True,
        collate_fn=sequence_collate_fn,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 0.001)),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    negative_generator = torch.Generator(device=device).manual_seed(seed)
    epochs = int(training_config.get("epochs", 2))
    final_train_loss = 0.0
    for _ in range(epochs):
        final_train_loss = train_one_epoch(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            num_items=mappings.num_items,
            num_negatives=int(training_config.get("num_negatives", 20)),
            device=device,
            generator=negative_generator,
        )

    checkpoint_path = resolve_path(config.get("output", {})["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    training_summary = {
        "method": method,
        "device": str(device),
        "num_items": mappings.num_items,
        "num_train_examples": len(train_examples),
        "epochs": epochs,
        "final_train_loss": final_train_loss,
        "checkpoint_path": str(checkpoint_path),
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "mappings": mappings,
            "training_summary": training_summary,
        },
        checkpoint_path,
    )
    return training_summary
