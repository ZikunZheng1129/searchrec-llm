"""Tiny local cross-encoder ranker without external transformer libraries."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.retrieval.text_utils import build_item_text, tokenize

PAD_ID = 0
CLS_ID = 1
SEP_ID = 2
UNK_ID = 3


class PairTextVectorizer:
    """Build and apply a small token vocabulary for query-item pairs."""

    def __init__(self, max_vocab_size: int = 5000, max_pair_len: int = 64) -> None:
        self.max_vocab_size = int(max_vocab_size)
        self.max_pair_len = int(max_pair_len)
        self.vocabulary: dict[str, int] = {}

    def fit(self, df: pd.DataFrame, items: pd.DataFrame) -> PairTextVectorizer:
        """Fit vocabulary from train query text and item metadata."""
        item_text = _item_text_lookup(items)
        counts: Counter[str] = Counter()
        for _, row in df.iterrows():
            counts.update(tokenize(str(row.get("query_text", ""))))
            counts.update(tokenize(item_text.get(str(row.get("candidate_item_id")), "")))
        sorted_terms = sorted(counts.items(), key=lambda row: (-row[1], row[0]))
        limit = max(0, self.max_vocab_size - 4)
        self.vocabulary = {term: index + 4 for index, (term, _) in enumerate(sorted_terms[:limit])}
        return self

    def encode_pair(self, query_text: str, item_text: str) -> list[int]:
        """Encode one query-item pair as [CLS] query [SEP] item."""
        query_ids = [self.vocabulary.get(token, UNK_ID) for token in tokenize(query_text)]
        item_ids = [self.vocabulary.get(token, UNK_ID) for token in tokenize(item_text)]
        ids = [CLS_ID, *query_ids, SEP_ID, *item_ids]
        ids = ids[: self.max_pair_len]
        if len(ids) < self.max_pair_len:
            ids.extend([PAD_ID] * (self.max_pair_len - len(ids)))
        return ids

    def to_dict(self) -> dict[str, Any]:
        """Serialize vectorizer state."""
        return {
            "max_vocab_size": self.max_vocab_size,
            "max_pair_len": self.max_pair_len,
            "vocabulary": self.vocabulary,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> PairTextVectorizer:
        """Restore vectorizer state."""
        vectorizer = cls(
            max_vocab_size=int(state.get("max_vocab_size", 5000)),
            max_pair_len=int(state.get("max_pair_len", 64)),
        )
        vectorizer.vocabulary = {str(key): int(value) for key, value in state["vocabulary"].items()}
        return vectorizer


def _item_text_lookup(items: pd.DataFrame) -> dict[str, str]:
    table = items.copy()
    table["item_id"] = table["item_id"].astype(str)
    return {str(row["item_id"]): build_item_text(row) for _, row in table.iterrows()}


class CrossEncoderDataset(Dataset):
    """Torch dataset for text-pair ranking."""

    def __init__(
        self,
        df: pd.DataFrame,
        items: pd.DataFrame,
        vectorizer: PairTextVectorizer,
        label_column: str = "label",
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.item_text = _item_text_lookup(items)
        self.vectorizer = vectorizer
        self.label_column = label_column

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        input_ids = self.vectorizer.encode_pair(
            str(row.get("query_text", "")),
            self.item_text.get(str(row.get("candidate_item_id", "")), ""),
        )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label": torch.tensor(float(row[self.label_column]), dtype=torch.float32),
        }


class TinyCrossEncoderRanker(torch.nn.Module):
    """A tiny TransformerEncoder classifier over query-item token pairs."""

    def __init__(
        self,
        vocab_size: int,
        max_pair_len: int,
        embedding_dim: int = 32,
        num_heads: int = 2,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.position_embedding = torch.nn.Embedding(max_pair_len, embedding_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = torch.nn.Linear(embedding_dim, 1)
        self.max_pair_len = int(max_pair_len)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        embeddings = self.token_embedding(input_ids) + self.position_embedding(positions)
        padding_mask = input_ids.eq(PAD_ID)
        encoded = self.encoder(embeddings, src_key_padding_mask=padding_mask)
        cls_repr = encoded[:, 0, :]
        return self.classifier(cls_repr).squeeze(-1)


class CrossEncoderRankerWrapper:
    """Train, save, load, and score the tiny cross-encoder."""

    backend = "tiny_torch_transformer"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.vectorizer: PairTextVectorizer | None = None
        self.model: TinyCrossEncoderRanker | None = None
        self.device = torch.device("cpu")

    def fit(
        self,
        train_df: pd.DataFrame,
        items: pd.DataFrame,
        label_column: str = "label",
    ) -> CrossEncoderRankerWrapper:
        """Fit the tiny cross-encoder."""
        seed = int(self.config.get("seed", 42))
        torch.manual_seed(seed)
        np.random.seed(seed)
        training_config = self.config.get("training", {})
        model_config = self.config.get("model", {})
        self.device = torch.device(str(training_config.get("device", "cpu")))
        self.vectorizer = PairTextVectorizer(
            max_vocab_size=int(model_config.get("vocab_size", 5000)),
            max_pair_len=int(model_config.get("max_pair_len", 64)),
        ).fit(train_df, items)
        vocab_size = max([UNK_ID, *self.vectorizer.vocabulary.values()]) + 1
        self.model = TinyCrossEncoderRanker(
            vocab_size=vocab_size,
            max_pair_len=self.vectorizer.max_pair_len,
            embedding_dim=int(model_config.get("embedding_dim", 32)),
            num_heads=int(model_config.get("num_heads", 2)),
            num_layers=int(model_config.get("num_layers", 1)),
            dropout=float(model_config.get("dropout", 0.1)),
        ).to(self.device)
        dataset = CrossEncoderDataset(train_df, items, self.vectorizer, label_column)
        loader = DataLoader(
            dataset,
            batch_size=int(training_config.get("batch_size", 32)),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(training_config.get("learning_rate", 0.001)),
            weight_decay=float(training_config.get("weight_decay", 0.0)),
        )
        loss_fn = torch.nn.BCEWithLogitsLoss()
        self.model.train()
        for _ in range(int(training_config.get("epochs", 3))):
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["label"].to(self.device)
                logits = self.model(input_ids)
                loss = loss_fn(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self

    def predict(self, df: pd.DataFrame, items: pd.DataFrame) -> np.ndarray:
        """Predict relevance scores."""
        if self.model is None or self.vectorizer is None:
            return np.zeros(len(df), dtype=float)
        dataset = CrossEncoderDataset(df, items, self.vectorizer)
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        scores: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                logits = self.model(batch["input_ids"].to(self.device))
                scores.append(torch.sigmoid(logits).cpu().numpy().astype(float))
        return np.concatenate(scores) if scores else np.zeros(0, dtype=float)

    def save(self, path: str | Path) -> None:
        """Save model checkpoint."""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Cannot save an unfit cross-encoder ranker")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "vectorizer": self.vectorizer.to_dict(),
                "config": self.config,
                "backend": self.backend,
            },
            output,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> CrossEncoderRankerWrapper:
        """Load a saved cross-encoder ranker."""
        checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
        wrapper = cls(checkpoint.get("config", {}))
        wrapper.vectorizer = PairTextVectorizer.from_dict(checkpoint["vectorizer"])
        wrapper.device = torch.device(device)
        model_config = wrapper.config.get("model", {})
        vocab_size = max([UNK_ID, *wrapper.vectorizer.vocabulary.values()]) + 1
        wrapper.model = TinyCrossEncoderRanker(
            vocab_size=vocab_size,
            max_pair_len=wrapper.vectorizer.max_pair_len,
            embedding_dim=int(model_config.get("embedding_dim", 32)),
            num_heads=int(model_config.get("num_heads", 2)),
            num_layers=int(model_config.get("num_layers", 1)),
            dropout=float(model_config.get("dropout", 0.1)),
        ).to(wrapper.device)
        wrapper.model.load_state_dict(checkpoint["model_state_dict"])
        return wrapper
