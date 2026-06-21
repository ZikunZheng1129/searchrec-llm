"""PyTorch MLP ranking model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.ranking.dataset import (
    RankingFeatureDataset,
    apply_feature_standardizer,
)


class MLPRanker(torch.nn.Module):
    """Small feed-forward ranker over numeric features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(previous_dim, int(hidden_dim)))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(float(dropout)))
            previous_dim = int(hidden_dim)
        layers.append(torch.nn.Linear(previous_dim, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class MLPRankerWrapper:
    """Train, save, load, and score an MLPRanker."""

    backend = "torch_mlp"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.feature_columns: list[str] = []
        self.standardizer: dict[str, Any] = {}
        self.model: MLPRanker | None = None
        self.device = torch.device("cpu")

    def _fit_standardizer(self, train_df: pd.DataFrame) -> None:
        means = train_df[self.feature_columns].astype(float).mean(axis=0)
        stds = (
            train_df[self.feature_columns].astype(float).std(axis=0).replace(0.0, 1.0).fillna(1.0)
        )
        self.standardizer = {
            "feature_columns": list(self.feature_columns),
            "mean": means.to_dict(),
            "std": stds.to_dict(),
        }

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_columns: list[str],
        label_column: str = "label",
    ) -> MLPRankerWrapper:
        """Fit the MLP ranker."""
        seed = int(self.config.get("seed", 42))
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.feature_columns = list(feature_columns)
        self._fit_standardizer(train_df)
        train_transformed = apply_feature_standardizer(train_df, self.standardizer)
        training_config = self.config.get("training", {})
        model_config = self.config.get("model", {})
        self.device = torch.device(str(training_config.get("device", "cpu")))
        self.model = MLPRanker(
            input_dim=len(self.feature_columns),
            hidden_dims=[int(value) for value in model_config.get("hidden_dims", [64, 32])],
            dropout=float(model_config.get("dropout", 0.1)),
        ).to(self.device)
        dataset = RankingFeatureDataset(train_transformed, self.feature_columns, label_column)
        loader = DataLoader(
            dataset,
            batch_size=int(training_config.get("batch_size", 64)),
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
        for _ in range(int(training_config.get("epochs", 5))):
            for batch in loader:
                features = batch["features"].to(self.device)
                labels = batch["label"].to(self.device)
                logits = self.model(features)
                loss = loss_fn(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict relevance scores."""
        if self.model is None:
            return np.zeros(len(df), dtype=float)
        transformed = apply_feature_standardizer(df, self.standardizer)
        features = torch.tensor(
            transformed[self.feature_columns].astype(float).to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        self.model.eval()
        with torch.no_grad():
            logits = self.model(features)
            return torch.sigmoid(logits).cpu().numpy().astype(float)

    def save(self, path: str | Path) -> None:
        """Save model checkpoint."""
        if self.model is None:
            raise ValueError("Cannot save an unfit MLP ranker")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "feature_columns": self.feature_columns,
                "standardizer": self.standardizer,
                "config": self.config,
                "backend": self.backend,
            },
            output,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> MLPRankerWrapper:
        """Load a saved MLP ranker."""
        checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
        wrapper = cls(checkpoint.get("config", {}))
        wrapper.feature_columns = list(checkpoint["feature_columns"])
        wrapper.standardizer = checkpoint["standardizer"]
        wrapper.device = torch.device(device)
        model_config = wrapper.config.get("model", {})
        wrapper.model = MLPRanker(
            input_dim=len(wrapper.feature_columns),
            hidden_dims=[int(value) for value in model_config.get("hidden_dims", [64, 32])],
            dropout=float(model_config.get("dropout", 0.1)),
        ).to(wrapper.device)
        wrapper.model.load_state_dict(checkpoint["model_state_dict"])
        return wrapper
