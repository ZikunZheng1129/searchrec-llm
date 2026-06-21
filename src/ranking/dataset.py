"""Dataset helpers for ranking models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

REQUIRED_RANKING_COLUMNS = {
    "query_id",
    "query_text",
    "split",
    "target_item_id",
    "candidate_item_id",
    "label",
}

NON_FEATURE_COLUMNS = {
    "query_id",
    "query_text",
    "split",
    "target_item_id",
    "candidate_item_id",
    "candidate_source",
    "label",
    "price_bucket",
    "diversity_category",
    "category",
    "brand",
}


def validate_ranking_candidates(df: pd.DataFrame) -> None:
    """Validate the minimum ranking candidate schema."""
    missing = sorted(REQUIRED_RANKING_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Ranking candidates missing required columns: {missing}")
    if df.empty:
        raise ValueError("Ranking candidates are empty")
    duplicate_mask = df.duplicated(["query_id", "candidate_item_id"])
    if duplicate_mask.any():
        raise ValueError("Ranking candidates contain duplicate candidate_item_id per query_id")


def get_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return stable numeric feature columns for feature-based rankers."""
    columns = []
    for column in sorted(df.columns):
        if column in NON_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            columns.append(column)
    return columns


def split_ranking_candidates(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Filter ranking candidates by split."""
    validate_ranking_candidates(df)
    return df[df["split"].astype(str) == str(split)].copy().reset_index(drop=True)


def _feature_stats(train_df: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    means = train_df[feature_columns].astype(float).mean(axis=0)
    stds = train_df[feature_columns].astype(float).std(axis=0).replace(0.0, 1.0).fillna(1.0)
    return {
        "feature_columns": list(feature_columns),
        "mean": means.to_dict(),
        "std": stds.to_dict(),
    }


def apply_feature_standardizer(df: pd.DataFrame, standardizer: dict[str, Any]) -> pd.DataFrame:
    """Apply saved z-score feature scaling to a candidate DataFrame."""
    output = df.copy()
    for column in standardizer.get("feature_columns", []):
        mean = float(standardizer.get("mean", {}).get(column, 0.0))
        std = float(standardizer.get("std", {}).get(column, 1.0)) or 1.0
        output[column] = (pd.to_numeric(output[column], errors="coerce").fillna(0.0) - mean) / std
    return output


def standardize_features(
    train_df: pd.DataFrame,
    other_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit a standardizer on train_df and apply it to other_df."""
    feature_columns = get_numeric_feature_columns(train_df)
    standardizer = _feature_stats(train_df, feature_columns)
    return apply_feature_standardizer(other_df, standardizer), standardizer


class RankingFeatureDataset(Dataset):
    """Torch dataset for numeric ranking features."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        label_column: str = "label",
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.feature_columns = list(feature_columns)
        self.label_column = label_column
        self.features = self.df[self.feature_columns].astype(float).to_numpy(dtype=np.float32)
        self.labels = self.df[self.label_column].astype(float).to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        return {
            "features": torch.tensor(self.features[index], dtype=torch.float32),
            "label": torch.tensor(self.labels[index], dtype=torch.float32),
            "query_id": str(row["query_id"]),
            "candidate_item_id": str(row["candidate_item_id"]),
        }
