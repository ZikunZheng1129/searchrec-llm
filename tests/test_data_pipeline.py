from pathlib import Path

import pandas as pd

from src.data.dataset import INTERACTION_COLUMNS, ITEM_COLUMNS, USER_COLUMNS, attach_split_column
from src.data.negative_sampling import sample_negative_items
from src.data.preprocess import (
    generate_synthetic_interactions,
    generate_synthetic_items,
    generate_synthetic_users,
)
from src.data.split import build_user_sequences, leave_one_out_split
from src.pipelines.build_dataset import run_build_dataset
from src.utils.config import resolve_project_path
from src.utils.io import read_parquet


def test_synthetic_generation_is_non_empty_valid_and_deterministic() -> None:
    categories = ["Electronics", "Beauty"]
    users_first = generate_synthetic_users(num_users=5, seed=7)
    items_first = generate_synthetic_items(num_items=8, categories=categories, seed=7)
    interactions_first = generate_synthetic_interactions(
        users=users_first,
        items=items_first,
        num_interactions=30,
        start_timestamp="2024-01-01",
        end_timestamp="2024-01-31",
        seed=7,
    )

    users_second = generate_synthetic_users(num_users=5, seed=7)
    items_second = generate_synthetic_items(num_items=8, categories=categories, seed=7)
    interactions_second = generate_synthetic_interactions(
        users=users_second,
        items=items_second,
        num_interactions=30,
        start_timestamp="2024-01-01",
        end_timestamp="2024-01-31",
        seed=7,
    )

    assert not users_first.empty
    assert not items_first.empty
    assert not interactions_first.empty
    assert set(USER_COLUMNS).issubset(users_first.columns)
    assert set(ITEM_COLUMNS).issubset(items_first.columns)
    assert set(INTERACTION_COLUMNS).issubset(interactions_first.columns)
    pd.testing.assert_frame_equal(users_first, users_second)
    pd.testing.assert_frame_equal(items_first, items_second)
    pd.testing.assert_frame_equal(interactions_first, interactions_second)


def _sample_interactions() -> pd.DataFrame:
    rows = []
    for user_index in range(2):
        user_id = f"user_{user_index}"
        for event_index in range(4):
            rows.append(
                {
                    "row_id": f"{user_id}_{event_index}",
                    "user_id": user_id,
                    "item_id": f"item_{event_index}",
                    "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(days=event_index),
                    "event_type": "click",
                    "rating": 3.0,
                    "event_weight": 2.0,
                }
            )
    return pd.DataFrame(rows)


def test_leave_one_out_split_is_time_aware_and_disjoint() -> None:
    interactions = _sample_interactions()

    train, val, test = leave_one_out_split(interactions)

    assert not train.empty
    assert not val.empty
    assert not test.empty
    assert set(train["row_id"]).isdisjoint(set(val["row_id"]))
    assert set(train["row_id"]).isdisjoint(set(test["row_id"]))
    assert set(val["row_id"]).isdisjoint(set(test["row_id"]))

    for user_id, original_group in interactions.groupby("user_id"):
        if len(original_group) < 3:
            continue
        train_ts = train.loc[train["user_id"] == user_id, "timestamp"]
        val_ts = val.loc[val["user_id"] == user_id, "timestamp"].iloc[0]
        test_ts = test.loc[test["user_id"] == user_id, "timestamp"].iloc[0]

        assert test_ts > val_ts
        assert val_ts > train_ts.max()


def test_user_sequences_match_lengths_and_are_ordered() -> None:
    interactions = _sample_interactions().sample(frac=1.0, random_state=11).reset_index(drop=True)

    sequences = build_user_sequences(interactions)

    assert not sequences.empty
    for _, row in sequences.iterrows():
        assert row["sequence_length"] == len(row["item_sequence"])
        timestamps = pd.to_datetime(row["timestamp_sequence"])
        assert timestamps.is_monotonic_increasing


def test_negative_sampling_excludes_user_interacted_items() -> None:
    train, val, test = leave_one_out_split(_sample_interactions())
    interactions = attach_split_column(train, val, test)
    all_item_ids = [f"item_{index}" for index in range(8)]

    negatives = sample_negative_items(
        interactions=interactions,
        all_item_ids=all_item_ids,
        num_negatives_per_positive=3,
        seed=99,
    )

    assert {"user_id", "positive_item_id", "negative_item_id", "split"} == set(negatives.columns)
    interacted_by_user = interactions.groupby("user_id")["item_id"].apply(set).to_dict()
    for _, row in negatives.iterrows():
        assert row["negative_item_id"] not in interacted_by_user[row["user_id"]]


def test_build_dataset_pipeline_writes_expected_parquet_files() -> None:
    config_path = resolve_project_path("configs/data/debug_sample.yaml")
    summary = run_build_dataset(config_path)
    expected_names = {
        "train",
        "val",
        "test",
        "item_metadata",
        "user_sequences",
        "negative_samples",
    }

    assert set(summary["output_paths"]) == expected_names
    for output_path in summary["output_paths"].values():
        path = Path(output_path)
        assert path.exists()
        assert not read_parquet(path).empty
