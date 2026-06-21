from pathlib import Path

import pandas as pd
import yaml

from src.evaluation.report import generate_validation_artifacts


def _write_test_config(tmp_path: Path) -> Path:
    config_dir = tmp_path / "validation" / "experiments"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "stage5_debug_validation.yaml"
    config = {
        "seed": 42,
        "inputs": {
            "retrieval_results_path": "validation/results/retrieval_results.csv",
            "recommender_results_path": "validation/results/recommender_baselines.csv",
        },
        "optional_inputs": {
            "sequence_results_path": "validation/results/sequence_results.csv",
            "ranking_results_path": "validation/results/ranking_results.csv",
        },
        "outputs": {
            "final_leaderboard_path": "validation/results/final_leaderboard.csv",
            "final_model_selection_report_path": (
                "validation/reports/final_model_selection_report.md"
            ),
            "latency_quality_report_path": "validation/reports/latency_quality_tradeoff.md",
            "validation_summary_path": "validation/reports/validation_summary.md",
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _write_fake_results(tmp_path: Path) -> None:
    results_dir = tmp_path / "validation" / "results"
    results_dir.mkdir(parents=True)
    retrieval = pd.DataFrame(
        [
            {
                "method": "bm25",
                "split": "test",
                "num_queries": 5,
                "recall_at_50": 1.0,
                "mrr_at_10": 0.5,
                "query_coverage": 1.0,
                "avg_latency_ms": 1.0,
                "p95_latency_ms": 2.0,
                "index_backend": "local",
                "config_path": "configs/retrieval/bm25_debug.yaml",
            }
        ]
    )
    recommendation = pd.DataFrame(
        [
            {
                "method": "itemcf",
                "split": "test",
                "num_users": 6,
                "ndcg_at_10": 0.3,
                "recall_at_10": 0.2,
                "coverage_at_10": 0.8,
                "avg_latency_ms": 1.5,
                "p95_latency_ms": 2.5,
                "config_path": "configs/recommendation/itemcf_debug.yaml",
            }
        ]
    )
    retrieval.to_csv(results_dir / "retrieval_results.csv", index=False)
    recommendation.to_csv(results_dir / "recommender_baselines.csv", index=False)


def test_generate_validation_artifacts_writes_expected_files(tmp_path: Path) -> None:
    config_path = _write_test_config(tmp_path)
    _write_fake_results(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    artifacts = generate_validation_artifacts(config, config_path=config_path)

    assert Path(artifacts["final_leaderboard_path"]).exists()
    assert Path(artifacts["final_model_selection_report_path"]).exists()
    assert Path(artifacts["latency_quality_report_path"]).exists()
    assert Path(artifacts["validation_summary_path"]).exists()


def test_generated_reports_mention_synthetic_debug_caveat(tmp_path: Path) -> None:
    config_path = _write_test_config(tmp_path)
    _write_fake_results(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    artifacts = generate_validation_artifacts(config, config_path=config_path)
    report_text = Path(artifacts["final_model_selection_report_path"]).read_text(encoding="utf-8")

    assert "synthetic/local debug data" in report_text


def test_leaderboard_contains_no_fake_future_stages(tmp_path: Path) -> None:
    config_path = _write_test_config(tmp_path)
    _write_fake_results(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    artifacts = generate_validation_artifacts(config, config_path=config_path)
    leaderboard = pd.read_csv(artifacts["final_leaderboard_path"])

    assert set(leaderboard["stage"]) == {"retrieval", "recommendation"}
    assert len(leaderboard) == 2
