from __future__ import annotations

from pathlib import Path

REQUIRED_DOCS = [
    "README.md",
    "docs/project_overview.md",
    "docs/system_design.md",
    "docs/model_design.md",
    "docs/evaluation_plan.md",
    "docs/final_report.md",
    "docs/final_results_summary.md",
    "docs/project_completion_audit.md",
    "docs/limitations_and_future_work.md",
    "docs/demo_script.md",
    "docs/colab_training_plan.md",
    "docs/github_portfolio_checklist.md",
    "docs/resume_bullets.md",
    "docs/interview_talking_points.md",
    "docs/interview_q_and_a.md",
    "docs/project_pitch.md",
    "docs/recruiter_summary.md",
    "docs/linkedin_project_description.md",
]


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_stage12_required_docs_exist_and_are_nonempty():
    for path in REQUIRED_DOCS:
        doc_path = Path(path)
        assert doc_path.exists(), path
        assert doc_path.read_text(encoding="utf-8").strip(), path


def test_important_docs_include_synthetic_mock_and_not_production_caveats():
    important_docs = [
        "README.md",
        "docs/project_overview.md",
        "docs/final_report.md",
        "docs/final_results_summary.md",
    ]
    for path in important_docs:
        text = _read(path).lower()
        assert "synthetic" in text, path
        assert "mock" in text, path
        assert "not production" in text, path


def test_resume_bullets_avoid_forbidden_overclaims():
    text = _read("docs/resume_bullets.md").lower()
    forbidden_phrases = [
        "production deployed",
        "real tiktok data",
        "gmv lift",
        "increased revenue",
        "serving millions",
    ]
    for phrase in forbidden_phrases:
        assert phrase not in text


def test_final_report_covers_core_system_topics():
    text = _read("docs/final_report.md").lower()
    for keyword in [
        "retrieval",
        "recommendation",
        "ranking",
        "llm",
        "genrec",
        "validation",
        "api",
        "dashboard",
    ]:
        assert keyword in text


def test_final_results_summary_uses_current_validation_values():
    text = _read("docs/final_results_summary.md")
    for value in [
        "0.228785",
        "0.055705",
        "0.029743",
        "0.299915",
        "0.292996",
        "0.245409",
        "0.100000",
    ]:
        assert value in text


def test_package_check_script_exists():
    script = Path("scripts/check_project_package.sh")
    assert script.exists()
    assert "pytest tests/" in script.read_text(encoding="utf-8")
