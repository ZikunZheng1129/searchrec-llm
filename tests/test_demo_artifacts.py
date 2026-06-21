from __future__ import annotations

import py_compile
from pathlib import Path

from app.api.main import create_app
from src.utils.config import load_yaml_config


def test_app_configs_scripts_docs_and_docker_exist():
    for path in [
        "configs/app/api_debug.yaml",
        "configs/app/dashboard_debug.yaml",
        "scripts/launch_api.sh",
        "scripts/launch_dashboard.sh",
        "scripts/run_demo_stack.sh",
        "docker/Dockerfile",
        "docker/docker-compose.yml",
        "docs/demo_guide.md",
        "docs/api_reference.md",
        "docs/dashboard_guide.md",
    ]:
        assert Path(path).exists(), path
    assert load_yaml_config("configs/app/api_debug.yaml")["app"]["version"] == "0.11.0"


def test_streamlit_pages_are_syntactically_valid_and_api_factory_works():
    paths = [
        Path("app/dashboard/streamlit_app.py"),
        *sorted(Path("app/dashboard/pages").glob("*.py")),
    ]
    for path in paths:
        py_compile.compile(str(path), doraise=True)
    app = create_app()
    assert app.title == "TikSearchRec-LLM API"
