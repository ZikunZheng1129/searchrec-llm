.PHONY: setup test lint format api dashboard demo clean

PYTHON ?= python3

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest tests/

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

api:
	bash scripts/launch_api.sh

dashboard:
	bash scripts/launch_dashboard.sh

demo:
	bash scripts/run_demo_stack.sh

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
