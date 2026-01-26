format:
	uv run ruff format .

lint:
	uv run ruff check . --fix --preview

type-check:
	uv run mypy .

test:
	uv run pytest

check: format lint type-check test

sync:
	uv run wandb sync --sync-all
