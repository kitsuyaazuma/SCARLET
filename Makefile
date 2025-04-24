format:
	uv run ruff format .

lint:
	uv run ruff check . --fix
	uv run mypy . --ignore-missing-imports --check-untyped-defs

visualize:
	uv run tensorboard --logdir=./outputs
