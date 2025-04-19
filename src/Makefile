format:
	poetry run ruff format ../src

lint:
	poetry run ruff check . --fix
	poetry run mypy . --cache-dir=../.mypy_cache --config-file=../pyproject.toml

visualize:
	poetry run tensorboard --logdir=./outputs --samples_per_plugin=scalars=5000
