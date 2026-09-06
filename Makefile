.PHONY: setup download test lint baseline

setup:
	python -m pip install -e ".[dev]"

download:
	bash scripts/download_sparkov.sh

test:
	pytest -q

lint:
	ruff check src tests

baseline:
	python -m bt4012_fraud.baseline --train-csv data/raw/fraudTrain.csv --output-dir artifacts/baseline
