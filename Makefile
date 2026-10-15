.PHONY: setup download test lint baseline features

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

features:
	python -m bt4012_fraud.feature_ablation sparkov --train-csv data/raw/fraudTrain.csv --output-dir artifacts/feature_ablation/sparkov
