#!/usr/bin/env bash
set -euo pipefail

command -v kaggle >/dev/null 2>&1 || {
  echo "Kaggle CLI is missing. Run: python -m pip install -e '.[dev]'"
  exit 1
}

mkdir -p data/raw
kaggle datasets download -d kartik2112/fraud-detection -p data/raw --unzip

test -f data/raw/fraudTrain.csv
test -f data/raw/fraudTest.csv
echo "Sparkov train and test files are available under data/raw/."
