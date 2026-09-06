"""Train reproducible Dummy and Logistic Regression baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bt4012_fraud.data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    load_sparkov,
    temporal_train_validation_split,
)
from bt4012_fraud.evaluation import evaluate_binary_predictions

RANDOM_STATE = 4012


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )


def build_models() -> dict[str, Pipeline]:
    return {
        "dummy_prior": Pipeline(
            [("preprocess", build_preprocessor()), ("model", DummyClassifier(strategy="prior"))]
        ),
        "logistic_regression": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=300,
                        random_state=RANDOM_STATE,
                        solver="saga",
                    ),
                ),
            ]
        ),
    }


def run_baselines(train_csv: Path, output_dir: Path, validation_fraction: float) -> None:
    X, y, timestamps = load_sparkov(train_csv)
    split = temporal_train_validation_split(X, y, timestamps, validation_fraction)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, object] = {
        "data": {
            "train_rows": len(split.X_train),
            "validation_rows": len(split.X_validation),
            "train_end": split.train_end.isoformat(),
            "validation_start": split.validation_start.isoformat(),
            "train_fraud_rate": float(split.y_train.mean()),
            "validation_fraud_rate": float(split.y_validation.mean()),
        },
        "models": {},
    }

    for name, pipeline in build_models().items():
        pipeline.fit(split.X_train, split.y_train)
        probabilities = pipeline.predict_proba(split.X_validation)[:, 1]
        results["models"][name] = evaluate_binary_predictions(
            split.y_validation.to_numpy(), probabilities
        )
        joblib.dump(pipeline, output_dir / f"{name}.joblib")

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(json.dumps(results, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_baselines(args.train_csv, args.output_dir, args.validation_fraction)


if __name__ == "__main__":
    main()
