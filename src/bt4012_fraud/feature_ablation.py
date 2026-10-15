"""Compare transaction-only and card-history feature sets under chronological validation.

This is the feature-engineering ablation requested in the TA feedback: does adding
history features help, and by how much? It trains a Dummy reference and a class-weighted
Logistic Regression (the linear baseline) on each feature set, using identical splits and
metrics. The final test data are never touched:

* Sparkov uses only ``fraudTrain.csv``; ``fraudTest.csv`` is not read.
* IEEE-CIS holds out the latest ``--holdout-fraction`` of the labelled training file
  before anything else, and that block is never scored here.

History features are computed on the full input before splitting. This is safe because
each value only uses strictly earlier transactions and no labels, which matches what
would be available in production at scoring time.

Example::

    python -m bt4012_fraud.feature_ablation sparkov \\
        --train-csv data/raw/fraudTrain.csv --output-dir artifacts/feature_ablation/sparkov
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from bt4012_fraud.data import temporal_train_validation_split
from bt4012_fraud.evaluation import evaluate_binary_predictions
from bt4012_fraud.features import (
    IEEE_ANONYMISED_CATEGORICAL,
    IEEE_TRANSACTION_CATEGORICAL,
    ieee_history_features,
    ieee_transaction_features,
    merge_ieee_identity,
    provisional_ieee_entity,
    sparkov_history_features,
    sparkov_transaction_features,
)

RANDOM_STATE = 4012
# Heavy-tailed history features are log-transformed before scaling.
LOG_PREFIXES = ("secs_since_prev", "prior_txn_count", "txn_count_", "amt_sum_", "amt_mean_",
                "km_from_prev_txn", "implied_speed_kmh", "amt_to_prior_mean")


def _signed_log1p(values):
    return np.sign(values) * np.log1p(np.abs(values))


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    log_columns = [c for c in numeric if c.startswith(LOG_PREFIXES)]
    plain_columns = [c for c in numeric if c not in log_columns]
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ]
    transformers = []
    if plain_columns:
        transformers.append(("numeric", Pipeline(numeric_steps), plain_columns))
    if log_columns:
        transformers.append(
            (
                "numeric_log",
                Pipeline([("log", FunctionTransformer(_signed_log1p)), *numeric_steps]),
                log_columns,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers)


def build_models(numeric: list[str], categorical: list[str]) -> dict[str, Pipeline]:
    return {
        "dummy_prior": Pipeline(
            [
                ("preprocess", build_preprocessor(numeric, categorical)),
                ("model", DummyClassifier(strategy="prior")),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("preprocess", build_preprocessor(numeric, categorical)),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
    }


def _split_columns(frame: pd.DataFrame, categorical_pool: set[str]) -> tuple[list[str], list[str]]:
    categorical = [c for c in frame.columns if c in categorical_pool]
    numeric = [c for c in frame.columns if c not in categorical_pool]
    return numeric, categorical


def load_sparkov_feature_sets(train_csv: Path) -> tuple[dict[str, pd.DataFrame], pd.Series,
                                                        pd.Series, dict[str, object]]:
    raw = pd.read_csv(train_csv)
    transaction = sparkov_transaction_features(raw)
    history = sparkov_history_features(raw)
    sets = {
        "transaction_only": transaction,
        "transaction_plus_history": pd.concat([transaction, history], axis=1),
    }
    y = raw["is_fraud"].astype(int)
    timestamps = pd.to_datetime(raw["trans_date_trans_time"]).rename("timestamp")
    info = {"dataset": "sparkov", "source": str(train_csv), "entity": "cc_num"}
    return sets, y, timestamps, info


def load_ieee_feature_sets(
    transaction_csv: Path,
    identity_csv: Path | None,
    entity_columns: list[str],
    holdout_fraction: float,
    include_anonymised: bool,
) -> tuple[dict[str, pd.DataFrame], pd.Series, pd.Series, dict[str, object]]:
    transaction = pd.read_csv(transaction_csv)
    identity = pd.read_csv(identity_csv) if identity_csv else None
    merged = merge_ieee_identity(transaction, identity)

    # History is computed on the full file (past-only), then the held-out block is dropped.
    entity = provisional_ieee_entity(merged, entity_columns)
    base = ieee_transaction_features(merged, include_anonymised=include_anonymised)
    history = ieee_history_features(merged, entity)

    order = np.argsort(merged["TransactionDT"].to_numpy(), kind="stable")
    keep = np.sort(order[: int(len(order) * (1 - holdout_fraction))])
    holdout_start = int(merged["TransactionDT"].iloc[order[len(keep)]]) if holdout_fraction else None

    sets = {
        "transaction_only": base.iloc[keep],
        "transaction_plus_history": pd.concat([base, history], axis=1).iloc[keep],
    }
    y = merged["isFraud"].astype(int).iloc[keep]
    timestamps = pd.to_datetime(merged["TransactionDT"], unit="s").iloc[keep].rename("timestamp")
    info = {
        "dataset": "ieee_cis",
        "source": str(transaction_csv),
        "identity_source": str(identity_csv) if identity_csv else None,
        "entity": "provisional:" + "+".join(entity_columns),
        "entity_coverage": float(entity.notna().mean()),
        "include_anonymised": include_anonymised,
        "holdout_fraction": holdout_fraction,
        "holdout_starts_at_transactiondt": holdout_start,
    }
    return sets, y, timestamps, info


def run_ablation(
    feature_sets: dict[str, pd.DataFrame],
    y: pd.Series,
    timestamps: pd.Series,
    info: dict[str, object],
    categorical_pool: set[str],
    output_dir: Path,
    validation_fraction: float,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {"info": info, "random_state": RANDOM_STATE, "feature_sets": {}}

    for set_name, features in feature_sets.items():
        features = features.reset_index(drop=True)
        split = temporal_train_validation_split(
            features, y.reset_index(drop=True), timestamps.reset_index(drop=True),
            validation_fraction,
        )
        numeric, categorical = _split_columns(features, categorical_pool)
        set_result: dict[str, object] = {
            "numeric_features": numeric,
            "categorical_features": categorical,
            "split": {
                "train_rows": len(split.X_train),
                "validation_rows": len(split.X_validation),
                "train_end": split.train_end.isoformat(),
                "validation_start": split.validation_start.isoformat(),
                "train_fraud_rate": float(split.y_train.mean()),
                "validation_fraud_rate": float(split.y_validation.mean()),
            },
            "models": {},
        }
        for model_name, pipeline in build_models(numeric, categorical).items():
            started = time.perf_counter()
            pipeline.fit(split.X_train, split.y_train)
            probabilities = pipeline.predict_proba(split.X_validation)[:, 1]
            metrics = evaluate_binary_predictions(split.y_validation.to_numpy(), probabilities)
            metrics["fit_and_score_seconds"] = round(time.perf_counter() - started, 1)
            set_result["models"][model_name] = metrics
            print(f"{info['dataset']} | {set_name} | {model_name}: "
                  f"PR-AUC={metrics['pr_auc']:.4f} ROC-AUC={metrics['roc_auc']:.4f}", flush=True)
        results["feature_sets"][set_name] = set_result

    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_dir / "results.md").write_text(results_table(results), encoding="utf-8")
    return results


def results_table(results: dict[str, object]) -> str:
    info = results["info"]
    lines = [
        f"# Feature ablation: {info['dataset']}",
        "",
        (
            f"Entity for history features: `{info['entity']}`. Chronological validation; "
            "threshold 0.5 for precision, recall and F1."
        ),
        "",
        "| Feature set | Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | TP | FP | FN |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for set_name, set_result in results["feature_sets"].items():
        for model_name, m in set_result["models"].items():
            lines.append(
                f"| {set_name} | {model_name} | {m['pr_auc']:.4f} | {m['roc_auc']:.4f} | "
                f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
                f"{m['true_positives']} | {m['false_positives']} | {m['false_negatives']} |"
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="dataset", required=True)

    sparkov = sub.add_parser("sparkov", help="Sparkov fraudTrain.csv only")
    sparkov.add_argument("--train-csv", type=Path, required=True)

    ieee = sub.add_parser("ieee", help="IEEE-CIS labelled training files")
    ieee.add_argument("--transaction-csv", type=Path, required=True)
    ieee.add_argument("--identity-csv", type=Path, default=None)
    ieee.add_argument(
        "--entity-columns", default="card1,addr1",
        help="Provisional entity key until the team's card reconstruction is merged",
    )
    ieee.add_argument("--holdout-fraction", type=float, default=0.20)
    ieee.add_argument("--include-anonymised", action="store_true")

    for p in (sparkov, ieee):
        p.add_argument("--output-dir", type=Path, required=True)
        p.add_argument("--validation-fraction", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "sparkov":
        sets, y, timestamps, info = load_sparkov_feature_sets(args.train_csv)
        categorical_pool = {"category", "gender", "state"}
    else:
        sets, y, timestamps, info = load_ieee_feature_sets(
            args.transaction_csv, args.identity_csv, args.entity_columns.split(","),
            args.holdout_fraction, args.include_anonymised,
        )
        categorical_pool = set(IEEE_TRANSACTION_CATEGORICAL) | set(IEEE_ANONYMISED_CATEGORICAL)
    results = run_ablation(sets, y, timestamps, info, categorical_pool, args.output_dir,
                           args.validation_fraction)
    print(results_table(results))


if __name__ == "__main__":
    main()
