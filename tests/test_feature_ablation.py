import json

import numpy as np
import pandas as pd

from bt4012_fraud.feature_ablation import (
    load_ieee_feature_sets,
    load_sparkov_feature_sets,
    run_ablation,
)
from bt4012_fraud.features import IEEE_TRANSACTION_CATEGORICAL


def synthetic_sparkov(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    times = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 60 * 24 * 60, n)),
                                                         unit="min")
    fraud = (rng.random(n) < 0.08).astype(int)
    return pd.DataFrame(
        {
            "trans_date_trans_time": times.strftime("%Y-%m-%d %H:%M:%S"),
            "cc_num": rng.integers(1, 15, n),
            "amt": np.where(fraud == 1, rng.uniform(200, 900, n), rng.uniform(1, 120, n)).round(2),
            "category": rng.choice(["grocery_pos", "shopping_net", "gas_transport"], n),
            "gender": rng.choice(["F", "M"], n),
            "state": rng.choice(["CA", "NY", "TX"], n),
            "city_pop": rng.integers(100, 100_000, n),
            "dob": "1985-03-02",
            "lat": 34.0,
            "long": -118.0,
            "merch_lat": rng.uniform(33, 35, n),
            "merch_long": rng.uniform(-119, -117, n),
            "is_fraud": fraud,
        }
    )


def test_sparkov_ablation_smoke(tmp_path) -> None:
    csv = tmp_path / "fraudTrain.csv"
    synthetic_sparkov().to_csv(csv, index=False)
    sets, y, timestamps, info = load_sparkov_feature_sets(csv)
    out = tmp_path / "out"

    results = run_ablation(sets, y, timestamps, info, {"category", "gender", "state"}, out, 0.25)

    assert set(results["feature_sets"]) == {"transaction_only", "transaction_plus_history"}
    saved = json.loads((out / "metrics.json").read_text())
    history_features = saved["feature_sets"]["transaction_plus_history"]["numeric_features"]
    assert "txn_count_24h" in history_features
    assert "cc_num" not in history_features
    assert "| transaction_only | logistic_regression |" in (out / "results.md").read_text()


def test_ieee_ablation_drops_holdout_block(tmp_path) -> None:
    rng = np.random.default_rng(1)
    n = 300
    transaction = pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": np.sort(rng.integers(86_400, 86_400 * 40, n)),
            "TransactionAmt": rng.uniform(5, 300, n).round(2),
            "ProductCD": rng.choice(["W", "C"], n),
            "card1": rng.integers(1000, 1010, n),
            "addr1": rng.choice([100.0, 200.0, np.nan], n),
            "card4": rng.choice(["visa", "mastercard"], n),
            "card6": rng.choice(["debit", "credit"], n),
            "isFraud": (rng.random(n) < 0.1).astype(int),
        }
    )
    identity = pd.DataFrame({"TransactionID": np.arange(0, n, 4), "DeviceType": "desktop"})
    transaction_csv, identity_csv = tmp_path / "t.csv", tmp_path / "i.csv"
    transaction.to_csv(transaction_csv, index=False)
    identity.to_csv(identity_csv, index=False)

    sets, y, timestamps, info = load_ieee_feature_sets(
        transaction_csv, identity_csv, ["card1", "addr1"], holdout_fraction=0.2,
        include_anonymised=False,
    )
    assert len(y) == 240
    assert timestamps.max() < pd.to_datetime(info["holdout_starts_at_transactiondt"], unit="s")

    results = run_ablation(sets, y, timestamps, info, set(IEEE_TRANSACTION_CATEGORICAL),
                           tmp_path / "out", 0.25)
    assert results["info"]["entity"] == "provisional:card1+addr1"
