import numpy as np
import pandas as pd
import pytest

from bt4012_fraud.features import (
    add_history_features,
    ieee_history_features,
    ieee_transaction_features,
    merge_ieee_identity,
    provisional_ieee_entity,
    sparkov_history_features,
    sparkov_transaction_features,
)


def random_history_frame(seed: int = 0, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2020-01-01")
    # Minute resolution over ~10 days gives same-timestamp ties within entities.
    minutes = rng.integers(0, 60 * 24 * 10, size=n)
    minutes[:20] = minutes[0]
    frame = pd.DataFrame(
        {
            "entity": rng.choice(["a", "b", "c", "d"], size=n),
            "time": start + pd.to_timedelta(minutes, unit="min"),
            "amount": rng.gamma(2.0, 30.0, size=n).round(2),
            "category": rng.choice(["food", "travel", "online"], size=n),
            "lat": rng.uniform(30, 40, size=n),
            "lon": rng.uniform(-120, -80, size=n),
        }
    )
    frame.loc[[5, 17], "entity"] = None
    # Shuffle and use a non-default index to check alignment.
    frame = frame.sample(frac=1.0, random_state=seed)
    frame.index = frame.index * 3 + 7
    return frame


def brute_force_windows(frame: pd.DataFrame, window: str) -> pd.DataFrame:
    span = pd.Timedelta(window)
    rows = {}
    for index, row in frame.iterrows():
        if pd.isna(row["entity"]):
            rows[index] = (np.nan, np.nan)
            continue
        same = frame[frame["entity"] == row["entity"]]
        past = same[(same["time"] < row["time"]) & (same["time"] >= row["time"] - span)]
        rows[index] = (len(past), past["amount"].sum())
    return pd.DataFrame.from_dict(rows, orient="index", columns=["count", "sum"])


@pytest.mark.parametrize("window", ["1h", "24h", "7d"])
def test_rolling_windows_match_brute_force_strictly_past(window: str) -> None:
    frame = random_history_frame()
    features = add_history_features(frame, "entity", "time", "amount", windows=[window])
    expected = brute_force_windows(frame, window).reindex(frame.index)

    np.testing.assert_allclose(features[f"txn_count_{window}"], expected["count"])
    np.testing.assert_allclose(features[f"amt_sum_{window}"], expected["sum"], rtol=1e-9)


def test_current_transaction_never_in_its_own_history() -> None:
    frame = random_history_frame(seed=1)
    base = add_history_features(frame, "entity", "time", "amount", category_col="category")

    # Changing the current amount must not change that row's own history features.
    changed = frame.copy()
    target = changed.index[50]
    changed.loc[target, "amount"] = 1_000_000.0
    shifted = add_history_features(changed, "entity", "time", "amount", category_col="category")

    history_columns = [
        c for c in base.columns if c not in {"amt_to_prior_mean", "amt_prior_zscore"}
    ]
    pd.testing.assert_series_equal(
        base.loc[target, history_columns], shifted.loc[target, history_columns]
    )


def test_future_transactions_do_not_change_past_features() -> None:
    frame = random_history_frame(seed=2)
    cutoff = frame["time"].quantile(0.6)
    past_only = frame[frame["time"] <= cutoff]

    full = add_history_features(frame, "entity", "time", "amount", category_col="category")
    truncated = add_history_features(
        past_only, "entity", "time", "amount", category_col="category"
    )
    pd.testing.assert_frame_equal(full.loc[past_only.index], truncated)


def test_cumulative_features_and_first_transaction() -> None:
    frame = pd.DataFrame(
        {
            "entity": ["x", "x", "x", "y"],
            "time": pd.to_datetime(
                ["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 03:00", "2020-01-01 02:00"]
            ),
            "amount": [10.0, 30.0, 40.0, 5.0],
            "category": ["food", "food", "travel", "food"],
        }
    )
    features = add_history_features(frame, "entity", "time", "amount", category_col="category")

    assert features["prior_txn_count"].tolist() == [0, 1, 2, 0]
    assert features["is_first_txn"].tolist() == [1, 0, 0, 1]
    assert features["secs_since_prev"].iloc[1] == 3600
    assert features["secs_since_prev"].iloc[2] == 7200
    assert np.isnan(features["secs_since_prev"].iloc[0])
    assert features["amt_to_prior_mean"].iloc[2] == pytest.approx(40.0 / 20.0)
    assert features["amt_prior_zscore"].iloc[2] == pytest.approx(
        (40.0 - 20.0) / np.std([10.0, 30.0], ddof=1)
    )
    assert features["is_new_category"].tolist() == [1, 0, 1, 1]


def test_missing_entity_gets_missing_history() -> None:
    frame = random_history_frame(seed=3)
    features = add_history_features(frame, "entity", "time", "amount")
    missing = frame["entity"].isna()
    assert features.loc[missing].isna().all(axis=None)
    assert features.loc[~missing, "prior_txn_count"].notna().all()


def test_numeric_elapsed_seconds_are_supported() -> None:
    frame = pd.DataFrame(
        {"entity": ["x", "x", "x"], "time": [0, 1800, 90_000], "amount": [1.0, 2.0, 3.0]}
    )
    features = add_history_features(frame, "entity", "time", "amount", windows=["1h", "24h"])
    assert features["txn_count_1h"].tolist() == [0, 1, 0]
    # 90,000 s minus 24 h is 3,600 s, so neither earlier transaction is in that window.
    assert features["txn_count_24h"].tolist() == [0, 1, 0]


def sparkov_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trans_date_trans_time": [
                "2020-01-04 23:30:00",
                "2020-01-05 00:10:00",
                "2020-01-06 12:00:00",
            ],
            "cc_num": [111, 111, 222],
            "amt": [20.0, 500.0, 15.0],
            "category": ["grocery_pos", "shopping_net", "grocery_pos"],
            "gender": ["F", "F", "M"],
            "state": ["CA", "CA", "NY"],
            "city_pop": [1000, 1000, 50],
            "dob": ["1990-01-01", "1990-01-01", "1980-06-01"],
            "lat": [34.0, 34.0, 40.0],
            "long": [-118.0, -118.0, -74.0],
            "merch_lat": [34.1, 40.7, 40.1],
            "merch_long": [-118.1, -74.0, -74.1],
            "first": ["A", "A", "B"],
            "last": ["Z", "Z", "Y"],
            "street": ["s", "s", "t"],
            "trans_num": ["t1", "t2", "t3"],
            "is_fraud": [0, 1, 0],
        }
    )


def test_sparkov_features_exclude_identifiers_and_labels() -> None:
    frame = sparkov_sample()
    transaction = sparkov_transaction_features(frame)
    history = sparkov_history_features(frame)
    columns = set(transaction.columns) | set(history.columns)

    assert not {"cc_num", "first", "last", "street", "trans_num", "is_fraud"} & columns
    assert transaction["is_weekend"].tolist() == [1, 1, 0]
    assert transaction["is_night"].tolist() == [1, 1, 0]
    assert history["km_from_prev_txn"].iloc[1] > 3000
    assert history["implied_speed_kmh"].iloc[1] > 3000
    assert history["prior_txn_count"].tolist() == [0, 1, 0]


def test_ieee_features_and_provisional_entity() -> None:
    transaction = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [86_400, 90_000, 172_800, 180_000],
            "TransactionAmt": [49.95, 10.0, 75.5, 20.0],
            "ProductCD": ["W", "C", "W", "W"],
            "card1": [100, 100, 100, 200],
            "addr1": [300.0, 300.0, 300.0, np.nan],
            "card4": ["visa", "visa", "visa", "mastercard"],
            "card6": ["debit", "debit", "debit", "credit"],
            "isFraud": [0, 1, 0, 0],
        }
    )
    identity = pd.DataFrame({"TransactionID": [2], "DeviceType": ["mobile"]})
    merged = merge_ieee_identity(transaction, identity)

    features = ieee_transaction_features(merged)
    assert features["has_identity"].tolist() == [0, 1, 0, 0]
    assert features["amt_cents"].iloc[0] == pytest.approx(0.95)
    assert features["hour"].tolist() == [0, 1, 0, 2]
    assert "isFraud" not in features

    entity = provisional_ieee_entity(merged, ["card1", "addr1"])
    assert entity.iloc[0] == entity.iloc[1] == entity.iloc[2]
    assert pd.isna(entity.iloc[3])

    history = ieee_history_features(merged, entity)
    assert history["prior_txn_count"].iloc[:3].tolist() == [0, 1, 2]
    # Windows are [t - 24h, t): a transaction exactly 24 h earlier is included.
    assert history["txn_count_24h"].iloc[:3].tolist() == [0, 1, 2]
    assert history.iloc[3].isna().all()
