import pandas as pd

from bt4012_fraud.data import prepare_sparkov_frame, temporal_train_validation_split


def sample_frame() -> pd.DataFrame:
    rows = []
    for index in range(10):
        rows.append(
            {
                "trans_date_trans_time": f"2020-01-{index + 1:02d} 12:00:00",
                "is_fraud": int(index in {3, 9}),
                "amt": 10.0 + index,
                "category": "grocery_pos",
                "gender": "F",
                "state": "CA",
                "city_pop": 1000,
                "dob": "1990-01-01",
                "lat": 34.0,
                "long": -118.0,
                "merch_lat": 34.1,
                "merch_long": -118.1,
                "cc_num": 1234,
                "first": "Synthetic",
                "last": "Person",
                "street": "Example Street",
                "trans_num": f"transaction-{index}",
                "unix_time": index,
            }
        )
    return pd.DataFrame(rows[::-1])


def test_features_exclude_direct_identifiers() -> None:
    X, _, _ = prepare_sparkov_frame(sample_frame())
    assert not {"cc_num", "first", "last", "street", "trans_num"}.intersection(X.columns)
    assert {"age", "hour", "day_of_week", "distance_km"}.issubset(X.columns)


def test_temporal_split_keeps_future_in_validation() -> None:
    X, y, timestamps = prepare_sparkov_frame(sample_frame())
    split = temporal_train_validation_split(X, y, timestamps, validation_fraction=0.2)
    assert split.train_end <= split.validation_start
    assert len(split.X_train) == 8
    assert len(split.X_validation) == 2
