"""Sparkov data loading and leakage-conscious baseline features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "is_fraud"
TIMESTAMP = "trans_date_trans_time"
REQUIRED_COLUMNS = {
    TIMESTAMP,
    TARGET,
    "amt",
    "category",
    "gender",
    "state",
    "city_pop",
    "dob",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
}
DIRECT_IDENTIFIERS = {
    "cc_num",
    "first",
    "last",
    "street",
    "trans_num",
    "unix_time",
}
NUMERIC_FEATURES = ["amt", "city_pop", "age", "hour", "day_of_week", "distance_km"]
CATEGORICAL_FEATURES = ["category", "gender", "state"]


@dataclass(frozen=True)
class TemporalSplit:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    train_end: pd.Timestamp
    validation_start: pd.Timestamp


def _haversine_km(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> pd.Series:
    """Return great-circle distance between paired coordinates."""
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    a = np.sin(delta_lat / 2) ** 2 + (
        np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    )
    return pd.Series(6371.0 * 2 * np.arcsin(np.sqrt(a)), index=lat1.index)


def prepare_sparkov_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Validate Sparkov data and create interpretable transaction-level features."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Sparkov columns missing: {sorted(missing)}")

    data = frame.copy()
    timestamps = pd.to_datetime(data[TIMESTAMP], errors="raise")
    birth_dates = pd.to_datetime(data["dob"], errors="coerce")

    data["age"] = (timestamps - birth_dates).dt.days / 365.25
    data["hour"] = timestamps.dt.hour
    data["day_of_week"] = timestamps.dt.dayofweek
    data["distance_km"] = _haversine_km(
        data["lat"], data["long"], data["merch_lat"], data["merch_long"]
    )

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = data[feature_columns].copy()
    y = pd.to_numeric(data[TARGET], errors="raise").astype(int)

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("is_fraud must contain binary labels 0 and 1")
    if DIRECT_IDENTIFIERS.intersection(X.columns):
        raise AssertionError("Direct identifiers leaked into baseline features")

    return X, y, timestamps.rename("timestamp")


def load_sparkov(path: str | Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load a Sparkov CSV and return model features, label and timestamp."""
    return prepare_sparkov_frame(pd.read_csv(Path(path)))


def temporal_train_validation_split(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    validation_fraction: float = 0.20,
) -> TemporalSplit:
    """Split earlier transactions for training and later ones for validation."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not (len(X) == len(y) == len(timestamps)):
        raise ValueError("X, y and timestamps must have equal length")

    order = np.argsort(timestamps.to_numpy(), kind="stable")
    cutoff = int(len(order) * (1 - validation_fraction))
    if cutoff <= 0 or cutoff >= len(order):
        raise ValueError("Not enough rows for the requested split")

    train_idx, validation_idx = order[:cutoff], order[cutoff:]
    train_times = timestamps.iloc[train_idx]
    validation_times = timestamps.iloc[validation_idx]

    return TemporalSplit(
        X_train=X.iloc[train_idx].reset_index(drop=True),
        X_validation=X.iloc[validation_idx].reset_index(drop=True),
        y_train=y.iloc[train_idx].reset_index(drop=True),
        y_validation=y.iloc[validation_idx].reset_index(drop=True),
        train_end=train_times.max(),
        validation_start=validation_times.min(),
    )
