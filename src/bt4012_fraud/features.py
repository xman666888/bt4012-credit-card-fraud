"""Leakage-safe feature engineering for Sparkov and IEEE-CIS.

Two feature groups are produced so that the card-history ablation required by the TA
feedback can be run directly:

* **transaction-only** features describe the current transaction alone;
* **history** features summarise the same entity's *strictly earlier* transactions.

History features never use labels, so they remain valid in the label-delay experiment.
Entity identifiers (``cc_num`` for Sparkov, a reconstructed card ID for IEEE-CIS) are used
for grouping only and are never returned as model inputs.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from bt4012_fraud.data import _haversine_km

SECONDS_PER_DAY = 86_400
DEFAULT_WINDOWS: tuple[str, ...] = ("1h", "24h", "7d", "30d")

SPARKOV_TRANSACTION_NUMERIC = [
    "log_amt",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_night",
    "age",
    "distance_km",
    "log_city_pop",
]
SPARKOV_TRANSACTION_CATEGORICAL = ["category", "gender", "state"]

IEEE_TRANSACTION_NUMERIC = [
    "log_amt",
    "amt_cents",
    "hour",
    "day_of_week",
    "has_identity",
]
IEEE_TRANSACTION_CATEGORICAL = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
]
IEEE_ANONYMISED_NUMERIC = [f"C{i}" for i in range(1, 15)] + [f"D{i}" for i in range(1, 16)]
IEEE_ANONYMISED_CATEGORICAL = [f"M{i}" for i in range(1, 10)]


# ---------------------------------------------------------------------------
# Transaction-only features
# ---------------------------------------------------------------------------


def _as_text(values: pd.Series) -> pd.Series:
    """Categorical values as Python strings with ``np.nan`` for missing (sklearn-friendly)."""
    text = values.astype(object).map(str)
    return text.where(values.notna(), np.nan)


def sparkov_transaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return interpretable features computed from each Sparkov transaction alone."""
    timestamps = pd.to_datetime(frame["trans_date_trans_time"], errors="raise")
    birth_dates = pd.to_datetime(frame["dob"], errors="coerce")
    hour = timestamps.dt.hour

    features = pd.DataFrame(index=frame.index)
    features["log_amt"] = np.log1p(frame["amt"].clip(lower=0))
    features["hour"] = hour
    features["day_of_week"] = timestamps.dt.dayofweek
    features["is_weekend"] = (features["day_of_week"] >= 5).astype(int)
    features["is_night"] = ((hour >= 22) | (hour < 4)).astype(int)
    features["age"] = (timestamps - birth_dates).dt.days / 365.25
    features["distance_km"] = _haversine_km(
        frame["lat"], frame["long"], frame["merch_lat"], frame["merch_long"]
    )
    features["log_city_pop"] = np.log1p(frame["city_pop"].clip(lower=0))
    for column in SPARKOV_TRANSACTION_CATEGORICAL:
        features[column] = _as_text(frame[column])
    return features


def merge_ieee_identity(
    transaction: pd.DataFrame, identity: pd.DataFrame | None
) -> pd.DataFrame:
    """Left-join the optional identity table on ``TransactionID``."""
    if identity is None:
        merged = transaction.copy()
        merged["has_identity"] = 0
        return merged
    flagged = identity.assign(has_identity=1)
    merged = transaction.merge(flagged, on="TransactionID", how="left", validate="one_to_one")
    merged["has_identity"] = merged["has_identity"].fillna(0).astype(int)
    return merged


def ieee_transaction_features(
    frame: pd.DataFrame, include_anonymised: bool = False
) -> pd.DataFrame:
    """Return transaction-level IEEE-CIS features.

    ``frame`` must already be merged with the identity table (see
    :func:`merge_ieee_identity`). ``TransactionDT`` is elapsed seconds from an unknown
    reference point, so ``hour`` and ``day_of_week`` are relative cycles, not calendar
    values. Set ``include_anonymised`` to add the ``C``, ``D`` and ``M`` columns, whose
    meanings are undisclosed.
    """
    seconds = frame["TransactionDT"].astype("int64")
    amount = frame["TransactionAmt"].clip(lower=0)

    features = pd.DataFrame(index=frame.index)
    features["log_amt"] = np.log1p(amount)
    features["amt_cents"] = (amount - np.floor(amount)).round(3)
    features["hour"] = (seconds // 3600) % 24
    features["day_of_week"] = (seconds // SECONDS_PER_DAY) % 7
    features["has_identity"] = frame.get("has_identity", pd.Series(0, index=frame.index))
    for column in IEEE_TRANSACTION_CATEGORICAL:
        if column in frame:
            features[column] = _as_text(frame[column])
        else:
            features[column] = pd.Series(np.nan, index=frame.index, dtype=object)

    if include_anonymised:
        for column in IEEE_ANONYMISED_NUMERIC:
            if column in frame:
                features[column] = frame[column]
        for column in IEEE_ANONYMISED_CATEGORICAL:
            if column in frame:
                features[column] = _as_text(frame[column])
    return features


# ---------------------------------------------------------------------------
# History features (strictly past transactions of the same entity)
# ---------------------------------------------------------------------------


def _as_datetime(values: pd.Series) -> pd.Series:
    """Convert a datetime-like or elapsed-seconds column into datetimes."""
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_datetime(values, unit="s")
    return pd.to_datetime(values, errors="raise")


def add_history_features(
    frame: pd.DataFrame,
    entity_col: str,
    time_col: str,
    amount_col: str,
    windows: Sequence[str] = DEFAULT_WINDOWS,
    category_col: str | None = None,
    location_cols: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """Summarise each entity's strictly earlier transactions.

    Every value for a transaction at time ``t`` uses only that entity's transactions
    before ``t``. Rolling windows are half-open, ``[t - window, t)``, so transactions
    sharing the exact timestamp are excluded. Cumulative features order same-timestamp
    transactions by their original row order.

    Rows with a missing entity receive missing history features rather than being pooled
    into one artificial entity.

    Returns a DataFrame aligned to ``frame.index`` that contains no identifiers.
    """
    if frame.index.has_duplicates:
        raise ValueError("frame index must be unique")

    work = pd.DataFrame(
        {
            "_entity": frame[entity_col],
            "_time": _as_datetime(frame[time_col]),
            "_amount": frame[amount_col].astype(float),
            "_row": np.arange(len(frame)),
        },
        index=frame.index,
    )
    work["_amount_sq"] = work["_amount"] ** 2
    if category_col is not None:
        work["_category"] = frame[category_col]
    if location_cols is not None:
        work["_lat"] = frame[location_cols[0]].astype(float)
        work["_lon"] = frame[location_cols[1]].astype(float)

    known = work[work["_entity"].notna()].sort_values(
        ["_entity", "_time", "_row"], kind="stable"
    )
    grouped = known.groupby("_entity", sort=False)
    out = pd.DataFrame(index=known.index)

    previous_time = grouped["_time"].shift(1)
    out["secs_since_prev"] = (known["_time"] - previous_time).dt.total_seconds()
    out["prior_txn_count"] = grouped.cumcount()
    out["is_first_txn"] = (out["prior_txn_count"] == 0).astype(int)

    past_sum = grouped["_amount"].cumsum() - known["_amount"]
    past_sq_sum = grouped["_amount_sq"].cumsum() - known["_amount_sq"]
    count = out["prior_txn_count"].replace(0, np.nan)
    past_mean = past_sum / count
    past_var = (past_sq_sum / count - past_mean**2).clip(lower=0)
    past_std = np.sqrt(past_var * count / (count - 1).replace(0, np.nan))
    out["amt_to_prior_mean"] = known["_amount"] / past_mean.replace(0, np.nan)
    out["amt_prior_zscore"] = (known["_amount"] - past_mean) / past_std.replace(0, np.nan)

    timed = known.set_index("_time")
    timed_groups = timed.groupby("_entity", sort=False)["_amount"]
    for window in windows:
        rolling = timed_groups.rolling(window, closed="left")
        counts = rolling.count().to_numpy()
        sums = rolling.sum().to_numpy()
        out[f"txn_count_{window}"] = np.nan_to_num(counts, nan=0.0)
        out[f"amt_sum_{window}"] = np.nan_to_num(sums, nan=0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"amt_mean_{window}"] = np.where(counts > 0, sums / counts, np.nan)

    if category_col is not None:
        seen_before = known.groupby(["_entity", "_category"], sort=False, dropna=False).cumcount()
        out["is_new_category"] = (seen_before == 0).astype(int)

    if location_cols is not None:
        prev_lat = grouped["_lat"].shift(1)
        prev_lon = grouped["_lon"].shift(1)
        out["km_from_prev_txn"] = _haversine_km(known["_lat"], known["_lon"], prev_lat, prev_lon)
        hours = out["secs_since_prev"] / 3600
        out["implied_speed_kmh"] = out["km_from_prev_txn"] / hours.replace(0, np.nan)

    # Float throughout, so the dtype does not depend on whether any entity is missing.
    return out.astype(float).reindex(frame.index)


def sparkov_history_features(
    frame: pd.DataFrame, windows: Sequence[str] = DEFAULT_WINDOWS
) -> pd.DataFrame:
    """History features for Sparkov, grouped by the true card number ``cc_num``."""
    return add_history_features(
        frame,
        entity_col="cc_num",
        time_col="trans_date_trans_time",
        amount_col="amt",
        windows=windows,
        category_col="category",
        location_cols=("merch_lat", "merch_long"),
    )


def ieee_history_features(
    frame: pd.DataFrame,
    entity: pd.Series,
    windows: Sequence[str] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """History features for IEEE-CIS, grouped by a reconstructed card/entity ID.

    ``entity`` must be aligned to ``frame``. It should come from the team's documented
    card-reconstruction method; see ``docs/dataset_decision.md`` for why ``card1``-``card6``
    alone are not a verified identifier.
    """
    work = frame[["TransactionDT", "TransactionAmt", "ProductCD"]].copy()
    work["_entity"] = entity.reindex(frame.index)
    return add_history_features(
        work,
        entity_col="_entity",
        time_col="TransactionDT",
        amount_col="TransactionAmt",
        windows=windows,
        category_col="ProductCD",
    )


def provisional_ieee_entity(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Concatenate columns into a provisional entity key, missing if any part is missing.

    This is a placeholder until the team's card reconstruction is merged. Results that
    use it must be labelled as provisional.
    """
    parts = frame[list(columns)]
    key = parts.astype("string").fillna("").agg("|".join, axis=1)
    return key.where(parts.notna().all(axis=1)).rename("entity")
