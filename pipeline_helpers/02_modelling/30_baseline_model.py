"""Naive baseline models for hourly and period-average targets."""

from __future__ import annotations

import pandas as pd


MODEL_NAME = "baseline_model"
TIMESTAMP_COLUMN = "timestamp_utc"
HOURLY_LAG_DAYS = {
    24: 1,
    48: 2,
    168: 7,
}


def build_param_grid(
    feature_columns: list[str] | None = None,
    **_unused_options,
) -> list[dict[str, object]]:
    """Build available naive baselines from columns present in the table.

    The modelling dataset owns leakage-safe feature selection. The baseline only
    asks which lag columns survived that selection and falls back to a training
    mean if no lag baseline is available.
    """

    available_columns = set(feature_columns or [])

    period_grid = [
        {"lag_rows": lag}
        for lag in [1, 2, 4, 7, 12]
        if f"target_lag_{lag}" in available_columns
    ]
    if period_grid:
        return period_grid

    hourly_grid = [
        {"lag_rows": lag_rows}
        for lag_rows, day_lag in HOURLY_LAG_DAYS.items()
        if all(f"price_d{day_lag}_h{hour:02d}" in available_columns for hour in range(24))
    ]
    return hourly_grid or [{"method": "historical_mean"}]


def train(train_data: pd.DataFrame, params: dict[str, object]) -> None:
    """Fit the optional historical-mean baseline."""

    if params.get("method") == "historical_mean":
        target_column = str(params.get("_target_column", "day_ahead_price_eur_per_mwh"))
        return float(train_data[target_column].dropna().mean())
    return None


def predict(model_state: None, test_data: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """Predict target values with a generic row-lag column."""

    if params.get("method") == "historical_mean":
        return pd.Series(float(model_state), index=test_data.index)

    lag_rows = int(params["lag_rows"])
    baseline_column = f"target_lag_{lag_rows}"
    if baseline_column in test_data.columns:
        return test_data[baseline_column]

    if lag_rows not in HOURLY_LAG_DAYS:
        raise ValueError(f"Missing target lag column required by baseline: {baseline_column}")
    if TIMESTAMP_COLUMN not in test_data.columns:
        raise ValueError(f"Hourly baseline needs '{TIMESTAMP_COLUMN}' to choose price curve lag columns.")

    day_lag = HOURLY_LAG_DAYS[lag_rows]
    utc_hours = pd.to_datetime(test_data[TIMESTAMP_COLUMN], utc=True).dt.hour
    predictions = pd.Series(index=test_data.index, dtype=float)
    for hour in range(24):
        column = f"price_d{day_lag}_h{hour:02d}"
        if column not in test_data.columns:
            raise ValueError(f"Missing hourly price curve lag required by baseline: {column}")
        predictions.loc[utc_hours == hour] = test_data.loc[utc_hours == hour, column]
    return predictions
