"""Central feature-set selection for leakage-aware modelling.

The feature table may contain many columns. This module decides which columns
are allowed for a specific forecast interpretation.
"""

from __future__ import annotations

import re

import pandas as pd


FUNDAMENTAL_FEATURES = [
    "load_forecast_mw",
    "solar_forecast_mw",
    "wind_onshore_forecast_mw",
    "wind_offshore_forecast_mw",
    "wind_total_forecast_mw",
    "renewable_total_forecast_mw",
    "residual_load_forecast_mw",
    "wind_share_of_load",
    "solar_share_of_load",
    "renewable_share_of_load",
]

LINEAR_CALENDAR_FEATURES = [
    "local_is_weekend",
    "local_hour_sin",
    "local_hour_cos",
    "local_weekday_sin",
    "local_weekday_cos",
    "local_month_sin",
    "local_month_cos",
    "local_day_of_year_sin",
    "local_day_of_year_cos",
]

TREE_CALENDAR_FEATURES = [
    "local_hour",
    "local_weekday",
    "local_month",
    "local_day_of_year",
    *LINEAR_CALENDAR_FEATURES,
]

WEEKDAY_DUMMY_FEATURES = [f"local_weekday_{weekday}" for weekday in range(7)]

DAILY_PRICE_CURVE_LAGS = [1, 2, 3, 7]
DAILY_PRICE_SUMMARY_LAGS = [1, 7]
PRICE_LAG_PATTERN = re.compile(r"^price_lag_(\d+)$")
PRICE_DAILY_CURVE_PATTERN = re.compile(r"^price_d(\d+)_h\d{2}$")
PRICE_DAILY_SUMMARY_PATTERN = re.compile(r"^price_d(\d+)_(min|max|mean)$")


def existing_columns(table: pd.DataFrame, columns: list[str]) -> list[str]:
    """Keep columns present in the table."""

    return [column for column in columns if column in table.columns]


def missing_columns(table: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return requested columns absent from the table."""

    return [column for column in columns if column not in table.columns]


def require_columns(table: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return columns, raising a clear error if any are missing."""

    missing = missing_columns(table, columns)
    if missing:
        preview = ", ".join(missing[:10])
        if len(missing) > 10:
            preview = f"{preview}, ..."
        raise ValueError(f"Feature table is missing required features: {preview}")
    return columns.copy()


def calendar_features_for_model(model_name: str) -> list[str]:
    """Use richer raw calendar values for tree models and smooth values for linear models."""

    if model_name == "hist_gradient_boosting":
        return TREE_CALENDAR_FEATURES + WEEKDAY_DUMMY_FEATURES
    return LINEAR_CALENDAR_FEATURES + WEEKDAY_DUMMY_FEATURES


def day_ahead_price_features(table: pd.DataFrame) -> list[str]:
    """Price features valid when forecasting one whole delivery day ahead."""

    columns: list[str] = []
    columns.extend(existing_columns(table, ["price_lag_24", "price_lag_48", "price_lag_168"]))

    for day_lag in DAILY_PRICE_CURVE_LAGS:
        columns.extend(
            existing_columns(table, [f"price_d{day_lag}_h{hour:02d}" for hour in range(24)])
        )
    for day_lag in DAILY_PRICE_SUMMARY_LAGS:
        columns.extend(
            existing_columns(
                table,
                [
                    f"price_d{day_lag}_min",
                    f"price_d{day_lag}_max",
                    f"price_d{day_lag}_mean",
                ],
            )
        )
    return columns


def safe_period_price_features(table: pd.DataFrame, period_days: int) -> list[str]:
    """Keep only price features whose lag is at least the full forecast period."""

    minimum_lag_hours = 24 * period_days
    allowed: list[str] = []

    for column in table.columns:
        lag_match = PRICE_LAG_PATTERN.match(column)
        if lag_match and int(lag_match.group(1)) >= minimum_lag_hours:
            allowed.append(column)
            continue

        curve_match = PRICE_DAILY_CURVE_PATTERN.match(column)
        if curve_match and int(curve_match.group(1)) * 24 >= minimum_lag_hours:
            allowed.append(column)
            continue

        summary_match = PRICE_DAILY_SUMMARY_PATTERN.match(column)
        if summary_match and int(summary_match.group(1)) * 24 >= minimum_lag_hours:
            allowed.append(column)

    return sorted(allowed)


def get_hourly_feature_columns(
    table: pd.DataFrame,
    model_name: str,
    feature_mode: str = "day_ahead_full",
    period_days: int = 1,
) -> list[str]:
    """Return allowed hourly modelling columns.

    ``day_ahead_full`` may use previous delivery-day prices. ``period_hourly_safe``
    removes price features that would look inside the requested period.
    Rolling price features are intentionally excluded because the old
    ``price.shift(1).rolling(...)`` construction is not valid for full-day
    day-ahead forecasts.
    """

    base_columns = (
        existing_columns(table, FUNDAMENTAL_FEATURES)
        + require_columns(table, calendar_features_for_model(model_name))
    )

    if feature_mode == "day_ahead_full":
        price_columns = day_ahead_price_features(table)
    elif feature_mode == "period_hourly_safe":
        price_columns = safe_period_price_features(table, period_days)
    elif feature_mode == "fundamentals_calendar_only":
        price_columns = []
    else:
        raise ValueError(f"Unsupported feature mode: {feature_mode}")

    selected = base_columns + price_columns
    if not selected:
        raise ValueError("No feature columns were selected.")
    return list(dict.fromkeys(selected))
