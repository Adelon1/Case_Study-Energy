"""Build model-ready tables for all target options.

This module is the only place that knows how to turn the large hourly feature
store into a clean modelling table. Models receive only:

``timestamp_utc``, one target column, and selected feature columns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline_helpers.curve_translation.forecast_blocks import peakload_mask
from pipeline_helpers.modelling import constants


@dataclass(frozen=True)
class ModellingDataset:
    """Model-ready table, target column, and selected feature columns."""

    table: pd.DataFrame
    target_column: str
    feature_columns: list[str]
    target_option: str
    feature_mode: str


PERIOD_TARGET_COLUMN = "period_average_price_eur_per_mwh"

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
    "is_holiday",
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

# Monday (local_weekday_0) is dropped on purpose: with all seven dummies the set
# is collinear with the intercept, which destabilises the linear LEAR weights.
WEEKDAY_DUMMY_FEATURES = [f"local_weekday_{weekday}" for weekday in range(1, 7)]
DAILY_PRICE_CURVE_LAGS = [1, 2, 3, 7]
DAILY_PRICE_SUMMARY_LAGS = [1, 7]
PRICE_DAILY_CURVE_PATTERN = re.compile(r"^price_d(\d+)_h\d{2}$")
PRICE_DAILY_SUMMARY_PATTERN = re.compile(r"^price_d(\d+)_(min|max|mean)$")


def existing_columns(table: pd.DataFrame, columns: list[str]) -> list[str]:
    """Keep columns present in the table."""

    return [column for column in columns if column in table.columns]


def require_columns(table: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return columns, raising a clear error if any are missing."""

    missing = [column for column in columns if column not in table.columns]
    if missing:
        preview = ", ".join(missing[:10])
        if len(missing) > 10:
            preview = f"{preview}, ..."
        raise ValueError(f"Feature table is missing required features: {preview}")
    return columns.copy()


def calendar_features_for_model(model_name: str) -> list[str]:
    """Use raw calendar values for tree models and smooth values for linear models."""

    if model_name == "boosted_trees":
        return TREE_CALENDAR_FEATURES + WEEKDAY_DUMMY_FEATURES
    return LINEAR_CALENDAR_FEATURES + WEEKDAY_DUMMY_FEATURES


def day_ahead_price_features(table: pd.DataFrame) -> list[str]:
    """Price features valid when forecasting one whole delivery day ahead."""

    columns: list[str] = []
    columns.extend(
        existing_columns(
            table,
            [
                "price_rolling_mean_24",
                "price_rolling_std_24",
                "price_rolling_mean_168",
                "price_rolling_std_168",
            ],
        )
    )

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
        curve_match = PRICE_DAILY_CURVE_PATTERN.match(column)
        if curve_match and int(curve_match.group(1)) * 24 >= minimum_lag_hours:
            allowed.append(column)
            continue

        summary_match = PRICE_DAILY_SUMMARY_PATTERN.match(column)
        if summary_match and int(summary_match.group(1)) * 24 >= minimum_lag_hours:
            allowed.append(column)

    return sorted(allowed)


def select_hourly_features(
    table: pd.DataFrame,
    model_name: str,
    feature_mode: str,
    period_days: int,
) -> list[str]:
    """Select leakage-aware feature columns for Option A hourly rows."""

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

    return list(dict.fromkeys(base_columns + price_columns))


def add_target_lag_features(
    table: pd.DataFrame,
    target_column: str,
    lag_rows: list[int],
) -> pd.DataFrame:
    """Add generic target lag columns measured in rows of the modelling table."""

    result = table.sort_values(constants.TIMESTAMP_COLUMN).reset_index(drop=True)
    for lag in lag_rows:
        result[f"target_lag_{lag}"] = result[target_column].shift(lag)
    return result


def build_hourly_dataset(
    feature_table: pd.DataFrame,
    model_name: str,
    feature_mode: str,
    period_days: int,
) -> ModellingDataset:
    """Build hourly rows: one target value per delivery hour."""

    table = feature_table.copy()
    table[constants.TIMESTAMP_COLUMN] = pd.to_datetime(table[constants.TIMESTAMP_COLUMN], utc=True)
    table = table.sort_values(constants.TIMESTAMP_COLUMN).reset_index(drop=True)
    feature_columns = select_hourly_features(table, model_name, feature_mode, period_days)
    return ModellingDataset(
        table=table,
        target_column=constants.TARGET_COLUMN,
        feature_columns=feature_columns,
        target_option="hourly",
        feature_mode=feature_mode,
    )


def period_block_mask(table: pd.DataFrame, block: str) -> pd.Series:
    """Return the rows used by a period-average block."""

    if block == "baseload":
        return pd.Series(True, index=table.index)
    if block == "peakload":
        return peakload_mask(table[constants.TIMESTAMP_COLUMN])
    if block == "offpeak":
        return ~peakload_mask(table[constants.TIMESTAMP_COLUMN])
    raise ValueError("period-average modelling supports baseload, peakload, or offpeak.")


def build_period_average_dataset(
    feature_table: pd.DataFrame,
    period_days: int,
    block: str,
) -> ModellingDataset:
    """Build period-average rows: one target value per delivery period."""

    table = feature_table.copy()
    table[constants.TIMESTAMP_COLUMN] = pd.to_datetime(table[constants.TIMESTAMP_COLUMN], utc=True)
    table = table.sort_values(constants.TIMESTAMP_COLUMN).reset_index(drop=True)

    first_period = table[constants.TIMESTAMP_COLUMN].min().floor(f"{period_days}D")
    table["period_start"] = first_period + (
        (table[constants.TIMESTAMP_COLUMN] - first_period)
        // pd.Timedelta(days=period_days)
    ) * pd.Timedelta(days=period_days)
    table = table.loc[period_block_mask(table, block)].copy()

    aggregations = {
        constants.TIMESTAMP_COLUMN: ("period_start", "first"),
        "period_end": ("period_start", lambda values: values.iloc[0] + pd.Timedelta(days=period_days)),
        PERIOD_TARGET_COLUMN: (constants.TARGET_COLUMN, "mean"),
        "period_n_hours": (constants.TARGET_COLUMN, "size"),
    }
    for column in [
        "load_forecast_mw",
        "solar_forecast_mw",
        "wind_total_forecast_mw",
        "renewable_total_forecast_mw",
        "residual_load_forecast_mw",
    ]:
        if column in table.columns:
            aggregations[f"{column}_mean"] = (column, "mean")
            aggregations[f"{column}_min"] = (column, "min")
            aggregations[f"{column}_max"] = (column, "max")
            aggregations[f"{column}_std"] = (column, "std")

    period_rows = table.groupby("period_start", as_index=False).agg(**aggregations)
    period_rows = add_target_lag_features(period_rows, PERIOD_TARGET_COLUMN, [1, 2, 4, 7, 12])

    timestamps = pd.to_datetime(period_rows[constants.TIMESTAMP_COLUMN], utc=True)
    period_rows["period_month"] = timestamps.dt.month
    period_rows["period_month_sin"] = np.sin(2 * np.pi * period_rows["period_month"] / 12)
    period_rows["period_month_cos"] = np.cos(2 * np.pi * period_rows["period_month"] / 12)
    period_rows["period_start_weekday"] = timestamps.dt.weekday
    period_rows["period_start_weekday_sin"] = np.sin(
        2 * np.pi * period_rows["period_start_weekday"] / 7
    )
    period_rows["period_start_weekday_cos"] = np.cos(
        2 * np.pi * period_rows["period_start_weekday"] / 7
    )

    excluded = {
        constants.TIMESTAMP_COLUMN,
        "period_end",
        PERIOD_TARGET_COLUMN,
    }
    feature_columns = [column for column in period_rows.columns if column not in excluded]
    period_rows = period_rows.dropna(subset=[PERIOD_TARGET_COLUMN]).reset_index(drop=True)
    return ModellingDataset(
        table=period_rows,
        target_column=PERIOD_TARGET_COLUMN,
        feature_columns=feature_columns,
        target_option="period_average",
        feature_mode="period_average_safe",
    )


def build_modelling_dataset(
    feature_table: pd.DataFrame,
    model_name: str,
    target_option: str,
    feature_mode: str,
    period_days: int,
    block: str,
) -> ModellingDataset:
    """Build the table consumed by validation and model training."""

    if target_option == "hourly":
        return build_hourly_dataset(feature_table, model_name, feature_mode, period_days)
    if target_option == "period_average":
        return build_period_average_dataset(feature_table, period_days, block)
    raise ValueError(f"Unsupported target option: {target_option}")
