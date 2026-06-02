"""Build model-ready tables for all target options.

This module is the only place that knows how to turn the large hourly feature
store into a clean modelling table. Models receive only:

``timestamp_utc``, one target column, and selected feature columns.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

forecast_blocks = importlib.import_module("pipeline_helpers.03_curve_translation.01_forecast_blocks")
constants = importlib.import_module("pipeline_helpers.02_modelling.00_constants")

peakload_mask = forecast_blocks.peakload_mask


@dataclass(frozen=True)
class ModellingDataset:
    """Model-ready table, target column, and selected feature columns."""

    table: pd.DataFrame
    target_column: str
    feature_columns: list[str]
    forecast_setup: str
    feature_policy: str


PERIOD_TARGET_COLUMN = "period_average_price_eur_per_mwh"
LINEAR_MODELS = {"lear_model", "ransac_lasso_model"}
COMPACT_MODELS = {"boosted_tree_model", "theil_sen_model"}
BASELINE_MODEL = "baseline_model"

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

# Monday (local_weekday_0) is dropped on purpose: with all seven dummies the set
# is collinear with the intercept, which destabilises the linear LEAR weights.
WEEKDAY_DUMMY_FEATURES = [f"local_weekday_{weekday}" for weekday in range(1, 7)]
DAILY_PRICE_CURVE_LAGS = [1, 2, 3, 7]
DAILY_PRICE_SUMMARY_LAGS = [1, 7]
CORE_HOURLY_FUNDAMENTALS = [
    "load_forecast_mw",
    "solar_forecast_mw",
    "wind_total_forecast_mw",
]
PERIOD_FUNDAMENTAL_COLUMNS = [
    "load_forecast_mw",
    "solar_forecast_mw",
    "wind_total_forecast_mw",
    "renewable_total_forecast_mw",
    "residual_load_forecast_mw",
]
PERIOD_TARGET_LAGS = [1, 2, 4, 7, 12]
CORE_PERIOD_FEATURES = [
    "load_forecast_mw_mean",
    "solar_forecast_mw_mean",
    "wind_total_forecast_mw_mean",
    "target_lag_1",
    "target_lag_2",
]
RAW_PERIOD_CALENDAR_FEATURES = {"period_month", "period_start_weekday"}


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


def day_ahead_price_curve_features(table: pd.DataFrame) -> list[str]:
    """Full previous-day UTC price curves valid for day-ahead forecasts."""

    columns: list[str] = []
    for day_lag in DAILY_PRICE_CURVE_LAGS:
        columns.extend(
            existing_columns(table, [f"price_d{day_lag}_h{hour:02d}" for hour in range(24)])
        )
    return columns


def day_ahead_price_features(table: pd.DataFrame) -> list[str]:
    """Price curve and summary features valid for one delivery-day-ahead forecasts."""

    columns = day_ahead_price_curve_features(table)
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


def select_hourly_features(
    table: pd.DataFrame,
    model_name: str,
    forecast_setup: str,
) -> tuple[list[str], str]:
    """Select feature columns for hourly forecast setups."""

    if forecast_setup not in {"hourly_day_ahead", "hourly_period"}:
        raise ValueError(f"Unsupported hourly forecast setup: {forecast_setup}")

    full_price_columns = day_ahead_price_features(table)
    curve_price_columns = day_ahead_price_curve_features(table)

    if model_name == BASELINE_MODEL:
        if forecast_setup == "hourly_day_ahead":
            return full_price_columns, "baseline_hourly_price_lags"
        return [], "baseline_historical_mean"

    if model_name in LINEAR_MODELS:
        base_columns = (
            existing_columns(table, FUNDAMENTAL_FEATURES)
            + require_columns(table, LINEAR_CALENDAR_FEATURES + WEEKDAY_DUMMY_FEATURES)
        )
        if forecast_setup == "hourly_day_ahead":
            return list(dict.fromkeys(base_columns + full_price_columns)), "linear_hourly_all_safe_features"
        return list(dict.fromkeys(base_columns)), "linear_hourly_fundamentals_calendar"

    if model_name in COMPACT_MODELS:
        base_columns = require_columns(table, CORE_HOURLY_FUNDAMENTALS)
        if forecast_setup == "hourly_day_ahead":
            return list(dict.fromkeys(base_columns + curve_price_columns)), "compact_hourly_core_fundamentals_price_curves"
        return base_columns, "compact_hourly_core_fundamentals"

    raise ValueError(f"Unsupported model for hourly feature selection: {model_name}")


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
    forecast_setup: str,
) -> ModellingDataset:
    """Build hourly rows: one target value per delivery hour."""

    table = feature_table.copy()
    table[constants.TIMESTAMP_COLUMN] = pd.to_datetime(table[constants.TIMESTAMP_COLUMN], utc=True)
    table = table.sort_values(constants.TIMESTAMP_COLUMN).reset_index(drop=True)
    feature_columns, feature_policy = select_hourly_features(table, model_name, forecast_setup)
    return ModellingDataset(
        table=table,
        target_column=constants.TARGET_COLUMN,
        feature_columns=feature_columns,
        forecast_setup=forecast_setup,
        feature_policy=feature_policy,
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
    model_name: str,
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
    for column in PERIOD_FUNDAMENTAL_COLUMNS:
        if column in table.columns:
            aggregations[f"{column}_mean"] = (column, "mean")
            aggregations[f"{column}_min"] = (column, "min")
            aggregations[f"{column}_max"] = (column, "max")
            aggregations[f"{column}_std"] = (column, "std")

    period_rows = table.groupby("period_start", as_index=False).agg(**aggregations)
    period_rows = add_target_lag_features(period_rows, PERIOD_TARGET_COLUMN, PERIOD_TARGET_LAGS)

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

    feature_columns, feature_policy = select_period_average_features(period_rows, model_name)
    period_rows = period_rows.dropna(subset=[PERIOD_TARGET_COLUMN]).reset_index(drop=True)
    return ModellingDataset(
        table=period_rows,
        target_column=PERIOD_TARGET_COLUMN,
        feature_columns=feature_columns,
        forecast_setup="period_average",
        feature_policy=feature_policy,
    )


def select_period_average_features(
    period_rows: pd.DataFrame,
    model_name: str,
) -> tuple[list[str], str]:
    """Select feature columns for direct period-average forecasts."""

    if model_name == BASELINE_MODEL:
        columns = existing_columns(period_rows, [f"target_lag_{lag}" for lag in PERIOD_TARGET_LAGS])
        return columns, "baseline_period_target_lags" if columns else "baseline_historical_mean"

    if model_name in LINEAR_MODELS:
        excluded = {
            "period_start",
            constants.TIMESTAMP_COLUMN,
            "period_end",
            PERIOD_TARGET_COLUMN,
            *RAW_PERIOD_CALENDAR_FEATURES,
        }
        columns = [
            column
            for column in period_rows.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(period_rows[column])
        ]
        return columns, "linear_period_all_numeric_without_raw_calendar"

    if model_name in COMPACT_MODELS:
        return require_columns(period_rows, CORE_PERIOD_FEATURES), "compact_period_core_fundamentals_target_lags"

    raise ValueError(f"Unsupported model for period-average feature selection: {model_name}")


def build_modelling_dataset(
    feature_table: pd.DataFrame,
    model_name: str,
    period_days: int = 1,
    block: str = "baseload",
    forecast_setup: str = "hourly_day_ahead",
) -> ModellingDataset:
    """Build the table consumed by validation and model training."""

    if forecast_setup not in constants.FORECAST_SETUPS:
        raise ValueError(f"Unsupported forecast setup: {forecast_setup}")
    if forecast_setup in {"hourly_day_ahead", "hourly_period"}:
        return build_hourly_dataset(feature_table, model_name, forecast_setup)
    if forecast_setup == "period_average":
        return build_period_average_dataset(feature_table, model_name, period_days, block)
    raise ValueError(f"Unsupported forecast setup: {forecast_setup}")
