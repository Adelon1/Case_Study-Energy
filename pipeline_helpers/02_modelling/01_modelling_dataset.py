"""Build model-ready tables for all target options.

This module is the only place that knows how to turn the large hourly feature
store into a clean modelling table. Models receive only:

``timestamp_utc``, one target column, and selected feature columns.

Public entry point:
    ``build_modelling_dataset(...)``

Everything else in this file is a helper used to construct that one returned
``ModellingDataset`` object.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_modelling_dataset(
    feature_table: pd.DataFrame,
    model_name: str,
    period_days: int = 1,
    block: str = "baseload",
    forecast_setup: str = "hourly_day_ahead",
) -> ModellingDataset:
    """Build the single table consumed by validation and model training.

    This is the only function other modules should call from this file.

    Parameters mean:
      - ``forecast_setup="hourly_day_ahead"``: hourly target, one delivery day
        ahead, price-lag features may be allowed by the feature policy.
      - ``forecast_setup="hourly_period"``: hourly target across a longer
        prediction period, price-lag features are removed because they would not
        be known for the whole future period.
      - ``forecast_setup="period_average"``: one row per delivery period, target
        is the average price over that period.

    The function returns a ``ModellingDataset`` containing:
      - ``table``: timestamp + target + candidate feature columns.
      - ``target_column``: the column models must learn.
      - ``feature_columns``: leakage-safe features selected for this model/setup.
      - ``feature_policy``: a human-readable explanation stored in metadata.
    """

    if forecast_setup not in constants.FORECAST_SETUPS:
        raise ValueError(f"Unsupported forecast setup: {forecast_setup}")
    if forecast_setup in {"hourly_day_ahead", "hourly_period"}:
        return build_hourly_dataset(feature_table, model_name, forecast_setup)
    if forecast_setup == "period_average":
        return build_period_average_dataset(feature_table, model_name, period_days)
    raise ValueError(f"Unsupported forecast setup: {forecast_setup}")


# ---------------------------------------------------------------------------
# Feature policy constants and config
# ---------------------------------------------------------------------------

# The feature-policy decisions live in ``feature_sets.json`` so new models can
# be added without editing the dataset-building logic. Python still owns dynamic
# bundles such as "all price_d*_hXX columns" because those are generated column
# patterns, not fixed names.

PERIOD_TARGET_COLUMN = "period_average_price_eur_per_mwh"
FEATURE_CONFIG_PATH = Path(__file__).with_name("feature_sets.json")
DAILY_PRICE_CURVE_LAGS = [1, 2, 3, 7]
DAILY_PRICE_SUMMARY_LAGS = [1, 7]
PERIOD_TARGET_LAGS = [1, 2, 4, 7, 12]
RAW_PERIOD_CALENDAR_FEATURES = {"period_month", "period_start_weekday"}


# ---------------------------------------------------------------------------
# Generic column helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Feature-policy resolver
# ---------------------------------------------------------------------------


def load_feature_config(path: str | Path = FEATURE_CONFIG_PATH) -> dict[str, object]:
    """Read the model/setup to feature-bundle mapping."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def configured_bundle(config: dict[str, object], bundle_name: str) -> list[str]:
    """Return a literal feature bundle from ``feature_sets.json``."""

    bundles = config.get("bundles", {})
    if not isinstance(bundles, dict) or bundle_name not in bundles:
        raise ValueError(f"Unknown feature bundle: {bundle_name}")
    columns = bundles[bundle_name]
    if not isinstance(columns, list) or not all(isinstance(column, str) for column in columns):
        raise ValueError(f"Feature bundle '{bundle_name}' must be a list of column names.")
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


def all_numeric_period_without_raw_calendar(period_rows: pd.DataFrame) -> list[str]:
    """All numeric period features except target, timestamp, and raw calendar integers."""

    excluded = {
        "period_start",
        constants.TIMESTAMP_COLUMN,
        "period_end",
        PERIOD_TARGET_COLUMN,
        *RAW_PERIOD_CALENDAR_FEATURES,
    }
    return [
        column
        for column in period_rows.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(period_rows[column])
    ]


def resolve_feature_bundle(
    table: pd.DataFrame,
    config: dict[str, object],
    bundle_name: str,
) -> list[str]:
    """Resolve one configured bundle name into concrete table columns."""

    if bundle_name == "day_ahead_price_curve_features":
        return day_ahead_price_curve_features(table)
    if bundle_name == "day_ahead_price_features":
        return day_ahead_price_features(table)
    if bundle_name == "period_target_lags":
        return existing_columns(table, [f"target_lag_{lag}" for lag in PERIOD_TARGET_LAGS])
    if bundle_name == "all_numeric_period_without_raw_calendar":
        return all_numeric_period_without_raw_calendar(table)
    return require_columns(table, configured_bundle(config, bundle_name))


def resolve_feature_policy(
    table: pd.DataFrame,
    model_name: str,
    forecast_setup: str,
) -> tuple[list[str], str]:
    """Select feature columns using ``feature_sets.json``."""

    config = load_feature_config()
    policies = config.get("policies", {})
    if not isinstance(policies, dict) or model_name not in policies:
        raise ValueError(
            f"No feature policy configured for model '{model_name}'. "
            f"Add it to {FEATURE_CONFIG_PATH}."
        )
    model_policy = policies[model_name]
    if not isinstance(model_policy, dict) or forecast_setup not in model_policy:
        raise ValueError(
            f"No feature policy configured for model '{model_name}' and setup '{forecast_setup}'."
        )

    bundle_names = model_policy[forecast_setup]
    if not isinstance(bundle_names, list) or not all(
        isinstance(bundle_name, str) for bundle_name in bundle_names
    ):
        raise ValueError(
            f"Feature policy for model '{model_name}' and setup '{forecast_setup}' "
            "must be a list of bundle names."
        )

    columns: list[str] = []
    for bundle_name in bundle_names:
        columns.extend(resolve_feature_bundle(table, config, bundle_name))

    unique_columns = list(dict.fromkeys(columns))
    feature_policy = "+".join(bundle_names) if bundle_names else "no_features_historical_mean"
    return unique_columns, feature_policy


# ---------------------------------------------------------------------------
# Hourly target dataset
# ---------------------------------------------------------------------------


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


def add_target_lag_features_by_group(
    table: pd.DataFrame,
    target_column: str,
    group_column: str,
    lag_rows: list[int],
) -> pd.DataFrame:
    """Add target lags separately per group, e.g. per period-average block."""

    result = table.sort_values([group_column, constants.TIMESTAMP_COLUMN]).reset_index(drop=True)
    grouped_target = result.groupby(group_column, sort=False)[target_column]
    for lag in lag_rows:
        result[f"target_lag_{lag}"] = grouped_target.shift(lag)
    return result.sort_values([constants.TIMESTAMP_COLUMN, group_column]).reset_index(drop=True)


def build_hourly_dataset(
    feature_table: pd.DataFrame,
    model_name: str,
    forecast_setup: str,
) -> ModellingDataset:
    """Build hourly rows: one target value per delivery hour."""

    table = feature_table.copy()
    table[constants.TIMESTAMP_COLUMN] = pd.to_datetime(table[constants.TIMESTAMP_COLUMN], utc=True)
    table = table.sort_values(constants.TIMESTAMP_COLUMN).reset_index(drop=True)
    feature_columns, feature_policy = resolve_feature_policy(table, model_name, forecast_setup)
    return ModellingDataset(
        table=table,
        target_column=constants.TARGET_COLUMN,
        feature_columns=feature_columns,
        forecast_setup=forecast_setup,
        feature_policy=feature_policy,
    )


# ---------------------------------------------------------------------------
# Period-average target dataset
# ---------------------------------------------------------------------------


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
) -> ModellingDataset:
    """Build period-average rows: one target value per period and block."""

    table = feature_table.copy()
    table[constants.TIMESTAMP_COLUMN] = pd.to_datetime(table[constants.TIMESTAMP_COLUMN], utc=True)
    table = table.sort_values(constants.TIMESTAMP_COLUMN).reset_index(drop=True)

    first_period = table[constants.TIMESTAMP_COLUMN].min().floor(f"{period_days}D")
    table["period_start"] = first_period + (
        (table[constants.TIMESTAMP_COLUMN] - first_period)
        // pd.Timedelta(days=period_days)
    ) * pd.Timedelta(days=period_days)

    period_tables: list[pd.DataFrame] = []
    for block in ["baseload", "peakload", "offpeak"]:
        block_table = table.loc[period_block_mask(table, block)].copy()
        aggregations = {
            constants.TIMESTAMP_COLUMN: ("period_start", "first"),
            "period_end": ("period_start", lambda values: values.iloc[0] + pd.Timedelta(days=period_days)),
            PERIOD_TARGET_COLUMN: (constants.TARGET_COLUMN, "mean"),
            "period_n_hours": (constants.TARGET_COLUMN, "size"),
        }
        for column in configured_bundle(load_feature_config(), "period_source_fundamentals"):
            if column in block_table.columns:
                aggregations[f"{column}_mean"] = (column, "mean")
                aggregations[f"{column}_min"] = (column, "min")
                aggregations[f"{column}_max"] = (column, "max")
                aggregations[f"{column}_std"] = (column, "std")

        block_rows = block_table.groupby("period_start", as_index=False).agg(**aggregations)
        block_rows["block"] = block
        period_tables.append(block_rows)

    period_rows = pd.concat(period_tables, ignore_index=True).sort_values(
        [constants.TIMESTAMP_COLUMN, "block"]
    )
    period_rows["block_peakload"] = (period_rows["block"] == "peakload").astype(int)
    period_rows["block_offpeak"] = (period_rows["block"] == "offpeak").astype(int)
    period_rows = add_target_lag_features_by_group(
        period_rows,
        PERIOD_TARGET_COLUMN,
        "block",
        PERIOD_TARGET_LAGS,
    )

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

    feature_columns, feature_policy = resolve_feature_policy(
        period_rows,
        model_name,
        "period_average",
    )
    period_rows = period_rows.dropna(subset=[PERIOD_TARGET_COLUMN]).reset_index(drop=True)
    return ModellingDataset(
        table=period_rows,
        target_column=PERIOD_TARGET_COLUMN,
        feature_columns=feature_columns,
        forecast_setup="period_average",
        feature_policy=feature_policy,
    )
