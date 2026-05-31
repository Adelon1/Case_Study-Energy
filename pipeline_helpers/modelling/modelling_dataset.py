"""Build model-ready tables for Option A and Option B."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline_helpers.curve_translation.forecast_blocks import peakload_mask
from pipeline_helpers.modelling import constants
from pipeline_helpers.modelling.feature_sets import get_hourly_feature_columns


@dataclass(frozen=True)
class ModellingDataset:
    """Model-ready table, target column, and selected feature columns."""

    table: pd.DataFrame
    target_column: str
    feature_columns: list[str]
    target_option: str
    feature_mode: str


PERIOD_TARGET_COLUMN = "period_average_price_eur_per_mwh"


def build_hourly_dataset(
    feature_table: pd.DataFrame,
    model_name: str,
    feature_mode: str,
    period_days: int,
) -> ModellingDataset:
    """Build Option A hourly rows with leakage-aware selected features."""

    feature_columns = get_hourly_feature_columns(
        feature_table,
        model_name=model_name,
        feature_mode=feature_mode,
        period_days=period_days,
    )
    return ModellingDataset(
        table=feature_table.copy(),
        target_column=constants.TARGET_COLUMN,
        feature_columns=feature_columns,
        target_option="A",
        feature_mode=feature_mode,
    )


def add_period_history_features(period_rows: pd.DataFrame) -> pd.DataFrame:
    """Add history features that only use completed previous periods."""

    rows = period_rows.sort_values(constants.TIMESTAMP_COLUMN).reset_index(drop=True)
    target = rows[PERIOD_TARGET_COLUMN]
    rows["previous_period_price_mean"] = target.shift(1)
    rows["previous_4_period_price_mean"] = target.shift(1).rolling(4).mean()
    rows["previous_12_period_price_mean"] = target.shift(1).rolling(12).mean()
    return rows


def build_period_average_dataset(
    feature_table: pd.DataFrame,
    period_days: int,
    block: str,
) -> ModellingDataset:
    """Build Option B rows where one row is one delivery period.

    Fundamentals inside the period are treated as forecasts available when the
    forecast is made. Price-history features use only earlier completed periods.
    """

    table = feature_table.copy()
    table[constants.TIMESTAMP_COLUMN] = pd.to_datetime(table[constants.TIMESTAMP_COLUMN], utc=True)
    table = table.sort_values(constants.TIMESTAMP_COLUMN).reset_index(drop=True)

    period_start = table[constants.TIMESTAMP_COLUMN].min().floor(f"{period_days}D")
    table["period_start"] = period_start + (
        (table[constants.TIMESTAMP_COLUMN] - period_start)
        // pd.Timedelta(days=period_days)
    ) * pd.Timedelta(days=period_days)

    if block == "baseload":
        block_mask = pd.Series(True, index=table.index)
    elif block == "peakload":
        block_mask = peakload_mask(table[constants.TIMESTAMP_COLUMN])
    elif block == "offpeak":
        block_mask = ~peakload_mask(table[constants.TIMESTAMP_COLUMN])
    else:
        raise ValueError("Option B period-average modelling supports baseload, peakload, or offpeak.")
    table = table.loc[block_mask].copy()

    aggregations = {
        constants.TIMESTAMP_COLUMN: ("period_start", "first"),
        "period_end": ("period_start", lambda values: values.iloc[0] + pd.Timedelta(days=period_days)),
        PERIOD_TARGET_COLUMN: (constants.TARGET_COLUMN, "mean"),
        "period_n_hours": (constants.TARGET_COLUMN, "size"),
    }

    candidate_columns = [
        "load_forecast_mw",
        "solar_forecast_mw",
        "wind_total_forecast_mw",
        "renewable_total_forecast_mw",
        "residual_load_forecast_mw",
    ]
    for column in candidate_columns:
        if column in table.columns:
            aggregations[f"{column}_mean"] = (column, "mean")
            aggregations[f"{column}_min"] = (column, "min")
            aggregations[f"{column}_max"] = (column, "max")
            aggregations[f"{column}_std"] = (column, "std")

    period_rows = table.groupby("period_start", as_index=False).agg(**aggregations)
    period_rows[constants.TARGET_COLUMN] = period_rows[PERIOD_TARGET_COLUMN]
    period_rows = add_period_history_features(period_rows)

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

    feature_columns = [
        column
        for column in period_rows.columns
        if column
        not in {
            constants.TIMESTAMP_COLUMN,
            "period_end",
            PERIOD_TARGET_COLUMN,
            constants.TARGET_COLUMN,
        }
    ]
    period_rows = period_rows.dropna(subset=[PERIOD_TARGET_COLUMN]).reset_index(drop=True)
    return ModellingDataset(
        table=period_rows,
        target_column=PERIOD_TARGET_COLUMN,
        feature_columns=feature_columns,
        target_option="B",
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

    if target_option == "A":
        return build_hourly_dataset(feature_table, model_name, feature_mode, period_days)
    if target_option == "B":
        return build_period_average_dataset(feature_table, period_days, block)
    raise ValueError(f"Unsupported target option: {target_option}")
