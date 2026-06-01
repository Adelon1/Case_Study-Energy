"""Create modelling features from the clean hourly ENTSO-E dataset."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

constants = importlib.import_module("pipeline_helpers.01_entsoe_data.00_constants")


PRICE_COLUMN = "day_ahead_price_eur_per_mwh"


@dataclass(frozen=True)
class FeatureDataset:
    """Path and in-memory table produced by feature engineering."""

    path: Path
    table: pd.DataFrame


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series and return NaN where the denominator is zero."""

    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def add_fundamental_features(table: pd.DataFrame) -> pd.DataFrame:
    """Add residual load, total wind, and renewable share features when possible."""

    features = table.copy()

    if {"wind_onshore_forecast_mw", "wind_offshore_forecast_mw"}.issubset(features.columns):
        features["wind_total_forecast_mw"] = (
            features["wind_onshore_forecast_mw"] + features["wind_offshore_forecast_mw"]
        )

    if {"solar_forecast_mw", "wind_total_forecast_mw"}.issubset(features.columns):
        features["renewable_total_forecast_mw"] = (
            features["solar_forecast_mw"] + features["wind_total_forecast_mw"]
        )

    if {"load_forecast_mw", "solar_forecast_mw", "wind_total_forecast_mw"}.issubset(
        features.columns
    ):
        features["residual_load_forecast_mw"] = (
            features["load_forecast_mw"]
            - features["solar_forecast_mw"]
            - features["wind_total_forecast_mw"]
        )

    if {"wind_total_forecast_mw", "load_forecast_mw"}.issubset(features.columns):
        features["wind_share_of_load"] = safe_divide(
            features["wind_total_forecast_mw"],
            features["load_forecast_mw"],
        )

    if {"solar_forecast_mw", "load_forecast_mw"}.issubset(features.columns):
        features["solar_share_of_load"] = safe_divide(
            features["solar_forecast_mw"],
            features["load_forecast_mw"],
        )

    if {"renewable_total_forecast_mw", "load_forecast_mw"}.issubset(features.columns):
        features["renewable_share_of_load"] = safe_divide(
            features["renewable_total_forecast_mw"],
            features["load_forecast_mw"],
        )

    return features


def local_german_holiday_flag(timestamp_local: pd.Series) -> pd.Series:
    """Flag German nationwide public holidays on the local delivery date.

    Holidays are evaluated on the German local calendar date so the late-evening
    UTC hours of a holiday still map to the correct German day. Only nationwide
    holidays are used to avoid region-specific noise.
    """

    local_dates = timestamp_local.dt.date
    valid_years = timestamp_local.dt.year.dropna()
    if valid_years.empty:
        return pd.Series(0, index=timestamp_local.index, dtype=int)

    year_range = range(int(valid_years.min()), int(valid_years.max()) + 1)
    german_holidays = holidays.Germany(years=year_range)
    flags = [0 if date is None else int(date in german_holidays) for date in local_dates]
    return pd.Series(flags, index=timestamp_local.index, dtype=int)


def add_calendar_features(table: pd.DataFrame) -> pd.DataFrame:
    """Add German local calendar features.

    The canonical join key stays UTC, but every modelling calendar feature uses
    German market time because demand, the solar shape, weekday behaviour, and
    public holidays follow the local clock and calendar. UTC calendar columns
    are intentionally not produced: the models never used them. Full daily
    price-curve lags are still built on UTC hours elsewhere to keep 24 columns
    per day across 23- and 25-hour DST days.
    """

    features = table.copy()
    timestamp_utc = pd.to_datetime(features["timestamp_utc"], utc=True)
    timestamp_local = timestamp_utc.dt.tz_convert(constants.GERMANY_MARKET_TIMEZONE)

    features["local_hour"] = timestamp_local.dt.hour
    features["local_weekday"] = timestamp_local.dt.weekday
    features["local_month"] = timestamp_local.dt.month
    features["local_day_of_year"] = timestamp_local.dt.dayofyear
    features["is_holiday"] = local_german_holiday_flag(timestamp_local)

    features["local_hour_sin"] = np.sin(2 * np.pi * features["local_hour"] / 24)
    features["local_hour_cos"] = np.cos(2 * np.pi * features["local_hour"] / 24)
    features["local_weekday_sin"] = np.sin(2 * np.pi * features["local_weekday"] / 7)
    features["local_weekday_cos"] = np.cos(2 * np.pi * features["local_weekday"] / 7)
    features["local_month_sin"] = np.sin(2 * np.pi * features["local_month"] / 12)
    features["local_month_cos"] = np.cos(2 * np.pi * features["local_month"] / 12)
    features["local_day_of_year_sin"] = np.sin(
        2 * np.pi * features["local_day_of_year"] / 366
    )
    features["local_day_of_year_cos"] = np.cos(
        2 * np.pi * features["local_day_of_year"] / 366
    )

    weekday_dummies = pd.get_dummies(
        features["local_weekday"],
        prefix="local_weekday",
        dtype=int,
    )
    features = pd.concat([features, weekday_dummies], axis=1)
    for weekday in range(7):
        column = f"local_weekday_{weekday}"
        if column not in features.columns:
            features[column] = 0

    return features


def add_daily_price_curve_lag_features(table: pd.DataFrame) -> pd.DataFrame:
    """Add previous UTC daily price shapes for LEAR-style intraday dependence."""

    features = table.copy()
    timestamp_utc = pd.to_datetime(features["timestamp_utc"], utc=True)
    features["utc_date_for_lags"] = timestamp_utc.dt.date

    daily_price_curve = (
        features.assign(utc_hour_for_lags=timestamp_utc.dt.hour)
        .pivot_table(
            index="utc_date_for_lags",
            columns="utc_hour_for_lags",
            values=PRICE_COLUMN,
            aggfunc="last",
        )
        .reindex(columns=range(24))
    )

    for day_lag in [1, 2, 3, 7]:
        lagged_curve = daily_price_curve.shift(day_lag)
        lagged_curve.columns = [
            f"price_d{day_lag}_h{hour:02d}"
            for hour in lagged_curve.columns
        ]
        features = features.merge(
            lagged_curve,
            left_on="utc_date_for_lags",
            right_index=True,
            how="left",
        )

    for day_lag in [1, 7]:
        shifted_daily_price = daily_price_curve.shift(day_lag)
        features = features.merge(
            pd.DataFrame(
                {
                    f"price_d{day_lag}_min": shifted_daily_price.min(axis=1),
                    f"price_d{day_lag}_max": shifted_daily_price.max(axis=1),
                    f"price_d{day_lag}_mean": shifted_daily_price.mean(axis=1),
                }
            ),
            left_on="utc_date_for_lags",
            right_index=True,
            how="left",
        )

    return features.drop(columns=["utc_date_for_lags"])


def reindex_to_full_hourly_grid(table: pd.DataFrame) -> pd.DataFrame:
    """Reindex to a gap-free hourly UTC grid so row shifts equal real hours.

    The combine stage can drop rows where a forecast driver was still missing.
    Once a row is missing, ``shift(n)`` no longer means ``n`` hours, which would
    silently misalign every price lag and rolling feature. Re-inserting the
    missing hours as empty rows keeps all later shifts on a true hourly clock;
    those empty rows are removed again by the final ``dropna``.
    """

    table = table.copy()
    table["timestamp_utc"] = pd.to_datetime(table["timestamp_utc"], utc=True)
    full_range = pd.date_range(
        table["timestamp_utc"].min(),
        table["timestamp_utc"].max(),
        freq="h",
    )
    return (
        table.set_index("timestamp_utc")
        .reindex(full_range)
        .rename_axis("timestamp_utc")
        .reset_index()
    )


def build_feature_table(clean_hourly_dataset: pd.DataFrame) -> pd.DataFrame:
    """Build the modelling-ready feature table from clean hourly data."""

    features = clean_hourly_dataset.copy()
    features["timestamp_utc"] = pd.to_datetime(features["timestamp_utc"], utc=True)
    features = features.sort_values("timestamp_utc").reset_index(drop=True)
    features = reindex_to_full_hourly_grid(features)
    features = add_fundamental_features(features)
    features = add_calendar_features(features)
    features = add_daily_price_curve_lag_features(features)

    required_columns = [
        PRICE_COLUMN,
        "price_d1_h00",
        "price_d2_h00",
        "price_d3_h00",
        "price_d7_h00",
        "price_d1_mean",
        "price_d7_mean",
    ]
    features = features.dropna(subset=required_columns).reset_index(drop=True)
    return features


def write_feature_dataset(
    clean_hourly_csv_path: str | Path,
    output_path: str | Path,
) -> FeatureDataset:
    """Read the clean hourly CSV, build features, and write the feature CSV."""

    clean_hourly_csv_path = Path(clean_hourly_csv_path)
    output_path = Path(output_path)
    clean_hourly_dataset = pd.read_csv(clean_hourly_csv_path)
    feature_table = build_feature_table(clean_hourly_dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(output_path, index=False)
    return FeatureDataset(path=output_path, table=feature_table)
