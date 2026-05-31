"""Create modelling features from the clean hourly ENTSO-E dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_helpers.entsoe_data import constants


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


def add_calendar_features(table: pd.DataFrame) -> pd.DataFrame:
    """Add UTC and German local calendar features.

    UTC remains the canonical timestamp. Local calendar features are added for
    modelling because German power demand, solar shape, and weekday behaviour
    follow German market time. Full daily lag curves remain UTC-based elsewhere
    to avoid DST days with 23 or 25 local hours.
    """

    features = table.copy()
    timestamp_utc = pd.to_datetime(features["timestamp_utc"], utc=True)
    timestamp_local = timestamp_utc.dt.tz_convert(constants.GERMANY_MARKET_TIMEZONE)

    features["hour"] = timestamp_utc.dt.hour
    features["weekday"] = timestamp_utc.dt.weekday
    features["month"] = timestamp_utc.dt.month
    features["day_of_year"] = timestamp_utc.dt.dayofyear
    features["is_weekend"] = features["weekday"].isin([5, 6]).astype(int)

    features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
    features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
    features["weekday_sin"] = np.sin(2 * np.pi * features["weekday"] / 7)
    features["weekday_cos"] = np.cos(2 * np.pi * features["weekday"] / 7)
    features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
    features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)
    features["day_of_year_sin"] = np.sin(2 * np.pi * features["day_of_year"] / 366)
    features["day_of_year_cos"] = np.cos(2 * np.pi * features["day_of_year"] / 366)

    features["local_hour"] = timestamp_local.dt.hour
    features["local_weekday"] = timestamp_local.dt.weekday
    features["local_month"] = timestamp_local.dt.month
    features["local_day_of_year"] = timestamp_local.dt.dayofyear
    features["local_is_weekend"] = features["local_weekday"].isin([5, 6]).astype(int)

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


def add_price_lag_features(table: pd.DataFrame) -> pd.DataFrame:
    """Add price lag and rolling-price features.

    Rolling price features are shifted by 24 hours, not 1 hour. This means a
    full-day day-ahead forecast never uses actual prices from another hour of
    the delivery day.
    """

    features = table.copy()
    if PRICE_COLUMN not in features.columns:
        raise ValueError(f"Required target column missing: {PRICE_COLUMN}")

    price = features[PRICE_COLUMN]
    features["price_lag_24"] = price.shift(24)
    features["price_lag_48"] = price.shift(48)
    features["price_lag_168"] = price.shift(168)
    features["price_lag_336"] = price.shift(336)
    features["price_lag_720"] = price.shift(720)

    shifted_price = price.shift(24)
    features["price_rolling_mean_24"] = shifted_price.rolling(24).mean()
    features["price_rolling_std_24"] = shifted_price.rolling(24).std()
    features["price_rolling_mean_168"] = shifted_price.rolling(168).mean()
    features["price_rolling_std_168"] = shifted_price.rolling(168).std()

    return features


def add_daily_price_curve_lag_features(table: pd.DataFrame) -> pd.DataFrame:
    """Add previous UTC daily price shapes for LEAR-style intraday dependence."""

    features = table.copy()
    timestamp_utc = pd.to_datetime(features["timestamp_utc"], utc=True)
    features["utc_date_for_lags"] = timestamp_utc.dt.date

    daily_price_curve = features.pivot_table(
        index="utc_date_for_lags",
        columns="hour",
        values=PRICE_COLUMN,
        aggfunc="last",
    ).reindex(columns=range(24))

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


def build_feature_table(clean_hourly_dataset: pd.DataFrame) -> pd.DataFrame:
    """Build the modelling-ready feature table from clean hourly data."""

    features = clean_hourly_dataset.copy()
    features["timestamp_utc"] = pd.to_datetime(features["timestamp_utc"], utc=True)
    features = features.sort_values("timestamp_utc").reset_index(drop=True)
    features = add_fundamental_features(features)
    features = add_calendar_features(features)
    features = add_price_lag_features(features)
    features = add_daily_price_curve_lag_features(features)

    required_lag_columns = [
        "price_lag_24",
        "price_lag_48",
        "price_lag_168",
        "price_rolling_mean_24",
        "price_rolling_mean_168",
        "price_d1_h00",
        "price_d2_h00",
        "price_d3_h00",
        "price_d7_h00",
        "price_d1_mean",
        "price_d7_mean",
    ]
    features = features.dropna(subset=required_lag_columns).reset_index(drop=True)
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
