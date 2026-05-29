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
    """Add calendar features based on German local market time."""

    features = table.copy()
    local_time = pd.to_datetime(features["timestamp_utc"], utc=True).dt.tz_convert(
        constants.GERMANY_MARKET_TIMEZONE
    )

    features["hour"] = local_time.dt.hour
    features["weekday"] = local_time.dt.weekday
    features["month"] = local_time.dt.month
    features["day_of_year"] = local_time.dt.dayofyear
    features["is_weekend"] = features["weekday"].isin([5, 6]).astype(int)

    features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
    features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
    features["weekday_sin"] = np.sin(2 * np.pi * features["weekday"] / 7)
    features["weekday_cos"] = np.cos(2 * np.pi * features["weekday"] / 7)
    features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
    features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)

    return features


def add_price_lag_features(table: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe price lag and rolling-price features."""

    features = table.copy()
    if PRICE_COLUMN not in features.columns:
        raise ValueError(f"Required target column missing: {PRICE_COLUMN}")

    price = features[PRICE_COLUMN]
    features["price_lag_24"] = price.shift(24)
    features["price_lag_48"] = price.shift(48)
    features["price_lag_168"] = price.shift(168)

    shifted_price = price.shift(1)
    features["price_rolling_mean_24"] = shifted_price.rolling(24).mean()
    features["price_rolling_std_24"] = shifted_price.rolling(24).std()
    features["price_rolling_mean_168"] = shifted_price.rolling(168).mean()
    features["price_rolling_std_168"] = shifted_price.rolling(168).std()

    return features


def build_feature_table(clean_hourly_dataset: pd.DataFrame) -> pd.DataFrame:
    """Build the modelling-ready feature table from clean hourly data."""

    features = clean_hourly_dataset.copy()
    features["timestamp_utc"] = pd.to_datetime(features["timestamp_utc"], utc=True)
    features = features.sort_values("timestamp_utc").reset_index(drop=True)
    features = add_fundamental_features(features)
    features = add_calendar_features(features)
    features = add_price_lag_features(features)

    required_lag_columns = [
        "price_lag_24",
        "price_lag_48",
        "price_lag_168",
        "price_rolling_mean_24",
        "price_rolling_mean_168",
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
