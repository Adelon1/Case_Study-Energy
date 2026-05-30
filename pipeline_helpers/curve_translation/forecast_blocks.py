"""Aggregate hourly forecasts into curve-relevant delivery blocks."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pipeline_helpers.curve_translation import constants


@dataclass(frozen=True)
class DeliveryPeriod:
    """Half-open UTC delivery period used for block aggregation."""

    start_utc: pd.Timestamp
    end_utc: pd.Timestamp


@dataclass(frozen=True)
class BlockValues:
    """Forecast averages for common power trading blocks."""

    baseload: float
    peakload: float
    offpeak: float
    peak_base_spread: float
    row_count: int
    predicted_row_count: int


def parse_utc_period(start: str, end: str) -> DeliveryPeriod:
    """Parse YYYY-MM-DD dates into a half-open UTC delivery period."""

    return DeliveryPeriod(
        start_utc=pd.Timestamp(start, tz="UTC"),
        end_utc=pd.Timestamp(end, tz="UTC"),
    )


def resolve_named_period(predictions: pd.DataFrame, period_name: str) -> DeliveryPeriod:
    """Resolve prompt-like names from the first available prediction timestamp."""

    timestamps = pd.to_datetime(predictions["timestamp_utc"], utc=True)
    first_day = timestamps.min().normalize()

    if period_name == "next_week":
        start = first_day
        return DeliveryPeriod(start_utc=start, end_utc=start + pd.Timedelta(days=7))

    if period_name == "next_month":
        start = first_day.replace(day=1)
        if first_day.day != 1:
            start = start + pd.DateOffset(months=1)
        return DeliveryPeriod(start_utc=start, end_utc=start + pd.DateOffset(months=1))

    raise ValueError(f"Unsupported named period: {period_name}")


def filter_delivery_period(table: pd.DataFrame, period: DeliveryPeriod) -> pd.DataFrame:
    """Keep rows inside the requested delivery period."""

    timestamps = pd.to_datetime(table["timestamp_utc"], utc=True)
    mask = (timestamps >= period.start_utc) & (timestamps < period.end_utc)
    return table.loc[mask].copy()


def peakload_mask(timestamps: pd.Series) -> pd.Series:
    """Return weekday daytime rows in German local market time."""

    local = pd.to_datetime(timestamps, utc=True).dt.tz_convert(constants.MARKET_TIMEZONE)
    is_weekday = local.dt.weekday < 5
    is_peak_hour = (
        (local.dt.hour >= constants.PEAK_START_HOUR)
        & (local.dt.hour < constants.PEAK_END_HOUR)
    )
    return is_weekday & is_peak_hour


def calculate_block_values(predictions: pd.DataFrame) -> BlockValues:
    """Calculate baseload, peakload, offpeak, and peak/base spread."""

    complete_predictions = predictions.dropna(subset=["y_pred"]).copy()
    if complete_predictions.empty:
        raise ValueError("No non-null predictions available for the requested period.")

    peak_mask = peakload_mask(complete_predictions["timestamp_utc"])
    peak_predictions = complete_predictions.loc[peak_mask, "y_pred"]
    offpeak_predictions = complete_predictions.loc[~peak_mask, "y_pred"]

    baseload = float(complete_predictions["y_pred"].mean())
    peakload = float(peak_predictions.mean()) if not peak_predictions.empty else float("nan")
    offpeak = float(offpeak_predictions.mean()) if not offpeak_predictions.empty else float("nan")

    return BlockValues(
        baseload=baseload,
        peakload=peakload,
        offpeak=offpeak,
        peak_base_spread=peakload - baseload,
        row_count=len(predictions),
        predicted_row_count=len(complete_predictions),
    )
