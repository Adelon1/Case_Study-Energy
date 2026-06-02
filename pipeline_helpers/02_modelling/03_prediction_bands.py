"""Empirical forecast bands from out-of-sample residuals.

A point forecast alone says nothing about uncertainty. These helpers turn the
walk-forward validation residuals into a prediction band: for each delivery
hour we measure how far actual prices have historically fallen below and above
the forecast, then attach those offsets to future predictions. Peak hours get
wider bands than calm night hours because their residuals are wider.

The bands are leakage-safe in spirit: they are estimated only from realised
out-of-sample validation errors, never from the value being predicted.

Public entry points:
    ``residual_quantiles_by_hour(...)``
    ``add_prediction_bands(...)``
    ``band_coverage(...)``
"""

from __future__ import annotations

import importlib

import pandas as pd

constants = importlib.import_module("pipeline_helpers.02_modelling.00_constants")

HOUR_COLUMN = "local_hour"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def residual_quantiles_by_hour(
    predictions: pd.DataFrame,
    lower_quantile: float = constants.BAND_LOWER_QUANTILE,
    upper_quantile: float = constants.BAND_UPPER_QUANTILE,
) -> dict[object, tuple[float, float]]:
    """Return per-hour residual offsets ``(low, high)`` for the forecast band.

    The residual is ``y_true - y_pred``. When the predictions carry no hour
    column the whole sample shares a single ``None`` entry.
    """

    residuals = predictions["y_true"] - predictions["y_pred"]

    if HOUR_COLUMN not in predictions.columns:
        return {None: (residuals.quantile(lower_quantile), residuals.quantile(upper_quantile))}

    offsets: dict[object, tuple[float, float]] = {}
    for hour, hour_residuals in residuals.groupby(predictions[HOUR_COLUMN]):
        offsets[hour] = (
            float(hour_residuals.quantile(lower_quantile)),
            float(hour_residuals.quantile(upper_quantile)),
        )
    return offsets


def add_prediction_bands(
    predictions: pd.DataFrame,
    quantiles_by_hour: dict[object, tuple[float, float]],
) -> pd.DataFrame:
    """Attach ``y_pred_lower``/``y_pred_upper`` columns to a predictions frame."""

    result = predictions.copy()
    global_offsets = quantiles_by_hour.get(None)

    def offsets_for(hour: object) -> tuple[float, float]:
        if global_offsets is not None:
            return global_offsets
        return quantiles_by_hour.get(hour, (0.0, 0.0))

    if HOUR_COLUMN in result.columns and global_offsets is None:
        lower_offsets = result[HOUR_COLUMN].map(lambda hour: offsets_for(hour)[0])
        upper_offsets = result[HOUR_COLUMN].map(lambda hour: offsets_for(hour)[1])
    else:
        low, high = offsets_for(None)
        lower_offsets = low
        upper_offsets = high

    result["y_pred_lower"] = result["y_pred"] + lower_offsets
    result["y_pred_upper"] = result["y_pred"] + upper_offsets
    return result


def band_coverage(predictions: pd.DataFrame) -> float:
    """Share of actual prices that fall inside the forecast band."""

    inside = (predictions["y_true"] >= predictions["y_pred_lower"]) & (
        predictions["y_true"] <= predictions["y_pred_upper"]
    )
    return float(inside.mean())
