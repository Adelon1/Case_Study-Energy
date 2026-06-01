"""Common forecast error metrics."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

constants = importlib.import_module("pipeline_helpers.02_modelling.00_constants")


def mean_absolute_error_for_mask(errors: pd.Series, mask: pd.Series) -> float:
    """Return MAE on a subset, or NaN if the subset is empty."""

    if not mask.any():
        return float("nan")
    return float(errors.loc[mask].abs().mean())


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Calculate forecast error metrics for one validation window.

    Six metrics, each carrying distinct information:
      - mae/rmse: overall accuracy (rmse punishes large misses harder).
      - bias: systematic over- or under-prediction.
      - top_decile_mae / scarcity_price_mae: accuracy in high-price stress hours.
      - bottom_decile_mae: accuracy in low- and negative-price hours.
    """

    errors = y_pred - y_true
    top_decile_mask = y_true >= y_true.quantile(0.90)
    bottom_decile_mask = y_true <= y_true.quantile(0.10)
    scarcity_price_mask = y_true >= constants.SCARCITY_PRICE_THRESHOLD_EUR_PER_MWH

    return {
        "mae": float(errors.abs().mean()),  # Average absolute hourly price error.
        "rmse": float(np.sqrt((errors**2).mean())),  # Error metric that punishes large misses.
        "bias": float(errors.mean()),  # Average signed error; positive means overprediction.
        "top_decile_mae": mean_absolute_error_for_mask(errors, top_decile_mask),  # MAE on the highest-price 10% of hours.
        "bottom_decile_mae": mean_absolute_error_for_mask(errors, bottom_decile_mask),  # MAE on the lowest-price 10% of hours.
        "scarcity_price_mae": mean_absolute_error_for_mask(errors, scarcity_price_mask),  # MAE above the scarcity threshold.
    }
