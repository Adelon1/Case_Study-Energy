"""Common forecast error metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline_helpers.modelling import constants


def mean_absolute_error_for_mask(errors: pd.Series, mask: pd.Series) -> float:
    """Return MAE on a subset, or NaN if the subset is empty."""

    if not mask.any():
        return float("nan")
    return float(errors.loc[mask].abs().mean())


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int]:
    """Calculate standard regression metrics for one validation window."""

    errors = y_pred - y_true
    top_decile_threshold = y_true.quantile(0.90)
    bottom_decile_threshold = y_true.quantile(0.10)
    top_decile_mask = y_true >= top_decile_threshold
    bottom_decile_mask = y_true <= bottom_decile_threshold
    negative_price_mask = y_true < 0
    scarcity_price_mask = y_true >= constants.SCARCITY_PRICE_THRESHOLD_EUR_PER_MWH

    return {
        "n_predictions": int(errors.notna().sum()),  # Number of forecasted hours.
        "mae": float(errors.abs().mean()),  # Average absolute hourly price error.
        "rmse": float(np.sqrt((errors**2).mean())),  # Error metric that punishes large misses.
        "bias": float(errors.mean()),  # Average signed error; positive means overprediction.
        "top_decile_price_threshold": float(top_decile_threshold),  # 90th percentile true price.
        "bottom_decile_price_threshold": float(bottom_decile_threshold),  # 10th percentile true price.
        "n_top_decile_hours": int(top_decile_mask.sum()),  # Count of high-price stress hours.
        "n_bottom_decile_hours": int(bottom_decile_mask.sum()),  # Count of low-price stress hours.
        "n_negative_price_hours": int(negative_price_mask.sum()),  # Count of negative-price hours.
        "n_scarcity_price_hours": int(scarcity_price_mask.sum()),  # Count of fixed-threshold scarcity hours.
        "top_decile_mae": mean_absolute_error_for_mask(errors, top_decile_mask),  # MAE on high-price hours.
        "bottom_decile_mae": mean_absolute_error_for_mask(errors, bottom_decile_mask),  # MAE on low-price hours.
        "negative_price_mae": mean_absolute_error_for_mask(errors, negative_price_mask),  # MAE when price is below zero.
        "scarcity_price_mae": mean_absolute_error_for_mask(errors, scarcity_price_mask),  # MAE above scarcity threshold.
    }
