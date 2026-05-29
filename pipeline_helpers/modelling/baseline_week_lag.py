"""Seasonal naive baseline model.

The prediction is the observed day-ahead price from the same hour one week
earlier. The feature-building step already creates this as ``price_lag_168``.
"""

from __future__ import annotations

import pandas as pd


MODEL_NAME = "baseline_week_lag"
PARAM_GRID = [
    {"lag_hours": 24},
    {"lag_hours": 48},
    {"lag_hours": 168},
]


def train(train_data: pd.DataFrame, params: dict[str, object]) -> None:
    """The weekly lag baseline has no fitted parameters."""

    return None


def predict(model_state: None, test_data: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """Predict prices with the configured lag column."""

    lag_hours = int(params["lag_hours"])
    lag_column = f"price_lag_{lag_hours}"
    if lag_column not in test_data.columns:
        raise ValueError(f"Missing lag column required by baseline: {lag_column}")
    return test_data[lag_column]
