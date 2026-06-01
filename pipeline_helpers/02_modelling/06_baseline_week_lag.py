"""Naive baseline models for hourly and period-average targets."""

from __future__ import annotations

import pandas as pd


MODEL_NAME = "baseline_week_lag"


def build_param_grid(
    target_option: str = "hourly",
    **_unused_options,
) -> list[dict[str, object]]:
    """Build lag baselines measured in rows of the modelling table.

    The grid depends on the target option, because a row means a different span
    of time in each table: for hourly targets a sensible lag baseline is the
    same hour yesterday / two days ago / last week (24/48/168 rows), while for
    period-average targets a row is one whole delivery period, so the natural
    baselines are the previous few periods (1/2/4/7/12 rows).
    """

    if target_option == "period_average":
        return [
            {"lag_rows": 1},
            {"lag_rows": 2},
            {"lag_rows": 4},
            {"lag_rows": 7},
            {"lag_rows": 12},
        ]
    return [
        {"lag_rows": 24},
        {"lag_rows": 48},
        {"lag_rows": 168},
    ]


def train(train_data: pd.DataFrame, params: dict[str, object]) -> None:
    """The weekly lag baseline has no fitted parameters."""

    return None


def predict(model_state: None, test_data: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """Predict target values with a generic row-lag column."""

    lag_rows = int(params["lag_rows"])
    baseline_column = f"target_lag_{lag_rows}"
    if baseline_column not in test_data.columns:
        raise ValueError(f"Missing target lag column required by baseline: {baseline_column}")
    return test_data[baseline_column]
