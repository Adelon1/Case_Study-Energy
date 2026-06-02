"""Robust sparse linear benchmark using RANSAC around LASSO.

RANSAC repeatedly fits LASSO on random training subsets, identifies rows that
look like inliers, and refits on the best inlier set. This can improve the
normal-price fit when a few training spikes dominate the loss, but it can also
throw away real scarcity or negative-price regimes. The tail metrics decide
whether that trade-off is acceptable.

Model contract functions called by validation/window prediction:
    ``build_param_grid(...)``
    ``train(...)``
    ``predict(...)``
    ``output_folder_name(...)``
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib

import pandas as pd
from sklearn.linear_model import Lasso, RANSACRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

model_support = importlib.import_module("pipeline_helpers.02_modelling.05_model_support")


MODEL_NAME = "ransac_lasso_model"
HOUR_COLUMN = "local_hour"


@dataclass(frozen=True)
class RansacLassoState:
    """Fitted RANSAC-LASSO models and the feature list they expect."""

    hourly_models: dict[int, Pipeline]
    pooled_model: Pipeline | None
    feature_columns: list[str]
    target_transform: str


# ---------------------------------------------------------------------------
# Model contract: train and predict
# ---------------------------------------------------------------------------


def train(train_data: pd.DataFrame, params: dict[str, object]) -> RansacLassoState:
    """Fit one RANSAC-LASSO model per hour, or one pooled model without hour rows."""

    target_transform = str(params.get("target_transform", "raw"))
    if target_transform not in model_support.SUPPORTED_TARGET_TRANSFORMS:
        raise ValueError(f"Unsupported target transform: {target_transform}")

    target_column = model_support.resolve_target_column(params)
    feature_columns = model_support.resolve_feature_columns(train_data, params)

    if HOUR_COLUMN not in train_data.columns:
        model = fit_one_model(train_data, params, feature_columns, target_column, target_transform)
        return RansacLassoState(
            hourly_models={},
            pooled_model=model,
            feature_columns=feature_columns,
            target_transform=target_transform,
        )

    hourly_models: dict[int, Pipeline] = {}
    for hour in range(24):
        hour_rows = train_data.loc[train_data[HOUR_COLUMN] == hour]
        hourly_models[hour] = fit_one_model(
            hour_rows,
            params,
            feature_columns,
            target_column,
            target_transform,
            empty_message=f"RANSAC-LASSO has no complete training rows for hour {hour}.",
        )

    return RansacLassoState(
        hourly_models=hourly_models,
        pooled_model=None,
        feature_columns=feature_columns,
        target_transform=target_transform,
    )


def predict(
    model_state: RansacLassoState,
    test_data: pd.DataFrame,
    _params: dict[str, object],
) -> pd.Series:
    """Predict rows with complete selected features."""

    predictions = pd.Series(index=test_data.index, dtype=float)
    if test_data.empty:
        return predictions

    if model_state.pooled_model is not None:
        predict_into(predictions, model_state, model_state.pooled_model, test_data)
        return predictions

    if HOUR_COLUMN not in test_data.columns:
        raise ValueError(f"RANSAC-LASSO needs the hourly split column: {HOUR_COLUMN}")

    for hour, model in model_state.hourly_models.items():
        hour_rows = test_data.loc[test_data[HOUR_COLUMN] == hour]
        if not hour_rows.empty:
            predict_into(predictions, model_state, model, hour_rows)
    return predictions


# ---------------------------------------------------------------------------
# Model contract: hyperparameters and naming
# ---------------------------------------------------------------------------


def build_param_grid(
    target_transform: str = "raw",
    **_unused_options,
) -> list[dict[str, object]]:
    """Return a small grid for the robust sparse benchmark."""

    if target_transform not in model_support.SUPPORTED_TARGET_TRANSFORMS:
        raise ValueError(f"Unsupported target transform: {target_transform}")
    return [
        {
            "target_transform": target_transform,
            "alpha": alpha,
            "min_samples": 0.5,
            "residual_threshold": None,
            "max_trials": 100,
        }
        for alpha in [1.0]

    ]


def output_folder_name(target_transform: str = "raw", **_unused_options) -> str:
    """Name output folders by the target transform."""

    if target_transform not in model_support.SUPPORTED_TARGET_TRANSFORMS:
        raise ValueError(f"Unsupported target transform: {target_transform}")
    return f"{MODEL_NAME}_{target_transform}"


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def fit_one_model(
    train_data: pd.DataFrame,
    params: dict[str, object],
    feature_columns: list[str],
    target_column: str,
    target_transform: str,
    empty_message: str = "RANSAC-LASSO model has no complete training rows.",
) -> Pipeline:
    """Fit one scaled RANSAC-LASSO pipeline."""

    rows = train_data.dropna(subset=[target_column, *feature_columns])
    if rows.empty:
        raise ValueError(empty_message)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "regressor",
                RANSACRegressor(
                    estimator=Lasso(
                        alpha=float(params["alpha"]),
                        max_iter=20000,
                        tol=1e-4,
                        selection="random",
                        random_state=0,
                    ),
                    min_samples=float(params["min_samples"]),
                    residual_threshold=params["residual_threshold"],
                    max_trials=int(params["max_trials"]),
                    random_state=0,
                ),
            ),
        ]
    )
    model.fit(
        rows[feature_columns],
        model_support.transform_target(rows[target_column], target_transform),
    )
    return model


def predict_into(
    predictions: pd.Series,
    model_state: RansacLassoState,
    model: Pipeline,
    rows: pd.DataFrame,
) -> None:
    """Predict complete rows with one fitted model."""

    complete = rows[model_state.feature_columns].notna().all(axis=1)
    if not complete.any():
        return

    usable_index = rows.loc[complete].index
    raw_prediction = model.predict(rows.loc[usable_index, model_state.feature_columns])
    predictions.loc[usable_index] = model_support.inverse_transform_prediction(
        raw_prediction,
        model_state.target_transform,
    )
