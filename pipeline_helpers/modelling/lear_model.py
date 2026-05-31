"""LEAR-style regularised linear model.

This model estimates 24 separate hourly regressions. Each hour gets its own
LASSO or ElasticNet model, using the same feature set but different fitted
coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline_helpers.modelling import constants
from pipeline_helpers.modelling.feature_sets import get_hourly_feature_columns


MODEL_NAME = "lear_model"
HOUR_MODEL_COLUMN = "local_hour"
SUPPORTED_REGULARIZATION = {"lasso", "elasticnet", "ridge"}
SUPPORTED_TARGET_TRANSFORMS = {"raw", "asinh"}


@dataclass(frozen=True)
class LearModelState:
    """Fitted regularised models and the feature columns they expect."""

    hourly_models: dict[int, Pipeline]
    pooled_model: Pipeline | None
    feature_columns: list[str]
    target_transform: str
    n_train_rows_by_hour: dict[int, int]
    fitted_hours: list[int]


def build_param_grid(
    regularization: str = "lasso",
    target_transform: str = "raw",
) -> list[dict[str, object]]:
    """Build LEAR hyperparameter grid from user-level model choices."""

    validate_model_choices(regularization, target_transform)

    if regularization in {"lasso", "ridge"}:
        return [
            {
                "regularization": regularization,
                "target_transform": target_transform,
                "alpha": alpha,
            }
            for alpha in constants.LEAR_ALPHA_GRID
        ]

    valid_l1_ratios = [
        l1_ratio
        for l1_ratio in constants.LEAR_L1_RATIO_GRID
        if 0 < l1_ratio < 1
    ]
    if not valid_l1_ratios:
        raise ValueError("ElasticNet needs at least one l1_ratio strictly between 0 and 1.")

    return [
        {
            "regularization": "elasticnet",
            "target_transform": target_transform,
            "alpha": alpha,
            "l1_ratio": l1_ratio,
        }
        for alpha in constants.LEAR_ALPHA_GRID
        for l1_ratio in valid_l1_ratios
    ]


def output_folder_name(
    regularization: str = "lasso",
    target_transform: str = "raw",
) -> str:
    """Name output folders by model family and target transform."""

    validate_model_choices(regularization, target_transform)
    return f"{MODEL_NAME}_{regularization}_{target_transform}"


def validate_model_choices(regularization: str, target_transform: str) -> None:
    """Validate user-level LEAR choices."""

    if regularization not in SUPPORTED_REGULARIZATION:
        raise ValueError(f"Unsupported regularization: {regularization}")
    if target_transform not in SUPPORTED_TARGET_TRANSFORMS:
        raise ValueError(f"Unsupported target transform: {target_transform}")


def transform_target(y: pd.Series, target_transform: str) -> np.ndarray:
    """Transform the training target."""

    if target_transform == "raw":
        return y.to_numpy()
    if target_transform == "asinh":
        return np.arcsinh(y.to_numpy())
    raise ValueError(f"Unsupported target transform: {target_transform}")


def inverse_transform_prediction(y_pred: np.ndarray, target_transform: str) -> np.ndarray:
    """Map model predictions back to EUR/MWh."""

    if target_transform == "raw":
        return y_pred
    if target_transform == "asinh":
        return np.sinh(y_pred)
    raise ValueError(f"Unsupported target transform: {target_transform}")


def selected_feature_columns(train_data: pd.DataFrame, params: dict[str, object]) -> list[str]:
    """Return externally selected features or the default day-ahead feature set."""

    feature_columns = params.get("_feature_columns") or params.get("feature_columns")
    if feature_columns is None:
        feature_columns = get_hourly_feature_columns(
            train_data,
            model_name=MODEL_NAME,
            feature_mode="day_ahead_full",
            period_days=1,
        )
    feature_columns = list(feature_columns)
    missing_columns = [column for column in feature_columns if column not in train_data.columns]
    if missing_columns:
        missing_preview = ", ".join(missing_columns[:10])
        if len(missing_columns) > 10:
            missing_preview = f"{missing_preview}, ..."
        raise ValueError(
            "Configured LEAR features are missing from the feature table: "
            f"{missing_preview}"
        )
    return feature_columns


def build_regressor(params: dict[str, object]):
    """Create the regularised linear estimator requested by params."""

    regularization = str(params["regularization"])
    alpha = float(params["alpha"])

    if regularization == "lasso":
        return Lasso(alpha=alpha, max_iter=50000, tol=1e-4, selection="cyclic")
    if regularization == "ridge":
        return Ridge(alpha=alpha)
    if regularization == "elasticnet":
        return ElasticNet(
            alpha=alpha,
            l1_ratio=float(params["l1_ratio"]),
            max_iter=50000,
            tol=1e-4,
            selection="cyclic",
        )
    raise ValueError(f"Unsupported regularization: {regularization}")


def train(train_data: pd.DataFrame, params: dict[str, object]) -> LearModelState:
    """Fit one regularised regression per hour, or one pooled model without hour rows."""

    target_transform = str(params["target_transform"])
    feature_columns = selected_feature_columns(train_data, params)
    if not feature_columns:
        raise ValueError("No configured LEAR feature columns are available.")

    if HOUR_MODEL_COLUMN not in train_data.columns:
        modelling_data = train_data.dropna(subset=[constants.TARGET_COLUMN, *feature_columns])
        if modelling_data.empty:
            raise ValueError("LEAR pooled model has no complete training rows.")
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", build_regressor(params)),
            ]
        )
        estimator.fit(
            modelling_data[feature_columns],
            transform_target(modelling_data[constants.TARGET_COLUMN], target_transform),
        )
        return LearModelState(
            hourly_models={},
            pooled_model=estimator,
            feature_columns=feature_columns,
            target_transform=target_transform,
            n_train_rows_by_hour={-1: len(modelling_data)},
            fitted_hours=[],
        )

    hourly_models: dict[int, Pipeline] = {}
    n_train_rows_by_hour: dict[int, int] = {}
    for hour in range(24):
        hour_data = train_data.loc[train_data[HOUR_MODEL_COLUMN] == hour]
        modelling_data = hour_data.dropna(subset=[constants.TARGET_COLUMN, *feature_columns])
        n_train_rows_by_hour[hour] = len(modelling_data)
        if modelling_data.empty:
            continue

        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", build_regressor(params)),
            ]
        )
        x_train = modelling_data[feature_columns]
        y_train = transform_target(modelling_data[constants.TARGET_COLUMN], target_transform)
        estimator.fit(x_train, y_train)
        hourly_models[hour] = estimator

    missing_hours = sorted(set(range(24)) - set(hourly_models))
    if missing_hours:
        raise ValueError(f"LEAR did not fit models for hours: {missing_hours}")

    return LearModelState(
        hourly_models=hourly_models,
        pooled_model=None,
        feature_columns=feature_columns,
        target_transform=target_transform,
        n_train_rows_by_hour=n_train_rows_by_hour,
        fitted_hours=sorted(hourly_models),
    )


def predict(
    model_state: LearModelState,
    test_data: pd.DataFrame,
    _params: dict[str, object],
) -> pd.Series:
    """Predict each test row with the model fitted for its delivery hour."""

    predictions = pd.Series(index=test_data.index, dtype=float)
    if model_state.pooled_model is not None:
        complete_feature_mask = test_data[model_state.feature_columns].notna().all(axis=1)
        if complete_feature_mask.any():
            prediction_index = test_data.loc[complete_feature_mask].index
            transformed_prediction = model_state.pooled_model.predict(
                test_data.loc[prediction_index, model_state.feature_columns]
            )
            predictions.loc[prediction_index] = inverse_transform_prediction(
                transformed_prediction,
                model_state.target_transform,
            )
        return predictions

    if HOUR_MODEL_COLUMN not in test_data.columns:
        raise ValueError(f"LEAR needs hourly split column: {HOUR_MODEL_COLUMN}")

    for hour, estimator in model_state.hourly_models.items():
        hour_mask = test_data[HOUR_MODEL_COLUMN] == hour
        if not hour_mask.any():
            continue

        hour_data = test_data.loc[hour_mask]
        complete_feature_mask = hour_data[model_state.feature_columns].notna().all(axis=1)
        if not complete_feature_mask.any():
            continue

        prediction_index = hour_data.loc[complete_feature_mask].index
        x_test = hour_data.loc[complete_feature_mask, model_state.feature_columns]
        transformed_prediction = estimator.predict(x_test)
        predictions.loc[prediction_index] = inverse_transform_prediction(
            transformed_prediction,
            model_state.target_transform,
        )

    return predictions
