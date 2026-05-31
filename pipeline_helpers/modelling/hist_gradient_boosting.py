"""Nonlinear boosted-tree price model using sklearn histogram gradient boosting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from pipeline_helpers.modelling import constants
from pipeline_helpers.modelling.feature_sets import get_hourly_feature_columns


MODEL_NAME = "hist_gradient_boosting"
SUPPORTED_TARGET_TRANSFORMS = {"raw", "asinh"}


@dataclass(frozen=True)
class BoostedTreeModelState:
    """Fitted boosted-tree model and expected feature columns."""

    model: HistGradientBoostingRegressor
    feature_columns: list[str]
    target_transform: str


def build_param_grid(target_transform: str = "raw") -> list[dict[str, object]]:
    """Build a small, runtime-conscious boosted-tree hyperparameter grid."""

    validate_model_choices(target_transform)
    return [
        {
            "target_transform": target_transform,
            "loss": "absolute_error",
            "learning_rate": 0.03,
            "max_iter": 800,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
        },
        {
            "target_transform": target_transform,
            "loss": "absolute_error",
            "learning_rate": 0.05,
            "max_iter": 600,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 30,
            "l2_regularization": 0.0,
        },
        {
            "target_transform": target_transform,
            "loss": "absolute_error",
            "learning_rate": 0.05,
            "max_iter": 600,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 30,
            "l2_regularization": 0.1,
        },
        {
            "target_transform": target_transform,
            "loss": "squared_error",
            "learning_rate": 0.03,
            "max_iter": 800,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
        },
        {
            "target_transform": target_transform,
            "loss": "squared_error",
            "learning_rate": 0.05,
            "max_iter": 600,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 20,
            "l2_regularization": 0.1,
        },
    ]


def output_folder_name(target_transform: str = "raw") -> str:
    """Name output folders by model and target transform."""

    validate_model_choices(target_transform)
    return f"{MODEL_NAME}_{target_transform}"


def validate_model_choices(target_transform: str) -> None:
    """Validate user-level boosted-tree choices."""

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
            "Configured boosted-tree features are missing from the feature table: "
            f"{missing_preview}"
        )
    return feature_columns


def train(train_data: pd.DataFrame, params: dict[str, object]) -> BoostedTreeModelState:
    """Fit one pooled nonlinear model across all delivery hours."""

    target_transform = str(params["target_transform"])
    feature_columns = selected_feature_columns(train_data, params)
    modelling_data = train_data.dropna(subset=[constants.TARGET_COLUMN])
    if modelling_data.empty:
        raise ValueError("Boosted-tree model has no training rows with a target value.")

    model = HistGradientBoostingRegressor(
        loss=str(params.get("loss", "squared_error")),
        learning_rate=float(params["learning_rate"]),
        max_iter=int(params["max_iter"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        l2_regularization=float(params["l2_regularization"]),
        early_stopping=False,
        random_state=0,
    )
    model.fit(
        modelling_data[feature_columns],
        transform_target(modelling_data[constants.TARGET_COLUMN], target_transform),
    )
    return BoostedTreeModelState(
        model=model,
        feature_columns=feature_columns,
        target_transform=target_transform,
    )


def predict(
    model_state: BoostedTreeModelState,
    test_data: pd.DataFrame,
    _params: dict[str, object],
) -> pd.Series:
    """Predict all test rows with the pooled boosted-tree model."""

    predictions = pd.Series(index=test_data.index, dtype=float)
    if test_data.empty:
        return predictions

    transformed_prediction = model_state.model.predict(test_data[model_state.feature_columns])
    predictions.loc[test_data.index] = inverse_transform_prediction(
        transformed_prediction,
        model_state.target_transform,
    )
    return predictions
