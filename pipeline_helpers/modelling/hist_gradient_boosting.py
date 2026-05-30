"""Nonlinear boosted-tree price model using sklearn histogram gradient boosting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from pipeline_helpers.modelling import constants


MODEL_NAME = "hist_gradient_boosting"
SUPPORTED_TARGET_TRANSFORMS = {"raw", "asinh"}

FEATURE_COLUMNS = [
    "load_forecast_mw",
    "solar_forecast_mw",
    "wind_onshore_forecast_mw",
    "wind_offshore_forecast_mw",
    "wind_total_forecast_mw",
    "renewable_total_forecast_mw",
    "residual_load_forecast_mw",
    "wind_share_of_load",
    "solar_share_of_load",
    "renewable_share_of_load",
    "hour",
    "weekday",
    "month",
    "day_of_year",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "price_lag_24",
    "price_lag_48",
    "price_lag_168",
    "price_rolling_mean_24",
    "price_rolling_std_24",
    "price_rolling_mean_168",
    "price_rolling_std_168",
]

for day_lag in [1, 2, 7]:
    FEATURE_COLUMNS.extend(
        f"price_d{day_lag}_h{hour:02d}"
        for hour in range(24)
    )

for day_lag in [1, 7]:
    FEATURE_COLUMNS.extend(
        [
            f"price_d{day_lag}_min",
            f"price_d{day_lag}_max",
            f"price_d{day_lag}_mean",
        ]
    )


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
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 30,
            "l2_regularization": 0.0,
        },
        {
            "target_transform": target_transform,
            "learning_rate": 0.05,
            "max_iter": 500,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 50,
            "l2_regularization": 0.1,
        },
        {
            "target_transform": target_transform,
            "learning_rate": 0.03,
            "max_iter": 600,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 50,
            "l2_regularization": 0.1,
        },
        {
            "target_transform": target_transform,
            "learning_rate": 0.08,
            "max_iter": 300,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 80,
            "l2_regularization": 0.3,
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


def available_feature_columns(table: pd.DataFrame) -> list[str]:
    """Return configured feature columns, failing loudly if any are missing."""

    missing_columns = [
        column for column in FEATURE_COLUMNS if column not in table.columns
    ]
    if missing_columns:
        missing_preview = ", ".join(missing_columns[:10])
        if len(missing_columns) > 10:
            missing_preview = f"{missing_preview}, ..."
        raise ValueError(
            "Configured boosted-tree features are missing from the feature table: "
            f"{missing_preview}"
        )
    return FEATURE_COLUMNS.copy()


def train(train_data: pd.DataFrame, params: dict[str, object]) -> BoostedTreeModelState:
    """Fit one pooled nonlinear model across all delivery hours."""

    target_transform = str(params["target_transform"])
    feature_columns = available_feature_columns(train_data)
    modelling_data = train_data.dropna(subset=[constants.TARGET_COLUMN, *feature_columns])
    if modelling_data.empty:
        raise ValueError("Boosted-tree model has no complete training rows.")

    model = HistGradientBoostingRegressor(
        learning_rate=float(params["learning_rate"]),
        max_iter=int(params["max_iter"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        l2_regularization=float(params["l2_regularization"]),
        loss="squared_error",
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
    """Predict all complete test rows with the pooled boosted-tree model."""

    predictions = pd.Series(index=test_data.index, dtype=float)
    complete_feature_mask = test_data[model_state.feature_columns].notna().all(axis=1)
    if not complete_feature_mask.any():
        return predictions

    prediction_index = test_data.loc[complete_feature_mask].index
    transformed_prediction = model_state.model.predict(
        test_data.loc[complete_feature_mask, model_state.feature_columns]
    )
    predictions.loc[prediction_index] = inverse_transform_prediction(
        transformed_prediction,
        model_state.target_transform,
    )
    return predictions
