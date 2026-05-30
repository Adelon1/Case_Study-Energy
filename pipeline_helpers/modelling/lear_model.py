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


MODEL_NAME = "lear_model"
SUPPORTED_REGULARIZATION = {"lasso", "elasticnet", "ridge"}
SUPPORTED_TARGET_TRANSFORMS = {"raw", "asinh"}

FEATURE_COLUMNS = [
    "load_forecast_mw",
    "solar_forecast_mw",
    "wind_total_forecast_mw",
    "residual_load_forecast_mw",
    "wind_share_of_load",
    "solar_share_of_load",
    "renewable_share_of_load",
    "is_weekend",
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
class LearModelState:
    """Fitted hourly models and the feature columns they expect."""

    hourly_models: dict[int, Pipeline]
    feature_columns: list[str]
    target_transform: str


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
            "Configured LEAR features are missing from the feature table: "
            f"{missing_preview}"
        )
    return FEATURE_COLUMNS.copy()


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
    """Fit one regularised regression per delivery hour."""

    target_transform = str(params["target_transform"])
    feature_columns = available_feature_columns(train_data)
    if not feature_columns:
        raise ValueError("No configured LEAR feature columns are available.")

    hourly_models: dict[int, Pipeline] = {}
    for hour in range(24):
        hour_data = train_data.loc[train_data["hour"] == hour]
        modelling_data = hour_data.dropna(subset=[constants.TARGET_COLUMN, *feature_columns])
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

    if not hourly_models:
        raise ValueError("LEAR could not fit any hourly models.")

    return LearModelState(
        hourly_models=hourly_models,
        feature_columns=feature_columns,
        target_transform=target_transform,
    )


def predict(
    model_state: LearModelState,
    test_data: pd.DataFrame,
    _params: dict[str, object],
) -> pd.Series:
    """Predict each test row with the model fitted for its delivery hour."""

    predictions = pd.Series(index=test_data.index, dtype=float)

    for hour, estimator in model_state.hourly_models.items():
        hour_mask = test_data["hour"] == hour
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
