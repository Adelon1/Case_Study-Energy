"""LEAR-style regularised linear model.

LEAR fits one regularised linear regression per delivery hour. Every hour shares
the same leakage-safe feature set but learns its own coefficients and its own
regularisation strength: each hour selects its penalty by time-series
cross-validation, so volatile peak hours and calm night hours are tuned
independently instead of sharing one global alpha.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline_helpers.modelling import constants, model_support


MODEL_NAME = "lear_model"
HOUR_COLUMN = "local_hour"
SUPPORTED_REGULARIZATION = {"lasso", "elasticnet", "ridge"}


@dataclass(frozen=True)
class LearModelState:
    """Fitted pipelines plus the per-hour choices behind them."""

    hourly_models: dict[int, Pipeline]
    pooled_model: Pipeline | None
    feature_columns: list[str]
    target_transform: str
    alpha_by_hour: dict[int, float]
    fitted_hours: list[int]
    n_train_rows_by_hour: dict[int, int]


# --- Model contract: train and predict -------------------------------------


def train(train_data: pd.DataFrame, params: dict[str, object]) -> LearModelState:
    """Fit one cross-validated regression per hour, or one pooled model.

    Hourly rows (Option A) carry a ``local_hour`` column and get 24 independent
    models. Period-average rows (Option B) have no hour column and get a single
    pooled model.
    """

    regularization = str(params.get("regularization", "lasso"))
    target_transform = str(params.get("target_transform", "raw"))
    validate_model_choices(regularization, target_transform)

    target_column = model_support.resolve_target_column(params)
    feature_columns = model_support.resolve_feature_columns(train_data, params)

    if HOUR_COLUMN not in train_data.columns:
        return _train_pooled(train_data, params, feature_columns, target_column, target_transform)

    hourly_models: dict[int, Pipeline] = {}
    alpha_by_hour: dict[int, float] = {}
    n_train_rows_by_hour: dict[int, int] = {}
    for hour in range(24):
        rows = train_data.loc[train_data[HOUR_COLUMN] == hour].dropna(
            subset=[target_column, *feature_columns]
        )
        n_train_rows_by_hour[hour] = len(rows)
        if rows.empty:
            raise ValueError(f"LEAR has no complete training rows for hour {hour}.")

        model = _build_pipeline(params)
        model.fit(
            rows[feature_columns],
            model_support.transform_target(rows[target_column], target_transform),
        )
        hourly_models[hour] = model
        alpha_by_hour[hour] = float(model.named_steps["regressor"].alpha_)

    return LearModelState(
        hourly_models=hourly_models,
        pooled_model=None,
        feature_columns=feature_columns,
        target_transform=target_transform,
        alpha_by_hour=alpha_by_hour,
        fitted_hours=sorted(hourly_models),
        n_train_rows_by_hour=n_train_rows_by_hour,
    )


def predict(
    model_state: LearModelState,
    test_data: pd.DataFrame,
    _params: dict[str, object],
) -> pd.Series:
    """Predict each row with the model fitted for its delivery hour."""

    predictions = pd.Series(index=test_data.index, dtype=float)

    if model_state.pooled_model is not None:
        _predict_into(predictions, model_state, model_state.pooled_model, test_data)
        return predictions

    if HOUR_COLUMN not in test_data.columns:
        raise ValueError(f"LEAR needs the hourly split column: {HOUR_COLUMN}")

    for hour, model in model_state.hourly_models.items():
        hour_rows = test_data.loc[test_data[HOUR_COLUMN] == hour]
        if not hour_rows.empty:
            _predict_into(predictions, model_state, model, hour_rows)
    return predictions


# --- Hyperparameters and naming --------------------------------------------


def build_param_grid(
    regularization: str = "lasso",
    target_transform: str = "raw",
    **_unused_options,
) -> list[dict[str, object]]:
    """Return one configuration per family; alpha is tuned per hour in ``train``."""

    validate_model_choices(regularization, target_transform)
    return [{"regularization": regularization, "target_transform": target_transform}]


def output_folder_name(
    regularization: str = "lasso",
    target_transform: str = "raw",
    **_unused_options,
) -> str:
    """Name output folders by regularisation family and target transform."""

    validate_model_choices(regularization, target_transform)
    return f"{MODEL_NAME}_{regularization}_{target_transform}"


def validate_model_choices(regularization: str, target_transform: str) -> None:
    """Reject unsupported regularisation families or target transforms early."""

    if regularization not in SUPPORTED_REGULARIZATION:
        raise ValueError(f"Unsupported regularization: {regularization}")
    if target_transform not in model_support.SUPPORTED_TARGET_TRANSFORMS:
        raise ValueError(f"Unsupported target transform: {target_transform}")


# --- Internals -------------------------------------------------------------


def _train_pooled(
    train_data: pd.DataFrame,
    params: dict[str, object],
    feature_columns: list[str],
    target_column: str,
    target_transform: str,
) -> LearModelState:
    """Fit a single model when the data has no per-hour split (Option B)."""

    rows = train_data.dropna(subset=[target_column, *feature_columns])
    if rows.empty:
        raise ValueError("LEAR pooled model has no complete training rows.")

    model = _build_pipeline(params)
    model.fit(
        rows[feature_columns],
        model_support.transform_target(rows[target_column], target_transform),
    )
    return LearModelState(
        hourly_models={},
        pooled_model=model,
        feature_columns=feature_columns,
        target_transform=target_transform,
        alpha_by_hour={-1: float(model.named_steps["regressor"].alpha_)},
        fitted_hours=[],
        n_train_rows_by_hour={-1: len(rows)},
    )


def _predict_into(
    predictions: pd.Series,
    model_state: LearModelState,
    model: Pipeline,
    rows: pd.DataFrame,
) -> None:
    """Predict the rows with complete features and write them into ``predictions``."""

    complete = rows[model_state.feature_columns].notna().all(axis=1)
    if not complete.any():
        return

    usable_index = rows.loc[complete].index
    raw_prediction = model.predict(rows.loc[usable_index, model_state.feature_columns])
    predictions.loc[usable_index] = model_support.inverse_transform_prediction(
        raw_prediction,
        model_state.target_transform,
    )


def _build_pipeline(params: dict[str, object]) -> Pipeline:
    """Standard-scale features, then fit a cross-validated linear regressor."""

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", _build_cross_validated_regressor(params)),
        ]
    )


def _build_cross_validated_regressor(params: dict[str, object]):
    """Create the per-hour regressor that selects its own alpha by time-series CV."""

    regularization = str(params.get("regularization", "lasso"))
    splitter = TimeSeriesSplit(n_splits=constants.LEAR_CV_SPLITS)

    if regularization == "lasso":
        return LassoCV(
            alphas=constants.LEAR_ALPHA_PATH_LENGTH,
            cv=splitter,
            max_iter=50000,
            tol=1e-4,
            selection="random",
            random_state=0,
        )
    if regularization == "ridge":
        return RidgeCV(alphas=constants.RIDGE_ALPHA_GRID, cv=splitter)
    if regularization == "elasticnet":
        return ElasticNetCV(
            l1_ratio=constants.LEAR_L1_RATIO_GRID,
            alphas=constants.LEAR_ALPHA_PATH_LENGTH,
            cv=splitter,
            max_iter=50000,
            tol=1e-4,
            selection="random",
            random_state=0,
        )
    raise ValueError(f"Unsupported regularization: {regularization}")
