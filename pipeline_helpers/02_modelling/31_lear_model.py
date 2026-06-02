"""LEAR-style regularised linear model.

LEAR fits one regularised linear regression per delivery hour. Every hour shares
the same leakage-safe feature set but learns its own coefficients. The
regularisation grid is evaluated by the outer rolling validation, and this
module selects the best validated setting separately per hour.

Model contract functions called by validation/window prediction:
    ``build_param_grid(...)``
    ``train(...)``
    ``predict(...)``
    ``choose_best_params(...)``
    ``select_validation_predictions(...)``
    ``output_folder_name(...)``
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json

import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

constants = importlib.import_module("pipeline_helpers.02_modelling.00_constants")
model_support = importlib.import_module("pipeline_helpers.02_modelling.05_model_support")


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
    params_by_hour: dict[int, dict[str, object]]
    fitted_hours: list[int]
    n_train_rows_by_hour: dict[int, int]


# ---------------------------------------------------------------------------
# Model contract: train and predict
# ---------------------------------------------------------------------------


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
    params_by_hour: dict[int, dict[str, object]] = {}
    n_train_rows_by_hour: dict[int, int] = {}
    for hour in range(24):
        hour_params = params_for_hour(params, hour)
        rows = train_data.loc[train_data[HOUR_COLUMN] == hour].dropna(
            subset=[target_column, *feature_columns]
        )
        n_train_rows_by_hour[hour] = len(rows)
        if rows.empty:
            raise ValueError(f"LEAR has no complete training rows for hour {hour}.")

        model = _build_pipeline(hour_params)
        model.fit(
            rows[feature_columns],
            model_support.transform_target(rows[target_column], target_transform),
        )
        hourly_models[hour] = model
        alpha_by_hour[hour] = float(hour_params["alpha"])
        params_by_hour[hour] = public_model_params(hour_params)

    return LearModelState(
        hourly_models=hourly_models,
        pooled_model=None,
        feature_columns=feature_columns,
        target_transform=target_transform,
        alpha_by_hour=alpha_by_hour,
        params_by_hour=params_by_hour,
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


# ---------------------------------------------------------------------------
# Model contract: hyperparameters, selection, and naming
# ---------------------------------------------------------------------------


def build_param_grid(
    regularization: str = "lasso",
    target_transform: str = "raw",
    **_unused_options,
) -> list[dict[str, object]]:
    """Return fixed-penalty configurations for outer rolling validation."""

    validate_model_choices(regularization, target_transform)
    if regularization == "ridge":
        return [
            {
                "regularization": regularization,
                "target_transform": target_transform,
                "alpha": alpha,
            }
            for alpha in constants.RIDGE_ALPHA_GRID
        ]
    if regularization == "elasticnet":
        return [
            {
                "regularization": regularization,
                "target_transform": target_transform,
                "alpha": alpha,
                "l1_ratio": l1_ratio,
            }
            for alpha in constants.LEAR_ALPHA_GRID
            for l1_ratio in constants.LEAR_L1_RATIO_GRID
        ]
    return [
        {
            "regularization": regularization,
            "target_transform": target_transform,
            "alpha": alpha,
        }
        for alpha in constants.LEAR_ALPHA_GRID
    ]


def choose_best_params(metrics: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, object]:
    """Choose the best validated LEAR setting, separately by hour when possible."""

    if HOUR_COLUMN not in predictions.columns:
        average_metrics = metrics.groupby("params", as_index=False)["mae"].mean()
        return json.loads(average_metrics.sort_values("mae").iloc[0]["params"])

    params_by_hour: dict[int, dict[str, object]] = {}
    for hour, hour_predictions in predictions.dropna(subset=["y_pred"]).groupby(HOUR_COLUMN):
        scored = (
            hour_predictions.assign(abs_error=lambda frame: (frame["y_pred"] - frame["y_true"]).abs())
            .groupby("params", as_index=False)["abs_error"]
            .mean()
            .sort_values("abs_error")
        )
        if scored.empty:
            continue
        params_by_hour[int(hour)] = json.loads(scored.iloc[0]["params"])

    if sorted(params_by_hour) != list(range(24)):
        missing = sorted(set(range(24)) - set(params_by_hour))
        raise ValueError(f"LEAR could not choose validated params for hours: {missing}")

    first_params = next(iter(params_by_hour.values()))
    return {
        "regularization": first_params["regularization"],
        "target_transform": first_params["target_transform"],
        "params_by_hour": {str(hour): params for hour, params in params_by_hour.items()},
    }


def select_validation_predictions(
    predictions: pd.DataFrame,
    best_params: dict[str, object],
) -> pd.DataFrame:
    """Return rows predicted with each hour's selected validation setting."""

    params_by_hour = best_params.get("params_by_hour")
    if not params_by_hour or HOUR_COLUMN not in predictions.columns:
        return predictions.loc[predictions["params"] == params_to_string(best_params)].copy()

    selected_parts = []
    for hour_text, hour_params in params_by_hour.items():
        hour = int(hour_text)
        selected_parts.append(
            predictions.loc[
                (predictions[HOUR_COLUMN] == hour)
                & (predictions["params"] == params_to_string(hour_params))
            ]
        )
    if not selected_parts:
        return pd.DataFrame(columns=predictions.columns)
    return pd.concat(selected_parts, ignore_index=True).sort_values(
        ["fold", HOUR_COLUMN, "timestamp_utc"],
    )


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


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


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
        alpha_by_hour={-1: float(params.get("alpha", 1.0))},
        params_by_hour={-1: public_model_params(params)},
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
    """Standard-scale features, then fit a fixed-penalty linear regressor."""

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", _build_regressor(params)),
        ]
    )


def _build_regressor(params: dict[str, object]):
    """Create the fixed-penalty regressor selected by rolling validation."""

    regularization = str(params.get("regularization", "lasso"))
    alpha = float(params.get("alpha", 1.0))

    if regularization == "lasso":
        return Lasso(
            alpha=alpha,
            max_iter=50000,
            tol=1e-4,
            selection="random",
            random_state=0,
        )
    if regularization == "ridge":
        return Ridge(alpha=alpha)
    if regularization == "elasticnet":
        return ElasticNet(
            alpha=alpha,
            l1_ratio=float(params.get("l1_ratio", 0.5)),
            max_iter=50000,
            tol=1e-4,
            selection="random",
            random_state=0,
        )
    raise ValueError(f"Unsupported regularization: {regularization}")


def params_to_string(params: dict[str, object]) -> str:
    """Serialize params the same way as validation tables."""

    return json.dumps(params, sort_keys=True)


def params_for_hour(params: dict[str, object], hour: int) -> dict[str, object]:
    """Return this hour's selected params, or the shared grid params."""

    params_by_hour = params.get("params_by_hour")
    if isinstance(params_by_hour, dict):
        hour_params = params_by_hour.get(str(hour), params_by_hour.get(hour))
        if hour_params is None:
            raise ValueError(f"LEAR missing selected params for hour {hour}.")
        return dict(hour_params)
    return params


def public_model_params(params: dict[str, object]) -> dict[str, object]:
    """Drop injected validation keys before storing model diagnostics."""

    return {
        key: value
        for key, value in params.items()
        if not str(key).startswith("_") and key != "params_by_hour"
    }
