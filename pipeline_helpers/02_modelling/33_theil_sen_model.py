"""Robust linear benchmark using Theil-Sen regression."""

from __future__ import annotations

from dataclasses import dataclass
import importlib

import pandas as pd
from sklearn.linear_model import TheilSenRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

model_support = importlib.import_module("pipeline_helpers.02_modelling.05_model_support")


MODEL_NAME = "theil_sen_model"


@dataclass(frozen=True)
class TheilSenState:
    """Fitted robust linear pipeline and the feature list it expects."""

    model: Pipeline
    feature_columns: list[str]
    target_transform: str


def train(train_data: pd.DataFrame, params: dict[str, object]) -> TheilSenState:
    """Fit one pooled robust linear model on the selected feature set."""

    target_transform = str(params.get("target_transform", "raw"))
    if target_transform not in model_support.SUPPORTED_TARGET_TRANSFORMS:
        raise ValueError(f"Unsupported target transform: {target_transform}")

    target_column = model_support.resolve_target_column(params)
    feature_columns = model_support.resolve_feature_columns(train_data, params)
    model = fit_one_model(train_data, params, feature_columns, target_column, target_transform)

    return TheilSenState(
        model=model,
        feature_columns=feature_columns,
        target_transform=target_transform,
    )


def predict(
    model_state: TheilSenState,
    test_data: pd.DataFrame,
    _params: dict[str, object],
) -> pd.Series:
    """Predict rows with complete selected features."""

    predictions = pd.Series(index=test_data.index, dtype=float)
    if test_data.empty:
        return predictions

    predict_into(predictions, model_state, model_state.model, test_data)
    return predictions


def build_param_grid(
    target_transform: str = "raw",
    **_unused_options,
) -> list[dict[str, object]]:
    """Return a tiny grid because Theil-Sen is computationally expensive."""

    if target_transform not in model_support.SUPPORTED_TARGET_TRANSFORMS:
        raise ValueError(f"Unsupported target transform: {target_transform}")
    return [
        {
            "target_transform": target_transform,
            "max_subpopulation": 5000,
        },
    ]


def output_folder_name(target_transform: str = "raw", **_unused_options) -> str:
    """Name output folders by the target transform."""

    if target_transform not in model_support.SUPPORTED_TARGET_TRANSFORMS:
        raise ValueError(f"Unsupported target transform: {target_transform}")
    return f"{MODEL_NAME}_{target_transform}"


def fit_one_model(
    train_data: pd.DataFrame,
    params: dict[str, object],
    feature_columns: list[str],
    target_column: str,
    target_transform: str,
) -> Pipeline:
    """Fit one scaled Theil-Sen pipeline."""

    rows = train_data.dropna(subset=[target_column, *feature_columns])
    if rows.empty:
        raise ValueError("Theil-Sen model has no complete training rows.")

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "regressor",
                TheilSenRegressor(
                    max_subpopulation=int(params["max_subpopulation"]),
                    max_iter=300,
                    tol=1e-3,
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
    model_state: TheilSenState,
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
