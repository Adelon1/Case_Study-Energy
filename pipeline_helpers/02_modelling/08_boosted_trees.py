"""Gradient-boosted decision-tree price model.

A single pooled model predicts every delivery hour. Hour-of-day, weekday, and
month are passed as categorical features, so the trees can split on specific
hours and seasons and learn hour-specific, nonlinear behaviour without training
24 separate models. The default loss is absolute error, which is robust to the
price spikes that are common in day-ahead markets.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

model_support = importlib.import_module("pipeline_helpers.02_modelling.05_model_support")


MODEL_NAME = "boosted_trees"

# Calendar columns that are categories, not magnitudes. Passing them as
# categoricals lets the trees split on individual hours, weekdays, and months.
CATEGORICAL_FEATURES = ["local_hour", "local_weekday", "local_month"]


@dataclass(frozen=True)
class BoostedTreesState:
    """Fitted boosted-tree model and the feature columns it expects."""

    model: HistGradientBoostingRegressor
    feature_columns: list[str]


# --- Model contract: train and predict -------------------------------------


def train(train_data: pd.DataFrame, params: dict[str, object]) -> BoostedTreesState:
    """Fit one pooled boosted-tree model across all delivery hours."""

    target_column = model_support.resolve_target_column(params)
    feature_columns = model_support.resolve_feature_columns(train_data, params)
    rows = train_data.dropna(subset=[target_column])
    if rows.empty:
        raise ValueError("Boosted-trees model has no training rows with a target value.")

    model = HistGradientBoostingRegressor(
        loss=str(params["loss"]),
        learning_rate=float(params["learning_rate"]),
        max_iter=int(params["max_iter"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        l2_regularization=float(params["l2_regularization"]),
        categorical_features=_present_categoricals(feature_columns),
        early_stopping=False,  # rolling validation already measures generalisation
        random_state=0,
    )
    model.fit(rows[feature_columns], rows[target_column].to_numpy())
    return BoostedTreesState(model=model, feature_columns=feature_columns)


def predict(
    model_state: BoostedTreesState,
    test_data: pd.DataFrame,
    _params: dict[str, object],
) -> pd.Series:
    """Predict every test row with the pooled boosted-tree model."""

    predictions = pd.Series(index=test_data.index, dtype=float)
    if test_data.empty:
        return predictions

    predictions.loc[test_data.index] = model_state.model.predict(
        test_data[model_state.feature_columns]
    )
    return predictions


# --- Hyperparameters and naming --------------------------------------------


def build_param_grid(**_unused_options) -> list[dict[str, object]]:
    """Small, well-spread grid. Absolute error is robust to price spikes."""

    return [
        {
            "loss": "absolute_error",
            "learning_rate": 0.05,
            "max_iter": 700,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 50,
            "l2_regularization": 0.0,
        },
        {
            "loss": "absolute_error",
            "learning_rate": 0.03,
            "max_iter": 1000,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 100,
            "l2_regularization": 0.0,
        },
        {
            "loss": "absolute_error",
            "learning_rate": 0.05,
            "max_iter": 700,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 50,
            "l2_regularization": 1.0,
        },
        {
            "loss": "squared_error",
            "learning_rate": 0.05,
            "max_iter": 700,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 50,
            "l2_regularization": 0.0,
        },
    ]


def output_folder_name(**_unused_options) -> str:
    """Boosted trees have no user-facing variants, so the name is fixed."""

    return MODEL_NAME


# --- Internals -------------------------------------------------------------


def _present_categoricals(feature_columns: list[str]) -> list[str]:
    """Return the calendar categoricals that are actually in the feature set."""

    return [column for column in CATEGORICAL_FEATURES if column in feature_columns]
