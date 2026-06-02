"""Shared building blocks for model modules and the modelling pipeline steps.

Every model module exposes the same small contract to the validation engine:
``MODEL_NAME``, ``build_param_grid``, ``train``, and ``predict``. This module
holds the plumbing those modules and steps would otherwise duplicate: loading a
model module by name, reading the feature and target columns the dataset
injected, transforming the target, and naming output folders.

Public entry points:
    ``load_model_module(...)``
    ``model_run_name(...)``
    ``model_state_diagnostics(...)``
    ``resolve_target_column(...)``
    ``resolve_feature_columns(...)``
    ``transform_target(...)``
    ``inverse_transform_prediction(...)``
"""

from __future__ import annotations

import importlib
from types import ModuleType

import numpy as np
import pandas as pd

constants = importlib.import_module("pipeline_helpers.02_modelling.00_constants")


SUPPORTED_TARGET_TRANSFORMS = {"raw", "asinh"}


# ---------------------------------------------------------------------------
# Public API used by pipeline steps and model modules
# ---------------------------------------------------------------------------


def load_model_module(model_name: str) -> ModuleType:
    """Import a numbered model module by its stable public name."""

    module_name_by_model = {
        "baseline_model": "30_baseline_model",
        "lear_model": "31_lear_model",
        "boosted_tree_model": "32_boosted_tree_model",
        "theil_sen_model": "33_theil_sen_model",
        "ransac_lasso_model": "34_ransac_lasso_model",
    }
    module_name = module_name_by_model.get(model_name, model_name)
    return importlib.import_module(f"pipeline_helpers.02_modelling.{module_name}")


def model_run_name(model_module: ModuleType, model_options: dict[str, object]) -> str:
    """Return the model's output-folder name, or its ``MODEL_NAME`` as fallback."""

    if hasattr(model_module, "output_folder_name"):
        return model_module.output_folder_name(**model_options)
    return model_module.MODEL_NAME


def model_state_diagnostics(model_state: object) -> dict[str, object]:
    """Collect optional, human-readable diagnostics from a fitted model state."""

    diagnostics: dict[str, object] = {}
    for attribute in ["alpha_by_hour"]:
        if hasattr(model_state, attribute):
            diagnostics[attribute] = getattr(model_state, attribute)
    return diagnostics


def resolve_target_column(params: dict[str, object]) -> str:
    """Return the target column the validation engine injected, or the default."""

    return str(params.get("_target_column", constants.TARGET_COLUMN))


def resolve_feature_columns(train_data: pd.DataFrame, params: dict[str, object]) -> list[str]:
    """Return the leakage-safe feature columns chosen by the modelling dataset.

    The modelling dataset is the single source of truth for which columns are
    safe to use, so models never pick features themselves.
    """

    feature_columns = params.get("_feature_columns") or params.get("feature_columns")
    if not feature_columns:
        raise ValueError("Model needs '_feature_columns' provided by the modelling dataset.")

    feature_columns = list(feature_columns)
    missing = [column for column in feature_columns if column not in train_data.columns]
    if missing:
        preview = ", ".join(missing[:10])
        if len(missing) > 10:
            preview += ", ..."
        raise ValueError(f"Selected features missing from the feature table: {preview}")
    return feature_columns


def transform_target(y: pd.Series, target_transform: str) -> np.ndarray:
    """Map the training target into model space."""

    if target_transform == "raw":
        return y.to_numpy()
    if target_transform == "asinh":
        return np.arcsinh(y.to_numpy())
    raise ValueError(f"Unsupported target transform: {target_transform}")


def inverse_transform_prediction(y_pred: np.ndarray, target_transform: str) -> np.ndarray:
    """Map predictions back to EUR/MWh."""

    if target_transform == "raw":
        return y_pred
    if target_transform == "asinh":
        return np.sinh(y_pred)
    raise ValueError(f"Unsupported target transform: {target_transform}")
