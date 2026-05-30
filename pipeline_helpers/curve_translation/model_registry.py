"""Select existing model predictions or request period-specific predictions."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

from pipeline_helpers.curve_translation import constants
from pipeline_helpers.curve_translation.forecast_blocks import DeliveryPeriod
from pipeline_helpers.modelling.period_prediction import (
    PredictionPeriod,
    check_prediction_coverage,
    period_slug,
    prediction_file_covers,
    train_and_predict_period,
)


@dataclass(frozen=True)
class PredictionSource:
    """Prediction file selected or produced by the registry."""

    predictions_path: Path
    metrics_path: Path
    model_folder: Path
    source: str
    retrained: bool


def load_model_module(model_name: str):
    """Import a model module from ``pipeline_helpers.modelling``."""

    return importlib.import_module(f"pipeline_helpers.modelling.{model_name}")


def output_folder_name(model_module, model_options: dict[str, object]) -> str:
    """Return the model-specific output folder name."""

    if hasattr(model_module, "output_folder_name"):
        return model_module.output_folder_name(**model_options)
    return model_module.MODEL_NAME


def build_model_options(
    regularization: str | None = None,
    target_transform: str | None = None,
) -> dict[str, object]:
    """Build model options in the same style as the validation script."""

    options: dict[str, object] = {}
    if regularization is not None:
        options["regularization"] = regularization
    if target_transform is not None:
        options["target_transform"] = target_transform
    return options


def model_folder_for(
    feature_path: str | Path,
    model_name: str,
    model_options: dict[str, object],
    output_base_folder: str | Path | None = None,
) -> Path:
    """Resolve the output folder for a selected model."""

    feature_path = Path(feature_path)
    model_module = load_model_module(model_name)
    base_folder = Path(output_base_folder) if output_base_folder else feature_path.parent
    return base_folder / output_folder_name(model_module, model_options)


def as_prediction_period(period: DeliveryPeriod) -> PredictionPeriod:
    """Convert curve-translation periods to the modelling period type."""

    return PredictionPeriod(start_utc=period.start_utc, end_utc=period.end_utc)


def get_or_create_predictions(
    feature_path: str | Path,
    model_name: str,
    model_options: dict[str, object],
    period: DeliveryPeriod,
    output_base_folder: str | Path | None = None,
    force_retrain: bool = False,
) -> PredictionSource:
    """Use existing period/final predictions if possible; otherwise ask modelling to train."""

    model_folder = model_folder_for(feature_path, model_name, model_options, output_base_folder)
    modelling_period = as_prediction_period(period)
    period_folder = model_folder / "period_predictions" / period_slug(modelling_period)
    period_predictions_path = period_folder / "predictions.csv"
    period_metrics_path = period_folder / "metrics.csv"

    if not force_retrain and prediction_file_covers(period_predictions_path, modelling_period):
        if not period_metrics_path.exists():
            raise ValueError(
                f"Missing metrics file for selected period predictions: {period_metrics_path}"
            )
        check_prediction_coverage(
            period_predictions_path,
            modelling_period,
            constants.MIN_PREDICTION_COVERAGE,
        )
        return PredictionSource(
            predictions_path=period_predictions_path,
            metrics_path=period_metrics_path,
            model_folder=model_folder,
            source="existing_period_predictions",
            retrained=False,
        )

    predictions_path = model_folder / "final_holdout_predictions.csv"
    metrics_path = model_folder / "final_holdout_metrics.csv"

    if not force_retrain and prediction_file_covers(predictions_path, modelling_period):
        if not metrics_path.exists():
            raise ValueError(f"Missing metrics file for selected predictions: {metrics_path}")
        check_prediction_coverage(
            predictions_path,
            modelling_period,
            constants.MIN_PREDICTION_COVERAGE,
        )
        return PredictionSource(
            predictions_path=predictions_path,
            metrics_path=metrics_path,
            model_folder=model_folder,
            source="existing_final_holdout_predictions",
            retrained=False,
        )

    period_result = train_and_predict_period(
        feature_path=feature_path,
        model_name=model_name,
        model_options=model_options,
        output_folder=model_folder,
        period=modelling_period,
    )
    check_prediction_coverage(
        period_result.predictions_path,
        modelling_period,
        constants.MIN_PREDICTION_COVERAGE,
    )
    return PredictionSource(
        predictions_path=period_result.predictions_path,
        metrics_path=period_result.metrics_path,
        model_folder=model_folder,
        source="retrained_rolling_window",
        retrained=True,
    )
