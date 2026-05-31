"""Find, train, save, and reuse predictions for requested delivery periods."""

from __future__ import annotations

import importlib
import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from pipeline_helpers.curve_translation import constants as curve_constants
from pipeline_helpers.curve_translation.forecast_blocks import DeliveryPeriod
from pipeline_helpers.modelling import constants
from pipeline_helpers.modelling.model_io import default_models_base_folder
from pipeline_helpers.modelling.modelling_dataset import build_modelling_dataset
from pipeline_helpers.modelling.validation import (
    TimeWindow,
    add_months,
    load_feature_data,
    slice_window,
    train_predict_window,
)


@dataclass(frozen=True)
class PredictionPeriod:
    """Half-open UTC period to predict."""

    start_utc: pd.Timestamp
    end_utc: pd.Timestamp


@dataclass(frozen=True)
class PeriodPredictionResult:
    """Artifacts produced by a period-specific prediction run."""

    predictions_path: Path
    metrics_path: Path
    model_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class PredictionSource:
    """Prediction file selected or produced for a requested period."""

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
    target_option: str = "A",
    feature_mode: str = "period_hourly_safe",
    period_days: int | None = None,
    block: str = "baseload",
) -> Path:
    """Resolve the model artifact folder for the selected forecast setup."""

    model_module = load_model_module(model_name)
    base_folder = (
        Path(output_base_folder)
        if output_base_folder
        else default_models_base_folder(feature_path)
    )
    name_parts = [
        output_folder_name(model_module, model_options),
        f"option_{target_option.lower()}",
    ]
    if target_option == "A":
        name_parts.append(feature_mode)
    else:
        name_parts.append(f"{period_days or 'period'}d_{block}")
    return base_folder / "__".join(name_parts)


def as_prediction_period(period: DeliveryPeriod) -> PredictionPeriod:
    """Convert a curve-translation period to a modelling prediction period."""

    return PredictionPeriod(start_utc=period.start_utc, end_utc=period.end_utc)


def period_slug(period: PredictionPeriod) -> str:
    """Build a stable file suffix for a delivery period."""

    return f"{period.start_utc:%Y%m%d}_{period.end_utc:%Y%m%d}"


def write_json(data: dict[str, object], path: Path) -> None:
    """Write JSON with stable formatting."""

    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")


def read_json(path: Path) -> dict[str, object]:
    """Read a JSON object from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def read_best_params(
    model_folder: str | Path,
    model_module,
    model_options: dict[str, object],
) -> dict[str, object]:
    """Read best params from artifacts, falling back to validation summary or first grid row."""

    model_folder = Path(model_folder)
    best_params_path = model_folder / "best_params.json"
    if best_params_path.exists():
        return read_json(best_params_path)

    summary_path = model_folder / "validation_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path, sep=None, engine="python")
        if "params" in summary.columns and not summary.empty:
            return json.loads(summary.iloc[0]["params"])

    if hasattr(model_module, "build_param_grid"):
        return model_module.build_param_grid(**model_options)[0]
    if hasattr(model_module, "PARAM_GRID"):
        return model_module.PARAM_GRID[0]
    raise ValueError(
        f"No best params found in {model_folder}. Run validation first or provide model artifacts."
    )


def model_state_diagnostics(model_state) -> dict[str, object]:
    """Extract optional diagnostics from fitted model state objects."""

    diagnostics: dict[str, object] = {}
    for attribute in ["fitted_hours", "n_train_rows_by_hour", "feature_columns"]:
        if hasattr(model_state, attribute):
            diagnostics[attribute] = getattr(model_state, attribute)
    return diagnostics


def prediction_file_covers(predictions_path: str | Path, period: PredictionPeriod) -> bool:
    """Check whether a prediction file covers the full requested period."""

    predictions_path = Path(predictions_path)
    if not predictions_path.exists():
        return False
    predictions = pd.read_csv(predictions_path)
    if predictions.empty or "timestamp_utc" not in predictions.columns:
        return False
    timestamps = pd.to_datetime(predictions["timestamp_utc"], utc=True)
    if "period_end" in predictions.columns:
        period_ends = pd.to_datetime(predictions["period_end"], utc=True)
        return bool(
            (timestamps.min() <= period.start_utc)
            and (period_ends.max() >= period.end_utc)
        )
    return bool(
        (timestamps.min() <= period.start_utc)
        and (timestamps.max() >= period.end_utc - pd.Timedelta(hours=1))
    )


def check_prediction_coverage(
    predictions_path: str | Path,
    period: PredictionPeriod,
    minimum_coverage: float,
) -> None:
    """Warn if a prediction file has low non-null coverage inside the requested period."""

    predictions = pd.read_csv(predictions_path)
    timestamps = pd.to_datetime(predictions["timestamp_utc"], utc=True)
    period_predictions = predictions.loc[
        (timestamps >= period.start_utc) & (timestamps < period.end_utc)
    ]
    if "period_end" in predictions.columns:
        period_ends = pd.to_datetime(predictions["period_end"], utc=True)
        period_predictions = predictions.loc[
            (timestamps <= period.start_utc) & (period_ends >= period.end_utc)
        ]
    if period_predictions.empty:
        raise ValueError("Prediction file covers dates but has no rows in the requested period.")

    coverage = period_predictions["y_pred"].notna().mean()
    if coverage < minimum_coverage:
        warnings.warn(
            f"Prediction coverage is below {minimum_coverage:.0%}: {coverage:.2%}",
            RuntimeWarning,
        )


def train_and_predict_period(
    feature_path: str | Path,
    model_name: str,
    model_options: dict[str, object],
    output_folder: str | Path,
    period: PredictionPeriod,
    train_months: int = constants.TRAIN_MONTHS,
    target_option: str = "A",
    feature_mode: str = "period_hourly_safe",
    period_days: int | None = None,
    block: str = "baseload",
) -> PeriodPredictionResult:
    """Train on the previous rolling window and predict the requested period."""

    model_module = load_model_module(model_name)
    feature_table = load_feature_data(feature_path)
    if period_days is None:
        period_days = max(1, int((period.end_utc - period.start_utc) / pd.Timedelta(days=1)))
    modelling_dataset = build_modelling_dataset(
        feature_table=feature_table,
        model_name=model_module.MODEL_NAME,
        target_option=target_option,
        feature_mode=feature_mode,
        period_days=period_days,
        block=block,
    )
    table = modelling_dataset.table
    train_begin = add_months(period.start_utc, -train_months)
    train_end = period.start_utc

    data_begin = table[constants.TIMESTAMP_COLUMN].min()
    if train_begin < data_begin:
        raise ValueError(
            "Requested period is too early for the configured rolling training window. "
            f"Need data from {train_begin}, but feature data starts at {data_begin}."
        )

    train_data = slice_window(table, train_begin, train_end)
    test_data = slice_window(table, period.start_utc, period.end_utc)
    if train_data.empty:
        raise ValueError("Training window is empty.")
    if test_data.empty:
        raise ValueError("Requested prediction period is empty in the feature table.")

    output_folder = Path(output_folder)
    best_params = read_best_params(output_folder, model_module, model_options)
    prediction_result = train_predict_window(
        table=table,
        model_module=model_module,
        params=best_params,
        window=TimeWindow(
            train_begin=train_begin,
            train_end=train_end,
            test_begin=period.start_utc,
            test_end=period.end_utc,
        ),
        split_name="period_prediction",
        fold_number=1,
        feature_columns=modelling_dataset.feature_columns,
    )

    period_folder = output_folder / "period_predictions" / period_slug(period)
    period_folder.mkdir(parents=True, exist_ok=True)

    predictions_path = period_folder / "predictions.csv"
    metrics_path = period_folder / "metrics.csv"
    model_path = period_folder / "model.joblib"
    metadata_path = period_folder / "metadata.json"

    predictions = prediction_result.predictions
    if target_option == "B" and "period_end" in test_data.columns:
        predictions["period_end"] = test_data["period_end"]
        predictions["prediction_granularity"] = "period_average"
        predictions["block"] = block
    else:
        predictions["prediction_granularity"] = "hourly"
        predictions["block"] = block
    predictions.to_csv(predictions_path, index=False)
    pd.DataFrame([prediction_result.metric_row]).round(2).to_csv(metrics_path, index=False)
    joblib.dump(prediction_result.model_state, model_path)
    write_json(
        {
            "model": model_module.MODEL_NAME,
            "model_options": model_options,
            "params": best_params,
            "feature_csv": str(feature_path),
            "target_option": target_option,
            "feature_mode": modelling_dataset.feature_mode,
            "period_days": period_days,
            "block": block,
            "feature_columns": modelling_dataset.feature_columns,
            "train_begin": train_begin,
            "train_end": train_end,
            "prediction_begin": period.start_utc,
            "prediction_end": period.end_utc,
            "training_diagnostics": model_state_diagnostics(prediction_result.model_state),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        metadata_path,
    )

    return PeriodPredictionResult(
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        model_path=model_path,
        metadata_path=metadata_path,
    )


def get_or_create_predictions(
    feature_path: str | Path,
    model_name: str,
    model_options: dict[str, object],
    period: DeliveryPeriod,
    output_base_folder: str | Path | None = None,
    force_retrain: bool = False,
    target_option: str = "A",
    feature_mode: str = "period_hourly_safe",
    period_days: int | None = None,
    block: str = "baseload",
) -> PredictionSource:
    """Use existing period/final predictions, otherwise train for the requested period."""

    if period_days is None:
        period_days = max(1, int((period.end_utc - period.start_utc) / pd.Timedelta(days=1)))

    model_folder = model_folder_for(
        feature_path=feature_path,
        model_name=model_name,
        model_options=model_options,
        output_base_folder=output_base_folder,
        target_option=target_option,
        feature_mode=feature_mode,
        period_days=period_days,
        block=block,
    )
    modelling_period = as_prediction_period(period)
    period_folder = model_folder / "period_predictions" / period_slug(modelling_period)
    period_predictions_path = period_folder / "predictions.csv"
    period_metrics_path = period_folder / "metrics.csv"

    if not force_retrain and prediction_file_covers(period_predictions_path, modelling_period):
        if not period_metrics_path.exists():
            raise ValueError(f"Missing metrics file: {period_metrics_path}")
        check_prediction_coverage(
            period_predictions_path,
            modelling_period,
            curve_constants.MIN_PREDICTION_COVERAGE,
        )
        return PredictionSource(
            predictions_path=period_predictions_path,
            metrics_path=period_metrics_path,
            model_folder=model_folder,
            source="existing_period_predictions",
            retrained=False,
        )

    final_predictions_path = model_folder / "final_holdout_predictions.csv"
    final_metrics_path = model_folder / "final_holdout_metrics.csv"
    if not force_retrain and prediction_file_covers(final_predictions_path, modelling_period):
        if not final_metrics_path.exists():
            raise ValueError(f"Missing metrics file: {final_metrics_path}")
        check_prediction_coverage(
            final_predictions_path,
            modelling_period,
            curve_constants.MIN_PREDICTION_COVERAGE,
        )
        return PredictionSource(
            predictions_path=final_predictions_path,
            metrics_path=final_metrics_path,
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
        target_option=target_option,
        feature_mode=feature_mode,
        period_days=period_days,
        block=block,
    )
    check_prediction_coverage(
        period_result.predictions_path,
        modelling_period,
        curve_constants.MIN_PREDICTION_COVERAGE,
    )
    return PredictionSource(
        predictions_path=period_result.predictions_path,
        metrics_path=period_result.metrics_path,
        model_folder=model_folder,
        source="retrained_rolling_window",
        retrained=True,
    )
