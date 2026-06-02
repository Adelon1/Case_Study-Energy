"""Train, save, and reuse model predictions for explicit delivery windows.

Validation answers the question "which params work well across many rolling
historical windows?". This module answers the operational question "using those
params, train on this specific history and predict this specific future window".
"""

from __future__ import annotations

import importlib
import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

curve_constants = importlib.import_module("pipeline_helpers.03_curve_translation.00_constants")
forecast_blocks = importlib.import_module("pipeline_helpers.03_curve_translation.01_forecast_blocks")
constants = importlib.import_module("pipeline_helpers.02_modelling.00_constants")
model_io = importlib.import_module("pipeline_helpers.02_modelling.04_model_io")
model_support = importlib.import_module("pipeline_helpers.02_modelling.05_model_support")
modelling_dataset = importlib.import_module("pipeline_helpers.02_modelling.01_modelling_dataset")
validation = importlib.import_module("pipeline_helpers.02_modelling.09_validation")

DeliveryPeriod = forecast_blocks.DeliveryPeriod
default_models_base_folder = model_io.default_models_base_folder
read_json = model_io.read_json
write_json = model_io.write_json
load_model_module = model_support.load_model_module
model_run_name = model_support.model_run_name
model_state_diagnostics = model_support.model_state_diagnostics
build_modelling_dataset = modelling_dataset.build_modelling_dataset
TimeWindow = validation.TimeWindow
add_months = validation.add_months
load_feature_data = validation.load_feature_data
slice_window = validation.slice_window
train_predict_window = validation.train_predict_window


@dataclass(frozen=True)
class PredictionPeriod:
    """Half-open UTC period to predict."""

    start_utc: pd.Timestamp
    end_utc: pd.Timestamp


@dataclass(frozen=True)
class PeriodPredictionResult:
    """Artifacts produced by a period/window-specific prediction run."""

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
    forecast_setup: str = "hourly_period",
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
        model_run_name(model_module, model_options),
        forecast_setup,
    ]
    if forecast_setup in {"hourly_period", "period_average"}:
        name_parts.append(f"{period_days or 'period'}d_{block}")
    return base_folder / "__".join(name_parts)


def as_prediction_period(period: DeliveryPeriod) -> PredictionPeriod:
    """Convert a curve-translation period to a modelling prediction period."""

    return PredictionPeriod(start_utc=period.start_utc, end_utc=period.end_utc)


def period_slug(period: PredictionPeriod) -> str:
    """Build a stable file suffix for a delivery period."""

    return f"{period.start_utc:%Y%m%d}_{period.end_utc:%Y%m%d}"


def window_slug(window: TimeWindow) -> str:
    """Build a stable suffix that records both training and prediction windows."""

    return (
        f"train_{window.train_begin:%Y%m%d}_{window.train_end:%Y%m%d}"
        f"__predict_{window.test_begin:%Y%m%d}_{window.test_end:%Y%m%d}"
    )


def read_best_params(
    model_folder: str | Path,
    model_module,
    model_options: dict[str, object],
    feature_columns: list[str] | None = None,
) -> dict[str, object]:
    """Read best params from artifacts, falling back to validation summary or first grid row."""

    model_folder = Path(model_folder)
    best_params_path = model_folder / "best_params.json"
    if best_params_path.exists():
        return read_json(best_params_path)

    metadata_path = model_folder / "metadata.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        best_params = metadata.get("best_params")
        if isinstance(best_params, dict):
            return best_params

    summary_path = model_folder / "validation_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path, sep=None, engine="python")
        if "params" in summary.columns and not summary.empty:
            return json.loads(summary.iloc[0]["params"])

    return model_module.build_param_grid(
        **model_options,
        feature_columns=feature_columns,
    )[0]


def prediction_file_covers(predictions_path: str | Path, period: PredictionPeriod) -> bool:
    """Check whether a prediction file covers the full requested period."""

    predictions_path = Path(predictions_path)
    if not predictions_path.exists():
        return False
    predictions = pd.read_csv(predictions_path, sep=None, engine="python")
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

    predictions = pd.read_csv(predictions_path, sep=None, engine="python")
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


def save_window_prediction_result(
    prediction_result,
    test_data: pd.DataFrame,
    feature_path: str | Path,
    model_module,
    model_options: dict[str, object],
    best_params: dict[str, object],
    output_folder: str | Path,
    artifact_subfolder: str,
    forecast_setup: str,
    feature_policy: str,
    feature_columns: list[str],
    period_days: int,
    block: str,
    window: TimeWindow,
) -> PeriodPredictionResult:
    """Save one trained model, its predictions, metrics, and metadata.

    The same format is used for day-ahead simulations, period forecasts, and
    curve-translation retrains. Keeping this in one place prevents the classic
    "same prediction saved three different ways" mess.
    """

    output_folder = Path(output_folder)
    artifact_folder = output_folder / artifact_subfolder
    artifact_folder.mkdir(parents=True, exist_ok=True)

    predictions_path = artifact_folder / "predictions.csv"
    metrics_path = artifact_folder / "metrics.csv"
    model_path = artifact_folder / "model.joblib"
    metadata_path = artifact_folder / "metadata.json"

    predictions = prediction_result.predictions.copy()
    if forecast_setup == "period_average" and "period_end" in test_data.columns:
        predictions["period_end"] = test_data["period_end"]
        predictions["prediction_granularity"] = "period_average"
    else:
        predictions["prediction_granularity"] = "hourly"
    predictions["block"] = block
    predictions.to_csv(predictions_path, index=False, sep=";")

    metrics_frame = pd.DataFrame([prediction_result.metric_row])
    numeric_columns = metrics_frame.select_dtypes("number").columns
    metrics_frame[numeric_columns] = metrics_frame[numeric_columns].round(2)
    metrics_frame.to_csv(metrics_path, index=False, sep=";")

    metadata_params = {
        key: value
        for key, value in best_params.items()
        if key != "params_by_hour"
    }
    joblib.dump(prediction_result.model_state, model_path)
    write_json(
        {
            "model": model_module.MODEL_NAME,
            "model_options": model_options,
            "params": metadata_params,
            "feature_csv": str(feature_path),
            "forecast_setup": forecast_setup,
            "feature_policy": feature_policy,
            "period_days": period_days,
            "block": block,
            "feature_columns": feature_columns,
            "train_begin": window.train_begin,
            "train_end": window.train_end,
            "prediction_begin": window.test_begin,
            "prediction_end": window.test_end,
            "predictions_path": str(predictions_path),
            "metrics_path": str(metrics_path),
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


def train_and_predict_window(
    feature_path: str | Path,
    model_name: str,
    model_options: dict[str, object],
    output_folder: str | Path,
    window: TimeWindow,
    forecast_setup: str = "hourly_period",
    period_days: int | None = None,
    block: str = "baseload",
    artifact_parent: str = "window_predictions",
    artifact_subfolder: str | None = None,
) -> PeriodPredictionResult:
    """Train on one explicit window and predict one explicit window.

    Mathematically:

    ``train = [window.train_begin, window.train_end)``
    ``predict = [window.test_begin, window.test_end)``

    This is intentionally similar to one fold of rolling validation, but it is
    user-directed instead of automatically generated.
    """

    model_module = load_model_module(model_name)
    feature_table = load_feature_data(feature_path)
    if period_days is None:
        period_days = max(1, int((window.test_end - window.test_begin) / pd.Timedelta(days=1)))
    modelling_dataset = build_modelling_dataset(
        feature_table=feature_table,
        model_name=model_module.MODEL_NAME,
        forecast_setup=forecast_setup,
        period_days=period_days,
        block=block,
    )
    table = modelling_dataset.table
    feature_columns = modelling_dataset.feature_columns

    data_begin = table[constants.TIMESTAMP_COLUMN].min()
    data_end = table[constants.TIMESTAMP_COLUMN].max() + pd.Timedelta(hours=1)
    if window.train_begin < data_begin:
        raise ValueError(
            "Requested training window starts before the feature table. "
            f"Need data from {window.train_begin}, but feature data starts at {data_begin}."
        )
    if window.test_end > data_end:
        raise ValueError(
            "Requested prediction window ends after the feature table. "
            f"Need data until {window.test_end}, but feature data ends at {data_end}."
        )

    train_data = slice_window(table, window.train_begin, window.train_end)
    test_data = slice_window(table, window.test_begin, window.test_end)
    if train_data.empty:
        raise ValueError("Training window is empty.")
    if test_data.empty:
        raise ValueError("Requested prediction window is empty in the feature table.")

    output_folder = Path(output_folder)
    grid_options = {
        **model_options,
        "forecast_setup": forecast_setup,
        "period_days": period_days,
        "block": block,
    }
    best_params = read_best_params(
        output_folder,
        model_module,
        grid_options,
        feature_columns=feature_columns,
    )
    prediction_result = train_predict_window(
        table=table,
        model_module=model_module,
        params=best_params,
        window=TimeWindow(
            train_begin=window.train_begin,
            train_end=window.train_end,
            test_begin=window.test_begin,
            test_end=window.test_end,
        ),
        split_name="window_prediction",
        fold_number=1,
        feature_columns=feature_columns,
        target_column=modelling_dataset.target_column,
    )

    return save_window_prediction_result(
        prediction_result=prediction_result,
        test_data=test_data,
        feature_path=feature_path,
        model_module=model_module,
        model_options=model_options,
        best_params=best_params,
        output_folder=output_folder,
        artifact_subfolder=artifact_subfolder or f"{artifact_parent}/{window_slug(window)}",
        forecast_setup=forecast_setup,
        feature_policy=modelling_dataset.feature_policy,
        feature_columns=feature_columns,
        period_days=period_days,
        block=block,
        window=window,
    )


def train_and_predict_period(
    feature_path: str | Path,
    model_name: str,
    model_options: dict[str, object],
    output_folder: str | Path,
    period: PredictionPeriod,
    train_months: int = constants.TRAIN_MONTHS,
    forecast_setup: str = "hourly_period",
    period_days: int | None = None,
    block: str = "baseload",
) -> PeriodPredictionResult:
    """Train on the months immediately before a requested delivery period.

    This is the curve-translation convenience wrapper:

    ``train = [period.start - train_months, period.start)``
    ``predict = [period.start, period.end)``
    """

    window = TimeWindow(
        train_begin=add_months(period.start_utc, -train_months),
        train_end=period.start_utc,
        test_begin=period.start_utc,
        test_end=period.end_utc,
    )
    return train_and_predict_window(
        feature_path=feature_path,
        model_name=model_name,
        model_options=model_options,
        output_folder=output_folder,
        window=window,
        forecast_setup=forecast_setup,
        period_days=period_days,
        block=block,
        artifact_subfolder=f"period_predictions/{period_slug(period)}",
    )


def get_or_create_predictions(
    feature_path: str | Path,
    model_name: str,
    model_options: dict[str, object],
    period: DeliveryPeriod,
    output_base_folder: str | Path | None = None,
    force_retrain: bool = False,
    forecast_setup: str = "hourly_period",
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
        forecast_setup=forecast_setup,
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

    validation_predictions_path = model_folder / "predictions.csv"
    validation_metrics_path = model_folder / "validation_summary.csv"
    if not force_retrain and prediction_file_covers(validation_predictions_path, modelling_period):
        if not validation_metrics_path.exists():
            raise ValueError(f"Missing metrics file: {validation_metrics_path}")
        check_prediction_coverage(
            validation_predictions_path,
            modelling_period,
            curve_constants.MIN_PREDICTION_COVERAGE,
        )
        return PredictionSource(
            predictions_path=validation_predictions_path,
            metrics_path=validation_metrics_path,
            model_folder=model_folder,
            source="existing_validation_predictions",
            retrained=False,
        )

    period_result = train_and_predict_period(
        feature_path=feature_path,
        model_name=model_name,
        model_options=model_options,
        output_folder=model_folder,
        period=modelling_period,
        forecast_setup=forecast_setup,
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
