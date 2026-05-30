"""Train a selected model on a rolling window and predict a requested period."""

from __future__ import annotations

import importlib
import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from pipeline_helpers.modelling import constants
from pipeline_helpers.modelling.metrics import calculate_metrics
from pipeline_helpers.modelling.validation import add_months, load_feature_data, slice_window


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


def load_model_module(model_name: str):
    """Import a model module from ``pipeline_helpers.modelling``."""

    return importlib.import_module(f"pipeline_helpers.modelling.{model_name}")


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


def prediction_file_covers(predictions_path: str | Path, period: PredictionPeriod) -> bool:
    """Check whether a prediction file covers the full requested period."""

    predictions_path = Path(predictions_path)
    if not predictions_path.exists():
        return False
    predictions = pd.read_csv(predictions_path)
    if predictions.empty or "timestamp_utc" not in predictions.columns:
        return False
    timestamps = pd.to_datetime(predictions["timestamp_utc"], utc=True)
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
) -> PeriodPredictionResult:
    """Train on the previous rolling window and predict the requested period."""

    model_module = load_model_module(model_name)
    table = load_feature_data(feature_path)
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
    model_state = model_module.train(train_data, best_params)
    y_pred = model_module.predict(model_state, test_data, best_params)
    y_true = test_data[constants.TARGET_COLUMN]
    metrics = calculate_metrics(y_true, y_pred)

    period_folder = output_folder / "period_predictions" / period_slug(period)
    period_folder.mkdir(parents=True, exist_ok=True)

    predictions_path = period_folder / "predictions.csv"
    metrics_path = period_folder / "metrics.csv"
    model_path = period_folder / "model.joblib"
    metadata_path = period_folder / "metadata.json"

    predictions = pd.DataFrame(
        {
            constants.TIMESTAMP_COLUMN: test_data[constants.TIMESTAMP_COLUMN],
            "model": model_module.MODEL_NAME,
            "params": json.dumps(best_params, sort_keys=True),
            "split": "period_prediction",
            "fold": 1,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    predictions.to_csv(predictions_path, index=False)
    pd.DataFrame([{**metrics}]).round(2).to_csv(metrics_path, index=False)
    joblib.dump(model_state, model_path)
    write_json(
        {
            "model": model_module.MODEL_NAME,
            "model_options": model_options,
            "params": best_params,
            "feature_csv": str(feature_path),
            "train_begin": train_begin,
            "train_end": train_end,
            "prediction_begin": period.start_utc,
            "prediction_end": period.end_utc,
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
