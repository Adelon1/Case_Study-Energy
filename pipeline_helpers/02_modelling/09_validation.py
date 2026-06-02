"""Rolling-window validation helpers for model modules."""

from __future__ import annotations

import importlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pandas as pd

constants = importlib.import_module("pipeline_helpers.02_modelling.00_constants")
metrics = importlib.import_module("pipeline_helpers.02_modelling.02_metrics")

calculate_metrics = metrics.calculate_metrics


@dataclass(frozen=True)
class TimeWindow:
    """One train/test split."""

    train_begin: pd.Timestamp
    train_end: pd.Timestamp
    test_begin: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class ValidationResult:
    """Validation metrics, predictions, and selected parameters."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame
    best_params: dict[str, object]


@dataclass(frozen=True)
class TrainPredictResult:
    """One trained model, its predictions, and its metric row."""

    model_state: object
    metric_row: dict[str, object]
    predictions: pd.DataFrame


def load_feature_data(path: str | Path) -> pd.DataFrame:
    """Load a modelling feature CSV and sort it by timestamp."""

    table = pd.read_csv(path)
    table[constants.TIMESTAMP_COLUMN] = pd.to_datetime(
        table[constants.TIMESTAMP_COLUMN],
        utc=True,
    )
    return table.sort_values(constants.TIMESTAMP_COLUMN).reset_index(drop=True)


def add_months(timestamp: pd.Timestamp, months: int) -> pd.Timestamp:
    """Shift a timestamp by calendar months."""

    return timestamp + pd.DateOffset(months=months)


def floor_to_month(timestamp: pd.Timestamp) -> pd.Timestamp:
    """Return the first instant of the timestamp's month."""

    return pd.Timestamp(timestamp.strftime("%Y-%m-01"), tz=timestamp.tz)


def ceil_to_next_month(timestamp: pd.Timestamp) -> pd.Timestamp:
    """Return the next month start unless the timestamp is already at one."""

    month_begin = floor_to_month(timestamp)
    if timestamp == month_begin:
        return month_begin
    return add_months(month_begin, 1)


def infer_complete_month_range(table: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Infer the clean calendar-month data range available for modelling.

    Partial first and last months are excluded. The returned end timestamp is
    exclusive.
    """

    timestamps = table[constants.TIMESTAMP_COLUMN]
    data_begin = ceil_to_next_month(timestamps.min())
    last_timestamp = timestamps.max()
    next_month_begin = add_months(floor_to_month(last_timestamp), 1)

    if last_timestamp + pd.Timedelta(hours=1) == next_month_begin:
        data_end = next_month_begin
    else:
        data_end = floor_to_month(last_timestamp)

    if data_end <= data_begin:
        raise ValueError("Not enough complete calendar months available for validation.")

    return data_begin, data_end


def slice_window(table: pd.DataFrame, begin: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Return rows inside a half-open timestamp window."""

    timestamps = table[constants.TIMESTAMP_COLUMN]
    return table.loc[(timestamps >= begin) & (timestamps < end)].reset_index(drop=True)


def build_validation_windows(
    table: pd.DataFrame,
    train_months: int = constants.TRAIN_MONTHS,
    test_months: int = constants.TEST_MONTHS,
    step_months: int = constants.STEP_MONTHS,
) -> list[TimeWindow]:
    """Build rolling train/test windows across the complete modelling range."""

    data_begin, data_end = infer_complete_month_range(table)
    train_begin = data_begin
    windows: list[TimeWindow] = []

    while True:
        train_end = add_months(train_begin, train_months)
        test_end = add_months(train_end, test_months)
        if test_end > data_end:
            break
        windows.append(
            TimeWindow(
                train_begin=train_begin,
                train_end=train_end,
                test_begin=train_end,
                test_end=test_end,
            )
        )
        train_begin = add_months(train_begin, step_months)

    if not windows:
        raise ValueError(
            "No validation windows could be built. Need more history or shorter train/test windows."
        )
    return windows


def params_to_string(params: dict[str, object]) -> str:
    """Serialize model parameters for result tables."""

    return json.dumps(params, sort_keys=True)


def get_model_param_grid(
    model_module: ModuleType,
    model_options: dict[str, object] | None = None,
    feature_columns: list[str] | None = None,
) -> list[dict[str, object]]:
    """Return the parameter grid declared by a model module."""

    return model_module.build_param_grid(
        **(model_options or {}),
        feature_columns=feature_columns,
    )


def average_metrics_by_params(metrics: pd.DataFrame) -> pd.DataFrame:
    """Average validation metrics per parameter setting."""

    metric_columns = [
        column
        for column in constants.METRIC_NAMES
        if column in metrics.columns
    ]
    return metrics.groupby("params", as_index=False)[metric_columns].mean()


def choose_best_params(metrics: pd.DataFrame) -> dict[str, object]:
    """Choose parameters by MAE, with stress metrics as tie-breakers.

    MAE remains the primary objective because it best represents normal hourly
    price-level accuracy. The additional metrics matter when two settings have
    similar MAE: the chosen setting should also behave better in stressed price
    periods and avoid large misses.
    """

    average_metrics = average_metrics_by_params(metrics)
    sort_columns = [
        column
        for column in constants.MODEL_SELECTION_METRICS
        if column in average_metrics.columns
    ]
    best_params_string = average_metrics.sort_values(sort_columns).iloc[0]["params"]
    return json.loads(best_params_string)


def choose_best_params_for_model(
    model_module: ModuleType,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, object]:
    """Let a model customize parameter selection without hardcoding model names."""

    if hasattr(model_module, "choose_best_params"):
        return model_module.choose_best_params(metrics, predictions)
    return choose_best_params(metrics)


def select_predictions_for_params(
    model_module: ModuleType,
    predictions: pd.DataFrame,
    best_params: dict[str, object],
) -> pd.DataFrame:
    """Return validation predictions corresponding to the selected parameters."""

    if hasattr(model_module, "select_validation_predictions"):
        return model_module.select_validation_predictions(predictions, best_params)
    return predictions.loc[predictions["params"] == params_to_string(best_params)].copy()


def evaluate_model_on_window(
    table: pd.DataFrame,
    model_module: ModuleType,
    params: dict[str, object],
    window: TimeWindow,
    split_name: str,
    fold_number: int,
    feature_columns: list[str] | None = None,
    target_column: str = constants.TARGET_COLUMN,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Train and score one model configuration on one time window."""

    result = train_predict_window(
        table=table,
        model_module=model_module,
        params=params,
        window=window,
        split_name=split_name,
        fold_number=fold_number,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    prediction_coverage = float(result.metric_row["prediction_coverage"])

    if prediction_coverage < constants.MIN_PREDICTION_COVERAGE:
        warnings.warn(
            f"{model_module.MODEL_NAME} prediction coverage below "
            f"{constants.MIN_PREDICTION_COVERAGE:.0%}: "
            f"{prediction_coverage:.2%} for {split_name} fold {fold_number}",
            RuntimeWarning,
        )

    return result.metric_row, result.predictions


def train_predict_window(
    table: pd.DataFrame,
    model_module: ModuleType,
    params: dict[str, object],
    window: TimeWindow,
    split_name: str,
    fold_number: int,
    feature_columns: list[str] | None = None,
    target_column: str = constants.TARGET_COLUMN,
) -> TrainPredictResult:
    """Train one model on one window and return metrics plus predictions."""

    train_data = slice_window(table, window.train_begin, window.train_end)
    test_data = slice_window(table, window.test_begin, window.test_end)
    model_params = params.copy()
    if feature_columns is not None:
        model_params["_feature_columns"] = feature_columns
    model_params["_target_column"] = target_column

    model_state = model_module.train(train_data, model_params)

    y_pred = model_module.predict(model_state, test_data, model_params)
    y_true = test_data[target_column]
    metrics = calculate_metrics(y_true, y_pred)
    n_test_rows = len(test_data)
    n_predicted_rows = int(y_pred.notna().sum())
    prediction_coverage = n_predicted_rows / n_test_rows if n_test_rows else 0.0

    metric_row = {
        "model": model_module.MODEL_NAME,
        "params": params_to_string(params),
        "split": split_name,
        "fold": fold_number,
        "train_begin": window.train_begin,
        "train_end": window.train_end,
        "test_begin": window.test_begin,
        "test_end": window.test_end,
        "n_test_rows": n_test_rows,
        "n_predicted_rows": n_predicted_rows,
        "prediction_coverage": prediction_coverage,
        **metrics,
    }

    predictions = pd.DataFrame(
        {
            constants.TIMESTAMP_COLUMN: test_data[constants.TIMESTAMP_COLUMN],
            "model": model_module.MODEL_NAME,
            "params": params_to_string(params),
            "split": split_name,
            "fold": fold_number,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    if "local_hour" in test_data.columns:
        predictions["local_hour"] = test_data["local_hour"]
    return TrainPredictResult(
        model_state=model_state,
        metric_row=metric_row,
        predictions=predictions,
    )


def run_rolling_validation(
    table: pd.DataFrame,
    model_module: ModuleType,
    model_options: dict[str, object] | None = None,
    show_progress: bool = False,
    train_months: int = constants.TRAIN_MONTHS,
    test_months: int = constants.TEST_MONTHS,
    step_months: int = constants.STEP_MONTHS,
    feature_columns: list[str] | None = None,
    target_column: str = constants.TARGET_COLUMN,
) -> ValidationResult:
    """Tune a model module over its parameter grid using rolling validation."""

    windows = build_validation_windows(table, train_months, test_months, step_months)
    param_grid = get_model_param_grid(
        model_module,
        model_options,
        feature_columns=feature_columns,
    )
    metric_rows: list[dict[str, object]] = []
    prediction_tables: list[pd.DataFrame] = []

    for params in param_grid:
        if show_progress:
            print(f"Validating params: {params}")
        for fold_number, window in enumerate(windows, start=1):
            metric_row, predictions = evaluate_model_on_window(
                table,
                model_module,
                params,
                window,
                split_name="validation",
                fold_number=fold_number,
                feature_columns=feature_columns,
                target_column=target_column,
            )
            metric_rows.append(metric_row)
            prediction_tables.append(predictions)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    best_params = choose_best_params_for_model(model_module, metrics, predictions)

    return ValidationResult(metrics=metrics, predictions=predictions, best_params=best_params)
