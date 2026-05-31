"""Rolling-window validation and final holdout testing for model modules."""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pandas as pd

from pipeline_helpers.modelling import constants
from pipeline_helpers.modelling.metrics import calculate_metrics


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
    """Build rolling windows while reserving the last test window as holdout."""

    data_begin, data_end = infer_complete_month_range(table)
    final_holdout_begin = add_months(data_end, -test_months)
    train_begin = data_begin
    windows: list[TimeWindow] = []

    while True:
        train_end = add_months(train_begin, train_months)
        test_end = add_months(train_end, test_months)
        if test_end > final_holdout_begin:
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


def build_final_holdout_window(
    table: pd.DataFrame,
    train_months: int = constants.TRAIN_MONTHS,
    test_months: int = constants.TEST_MONTHS,
) -> TimeWindow:
    """Build the final untouched holdout split from the end of the data."""

    data_begin, data_end = infer_complete_month_range(table)
    test_begin = add_months(data_end, -test_months)
    train_begin = add_months(test_begin, -train_months)
    if train_begin < data_begin:
        raise ValueError("Not enough history before the final holdout window.")

    return TimeWindow(
        train_begin=train_begin,
        train_end=test_begin,
        test_begin=test_begin,
        test_end=data_end,
    )


def params_to_string(params: dict[str, object]) -> str:
    """Serialize model parameters for result tables."""

    return json.dumps(params, sort_keys=True)


def get_model_param_grid(
    model_module: ModuleType,
    model_options: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Return the parameter grid declared by a model module."""

    if hasattr(model_module, "build_param_grid"):
        return model_module.build_param_grid(**(model_options or {}))
    return model_module.PARAM_GRID


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


def evaluate_model_on_window(
    table: pd.DataFrame,
    model_module: ModuleType,
    params: dict[str, object],
    window: TimeWindow,
    split_name: str,
    fold_number: int,
    feature_columns: list[str] | None = None,
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
) -> TrainPredictResult:
    """Train one model on one window and return metrics plus predictions."""

    train_data = slice_window(table, window.train_begin, window.train_end)
    test_data = slice_window(table, window.test_begin, window.test_end)
    model_params = params.copy()
    if feature_columns is not None:
        model_params["_feature_columns"] = feature_columns

    train_started = time.perf_counter()
    model_state = model_module.train(train_data, model_params)
    train_seconds = time.perf_counter() - train_started

    predict_started = time.perf_counter()
    y_pred = model_module.predict(model_state, test_data, model_params)
    predict_seconds = time.perf_counter() - predict_started
    y_true = test_data[constants.TARGET_COLUMN]
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
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "total_seconds": train_seconds + predict_seconds,
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
) -> ValidationResult:
    """Tune a model module over its parameter grid using rolling validation."""

    windows = build_validation_windows(table, train_months, test_months, step_months)
    param_grid = get_model_param_grid(model_module, model_options)
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
            )
            metric_rows.append(metric_row)
            prediction_tables.append(predictions)

    metrics = pd.DataFrame(metric_rows)
    best_params = choose_best_params(metrics)
    predictions = pd.concat(prediction_tables, ignore_index=True)

    return ValidationResult(metrics=metrics, predictions=predictions, best_params=best_params)


def run_final_holdout_test(
    table: pd.DataFrame,
    model_module: ModuleType,
    params: dict[str, object],
    train_months: int = constants.TRAIN_MONTHS,
    test_months: int = constants.TEST_MONTHS,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate chosen parameters on the untouched final holdout window."""

    window = build_final_holdout_window(table, train_months, test_months)
    metric_row, predictions = evaluate_model_on_window(
        table,
        model_module,
        params,
        window,
        split_name="final_holdout",
        fold_number=1,
        feature_columns=feature_columns,
    )
    return pd.DataFrame([metric_row]), predictions
