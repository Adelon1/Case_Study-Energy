"""Pipeline step: run rolling validation and save compact model artifacts.

Example:
    .venv/bin/python pipeline_steps/02_validate_model.py

Outputs are written to ``models/<dataset-name>/<run-name>``.
"""

from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

constants = importlib.import_module("pipeline_helpers.02_modelling.00_constants")
model_io = importlib.import_module("pipeline_helpers.02_modelling.04_model_io")
model_support = importlib.import_module("pipeline_helpers.02_modelling.05_model_support")
modelling_dataset = importlib.import_module("pipeline_helpers.02_modelling.01_modelling_dataset")
prediction_bands = importlib.import_module("pipeline_helpers.02_modelling.03_prediction_bands")
validation = importlib.import_module("pipeline_helpers.02_modelling.09_validation")
metric_functions = importlib.import_module("pipeline_helpers.02_modelling.02_metrics")

default_models_base_folder = model_io.default_models_base_folder
write_json = model_io.write_json
load_model_module = model_support.load_model_module
model_run_name = model_support.model_run_name
model_state_diagnostics = model_support.model_state_diagnostics
build_modelling_dataset = modelling_dataset.build_modelling_dataset
add_prediction_bands = prediction_bands.add_prediction_bands
band_coverage = prediction_bands.band_coverage
residual_quantiles_by_hour = prediction_bands.residual_quantiles_by_hour
calculate_metrics = metric_functions.calculate_metrics
TimeWindow = validation.TimeWindow
average_metrics_by_params = validation.average_metrics_by_params
build_validation_windows = validation.build_validation_windows
load_feature_data = validation.load_feature_data
run_rolling_validation = validation.run_rolling_validation
select_predictions_for_params = validation.select_predictions_for_params
train_predict_window = validation.train_predict_window

SELECTED_PER_HOUR_PARAMS = json.dumps({"selection": "per_hour_validated"}, sort_keys=True)


def parse_command_line_arguments() -> SimpleNamespace:
    """Return default settings; the user edits them through prompts."""

    return SimpleNamespace(
        features="data/03_processed/germany_modelling_2021_2026/germany_model_features.csv",
        model="baseline_model",
        output_folder=None,
        forecast_setup="hourly_day_ahead",
        period_days=1,
        block="baseload",
        regularization=None,
        target_transform=None,
    )


def ask(prompt: str, default: str | None = None) -> str:
    """Ask one terminal question."""

    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or (default or "")


def ask_choice(prompt: str, choices: list[str], default: str) -> str:
    """Ask until the answer is one of the allowed choices."""

    choices_text = ", ".join(choices)
    while True:
        answer = ask(f"{prompt} ({choices_text})", default)
        if answer in choices:
            return answer
        print(f"Please choose one of: {choices_text}")


def apply_interactive_settings(args: SimpleNamespace) -> SimpleNamespace:
    """Fill validation settings interactively."""

    args.features = ask("Feature CSV", args.features)
    args.model = ask("Model", args.model)
    args.forecast_setup = ask_choice(
        "Forecast setup",
        constants.FORECAST_SETUPS,
        args.forecast_setup,
    )
    if args.forecast_setup == "hourly_day_ahead":
        args.period_days = 1
        args.block = "baseload"
    elif args.forecast_setup == "hourly_period":
        args.period_days = int(ask("Period length in days", str(args.period_days)))
        args.block = "baseload"
    else:
        args.period_days = int(ask("Period length in days", str(args.period_days)))
        args.block = ask_choice("Block", ["baseload", "peakload", "offpeak"], args.block)
    if args.model == "lear_model":
        args.regularization = ask_choice(
            "Regularization",
            ["lasso", "elasticnet", "ridge"],
            args.regularization or "lasso",
        )
    args.target_transform = ask_choice(
        "Target transform",
        ["raw", "asinh"],
        args.target_transform or "raw",
    )
    return args


def build_model_options(args: SimpleNamespace) -> dict[str, object]:
    """Collect model-specific settings from command-line arguments."""

    options: dict[str, object] = {}
    if args.regularization is not None:
        options["regularization"] = args.regularization
    if args.target_transform is not None:
        options["target_transform"] = args.target_transform
    return options


def build_grid_options(
    model_options: dict[str, object],
    args: SimpleNamespace,
) -> dict[str, object]:
    """Add dataset context for models whose parameter grid depends on it."""

    return {
        **model_options,
        "forecast_setup": args.forecast_setup,
        "period_days": args.period_days,
        "block": args.block,
    }


def summarize_validation(metrics: pd.DataFrame) -> pd.DataFrame:
    """Average validation metrics per parameter setting."""

    return (
        average_metrics_by_params(metrics)
        .merge(metrics[["model", "params"]].drop_duplicates(), on="params", how="left")
        [["model", "params", *constants.METRIC_NAMES]]
        .sort_values(constants.MODEL_SELECTION_METRICS)
        .reset_index(drop=True)
    )


def selected_prediction_metrics(
    predictions: pd.DataFrame,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate metrics for the stitched selected validation predictions."""

    selected_rows = []
    for fold, fold_predictions in predictions.groupby("fold", dropna=False):
        y_true = fold_predictions["y_true"]
        y_pred = fold_predictions["y_pred"]
        metric_values = calculate_metrics(y_true, y_pred)
        timestamp = pd.to_datetime(fold_predictions[constants.TIMESTAMP_COLUMN], utc=True)
        selected_rows.append(
            {
                "model": model_name,
                "params": SELECTED_PER_HOUR_PARAMS,
                "split": "selected_validation",
                "fold": fold,
                "train_begin": pd.NaT,
                "train_end": pd.NaT,
                "test_begin": timestamp.min(),
                "test_end": timestamp.max() + pd.Timedelta(hours=1),
                "n_test_rows": len(fold_predictions),
                "n_predicted_rows": int(y_pred.notna().sum()),
                "prediction_coverage": float(y_pred.notna().mean()),
                **metric_values,
            }
        )

    fold_metrics = pd.DataFrame(selected_rows)
    summary_metrics = calculate_metrics(predictions["y_true"], predictions["y_pred"])
    summary = pd.DataFrame(
        [
            {
                "model": model_name,
                "params": SELECTED_PER_HOUR_PARAMS,
                **summary_metrics,
            }
        ]
    )
    return fold_metrics, summary


def read_first_metric(path: Path, metric_name: str) -> float | None:
    """Read the first value for a metric from a CSV if available."""

    if not path.exists():
        return None
    table = pd.read_csv(path, sep=None, engine="python")
    if metric_name not in table.columns or table.empty:
        return None
    values = pd.to_numeric(table[metric_name], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[0])


def add_relative_mae_vs_baseline(
    table: pd.DataFrame,
    baseline_mae: float | None,
) -> pd.DataFrame:
    """Add rMAE against the baseline model when baseline metrics exist."""

    if baseline_mae is None or baseline_mae == 0 or "mae" not in table.columns:
        return table
    result = table.copy()
    result["relative_mae_vs_baseline"] = result["mae"] / baseline_mae
    return result


def baseline_summary_path(
    output_base_folder: Path,
    forecast_setup: str,
    period_days: int,
    block: str,
) -> Path:
    """Return expected baseline validation-summary path."""

    name_parts = ["baseline_model", forecast_setup]
    if forecast_setup in {"hourly_period", "period_average"}:
        name_parts.append(f"{period_days}d_{block}")
    baseline_folder = output_base_folder / "__".join(name_parts)
    return baseline_folder / "validation_summary.csv"


def expand_params_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Replace the escaped JSON ``params`` column with plain readable columns."""

    if "params" not in table.columns:
        return table

    parsed = table["params"].apply(lambda value: json.loads(value) if isinstance(value, str) else {})
    param_keys: list[str] = []
    for entry in parsed:
        for key in entry:
            if key not in param_keys:
                param_keys.append(key)

    result = table.drop(columns=["params"]).copy()
    for key in param_keys:
        result[key] = parsed.apply(lambda entry: entry.get(key))

    front = [column for column in ["model", *param_keys] if column in result.columns]
    rest = [column for column in result.columns if column not in front]
    return result[front + rest]


def format_result_table(table: pd.DataFrame) -> pd.DataFrame:
    """Format metric tables for readable CSV and terminal output."""

    formatted = table.copy()
    if "split" in formatted.columns:
        formatted = formatted.drop(columns=["split"])
    formatted = expand_params_columns(formatted)

    date_columns = ["train_begin", "train_end", "test_begin", "test_end"]
    for column in date_columns:
        if column in formatted.columns:
            formatted[column] = pd.to_datetime(formatted[column], utc=True).dt.strftime("%Y-%m-%d")

    numeric_columns = formatted.select_dtypes(include="number").columns
    float_columns = [
        column
        for column in numeric_columns
        if not column.startswith("n_") and column != "fold"
    ]
    formatted[float_columns] = formatted[float_columns].round(2)
    return formatted


def write_result_csv(table: pd.DataFrame, path: Path) -> None:
    """Write result CSVs in a spreadsheet-friendly format."""

    format_result_table(table).to_csv(path, index=False, sep=";")


def write_predictions_csv(predictions: pd.DataFrame, path: Path) -> None:
    """Write the out-of-sample predictions as a slim, readable file.

    Only the columns a reader (or a plotting/submission step) actually needs:
    timestamp, delivery hour, fold, actual, forecast, and the forecast band.
    """

    column_order = [
        constants.TIMESTAMP_COLUMN,
        "local_hour",
        "fold",
        "y_true",
        "y_pred",
        "y_pred_lower",
        "y_pred_upper",
    ]
    columns = [column for column in column_order if column in predictions.columns]
    clean = predictions[columns].copy()
    if constants.TIMESTAMP_COLUMN in clean.columns:
        clean[constants.TIMESTAMP_COLUMN] = pd.to_datetime(
            clean[constants.TIMESTAMP_COLUMN], utc=True
        ).dt.strftime("%Y-%m-%d %H:%M")
    float_columns = [
        column
        for column in ["y_true", "y_pred", "y_pred_lower", "y_pred_upper"]
        if column in clean.columns
    ]
    clean[float_columns] = clean[float_columns].round(2)
    clean.to_csv(path, index=False, sep=";")


def remove_old_validation_outputs(output_folder: Path) -> None:
    """Delete verbose validation artifacts from older runs in this folder."""

    old_file_names = [
        "final_holdout_metrics.csv",
        "final_holdout_predictions.csv",
        "best_params.json",
        "model_metadata.json",
        "final_holdout_model.joblib",
    ]
    for file_name in old_file_names:
        path = output_folder / file_name
        if path.exists() and path.is_file():
            path.unlink()


def save_final_model_artifacts(
    table: pd.DataFrame,
    feature_path: Path,
    output_folder: Path,
    model_module,
    model_options: dict[str, object],
    best_params: dict[str, object],
    feature_columns: list[str],
    target_column: str,
    saved_window: TimeWindow,
    saved_predictions: pd.DataFrame,
    validation_summary: pd.DataFrame,
    forecast_setup: str,
    feature_policy: str,
    period_days: int,
    block: str,
) -> tuple[Path, Path, Path]:
    """Save the last-fold model, its predictions, and validation metadata."""

    saved_result = train_predict_window(
        table=table,
        model_module=model_module,
        params=best_params,
        window=saved_window,
        split_name="last_validation_fold",
        fold_number=1,
        feature_columns=feature_columns,
        target_column=target_column,
    )

    model_path = output_folder / "model.joblib"
    predictions_path = output_folder / "predictions.csv"
    metadata_path = output_folder / "metadata.json"
    metadata_best_params = {
        key: value
        for key, value in best_params.items()
        if key != "params_by_hour"
    }
    model_feature_columns = list(getattr(saved_result.model_state, "feature_columns", feature_columns))

    joblib.dump(saved_result.model_state, model_path)
    write_predictions_csv(saved_predictions, predictions_path)
    write_json(
        {
            "model": model_module.MODEL_NAME,
            "model_options": model_options,
            "best_params": metadata_best_params,
            "feature_csv": str(feature_path),
            "forecast_setup": forecast_setup,
            "feature_policy": feature_policy,
            "period_days": period_days,
            "block": block,
            "feature_columns": model_feature_columns,
            "target_column": target_column,
            "validation_method": "rolling fixed-window validation",
            "selection_metric": constants.MODEL_SELECTION_METRICS[0],
            "validation_constants": {
                "train_months": constants.TRAIN_MONTHS,
                "test_months": constants.TEST_MONTHS,
                "step_months": constants.STEP_MONTHS,
            },
            "saved_model": {
                "meaning": "model trained on the last validation fold using best rolling-validation params",
                "train_begin": saved_window.train_begin,
                "train_end": saved_window.train_end,
                "test_begin": saved_window.test_begin,
                "test_end": saved_window.test_end,
            },
            "predictions_file": {
                "path": str(predictions_path),
                "meaning": "out-of-sample rolling-validation predictions for the selected params across all folds",
                "band_quantiles": [constants.BAND_LOWER_QUANTILE, constants.BAND_UPPER_QUANTILE],
                "band_coverage": band_coverage(saved_predictions),
            },
            "best_validation_row": validation_summary.iloc[0].to_dict(),
            "training_diagnostics": model_state_diagnostics(saved_result.model_state),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_note": "For another requested delivery period, retrain on that period's rolling training window.",
        },
        metadata_path,
    )
    return model_path, predictions_path, metadata_path


def main() -> None:
    """Run rolling validation and write compact result artifacts."""

    args = apply_interactive_settings(parse_command_line_arguments())
    feature_path = Path(args.features)
    output_base_folder = (
        Path(args.output_folder) if args.output_folder else default_models_base_folder(feature_path)
    )
    model_module = load_model_module(args.model)
    model_options = build_model_options(args)
    grid_options = build_grid_options(model_options, args)
    output_name_parts = [
        model_run_name(model_module, model_options),
        args.forecast_setup,
    ]
    if args.forecast_setup in {"hourly_period", "period_average"}:
        output_name_parts.append(f"{args.period_days}d_{args.block}")
    output_folder = output_base_folder / "__".join(output_name_parts)
    output_folder.mkdir(parents=True, exist_ok=True)
    remove_old_validation_outputs(output_folder)

    feature_table = load_feature_data(feature_path)
    modelling_dataset = build_modelling_dataset(
        feature_table=feature_table,
        model_name=model_module.MODEL_NAME,
        forecast_setup=args.forecast_setup,
        period_days=args.period_days,
        block=args.block,
    )
    table = modelling_dataset.table
    feature_columns = modelling_dataset.feature_columns
    validation_result = run_rolling_validation(
        table,
        model_module,
        model_options=grid_options,
        show_progress=True,
        feature_columns=feature_columns,
        target_column=modelling_dataset.target_column,
    )
    validation_summary_path = output_folder / "validation_summary.csv"
    validation_metrics_path = output_folder / "validation_metrics.csv"

    validation_windows = build_validation_windows(table)
    last_validation_window = validation_windows[-1]
    saved_predictions = select_predictions_for_params(
        model_module,
        validation_result.predictions,
        validation_result.best_params,
    )
    if saved_predictions.empty:
        saved_predictions = train_predict_window(
            table=table,
            model_module=model_module,
            params=validation_result.best_params,
            window=last_validation_window,
            split_name="last_validation_fold",
            fold_number=1,
            feature_columns=feature_columns,
            target_column=modelling_dataset.target_column,
        ).predictions

    validation_metrics = validation_result.metrics.copy()
    validation_summary = summarize_validation(validation_result.metrics)
    if "params_by_hour" in validation_result.best_params:
        selected_fold_metrics, selected_summary = selected_prediction_metrics(
            saved_predictions,
            model_module.MODEL_NAME,
        )
        validation_metrics = pd.concat(
            [selected_fold_metrics, validation_metrics],
            ignore_index=True,
        )
        validation_summary = pd.concat(
            [selected_summary, validation_summary],
            ignore_index=True,
        )

    baseline_path = baseline_summary_path(
        output_base_folder,
        modelling_dataset.forecast_setup,
        args.period_days,
        args.block,
    )
    if model_module.MODEL_NAME != "baseline_model":
        validation_summary = add_relative_mae_vs_baseline(
            validation_summary,
            read_first_metric(baseline_path, "mae"),
        )

    write_result_csv(validation_summary, validation_summary_path)
    write_result_csv(validation_metrics, validation_metrics_path)

    band_offsets = residual_quantiles_by_hour(saved_predictions)
    saved_predictions = add_prediction_bands(saved_predictions, band_offsets)
    coverage = band_coverage(saved_predictions)

    model_path, predictions_path, metadata_path = save_final_model_artifacts(
        table,
        feature_path,
        output_folder,
        model_module,
        model_options,
        validation_result.best_params,
        feature_columns,
        modelling_dataset.target_column,
        last_validation_window,
        saved_predictions,
        validation_summary,
        modelling_dataset.forecast_setup,
        modelling_dataset.feature_policy,
        args.period_days,
        args.block,
    )

    print(f"Model: {model_module.MODEL_NAME}")
    print(f"Forecast setup: {modelling_dataset.forecast_setup}")
    print(f"Feature policy: {modelling_dataset.feature_policy}")
    print(f"Selected features: {len(feature_columns)}")
    print(f"Best params from rolling validation: {validation_result.best_params}")
    print(
        f"Forecast band P{int(constants.BAND_LOWER_QUANTILE * 100)}-"
        f"P{int(constants.BAND_UPPER_QUANTILE * 100)} coverage: {coverage:.1%}"
    )
    print("\nValidation summary:")
    print(format_result_table(validation_summary).to_string(index=False))
    print("\nSaved outputs:")
    print(f"  validation summary: {validation_summary_path}")
    print(f"  validation metrics: {validation_metrics_path}")
    print(f"  out-of-sample predictions: {predictions_path}")
    print(f"  saved model: {model_path}")
    print(f"  metadata: {metadata_path}")


if __name__ == "__main__":
    main()
