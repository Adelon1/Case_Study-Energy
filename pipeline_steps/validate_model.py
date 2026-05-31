"""Pipeline step: run rolling validation and final holdout testing.

Example:
    .venv/bin/python pipeline_steps/validate_model.py \
      --features data/processed/germany_modelling_2021_2026/germany_model_features.csv \
      --model baseline_week_lag

    .venv/bin/python pipeline_steps/validate_model.py \
      --features data/processed/germany_modelling_2021_2026/germany_model_features.csv \
      --model lear_model \
      --regularization elasticnet \
      --target-transform raw

    .venv/bin/python pipeline_steps/validate_model.py \
      --features data/processed/germany_modelling_2021_2026/germany_model_features.csv \
      --model hist_gradient_boosting \
      --target-transform raw
      
Outputs are written to ``models/<dataset-name>/<run-name>`` unless
``--output-folder`` is provided.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_helpers.modelling import constants  # noqa: E402
from pipeline_helpers.modelling.model_io import default_models_base_folder, write_json  # noqa: E402
from pipeline_helpers.modelling.modelling_dataset import build_modelling_dataset  # noqa: E402
from pipeline_helpers.modelling.metrics import calculate_metrics  # noqa: E402
from pipeline_helpers.modelling.validation import (  # noqa: E402
    average_metrics_by_params,
    build_final_holdout_window,
    load_feature_data,
    run_final_holdout_test,
    run_rolling_validation,
    slice_window,
)


def parse_command_line_arguments() -> argparse.Namespace:
    """Read model validation settings from the command line."""

    parser = argparse.ArgumentParser(description="Validate a forecasting model.")
    parser.add_argument("--interactive", action="store_true", help="Ask for settings interactively.")
    parser.add_argument(
        "--features",
        default="data/processed/germany_modelling_2021_2026/germany_model_features.csv",
        help="Path to germany_model_features.csv.",
    )
    parser.add_argument(
        "--model",
        default="baseline_week_lag",
        help="Model module name inside pipeline_helpers/modelling.",
    )
    parser.add_argument(
        "--output-folder",
        default=None,
        help="Base folder for validation outputs. Defaults to models/<dataset-name>.",
    )
    parser.add_argument(
        "--target-option",
        choices=["A", "B"],
        default="A",
        help="Option A predicts hourly prices; Option B predicts period averages directly.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["day_ahead_full", "period_hourly_safe", "fundamentals_calendar_only"],
        default="day_ahead_full",
        help="Leakage-aware feature set for Option A.",
    )
    parser.add_argument(
        "--period-days",
        type=int,
        default=1,
        help="Delivery period length in days. Used for safe period features and Option B rows.",
    )
    parser.add_argument(
        "--block",
        choices=["baseload", "peakload", "offpeak"],
        default="baseload",
        help="Delivery block for Option B period-average targets.",
    )
    parser.add_argument(
        "--save-validation-predictions",
        action="store_true",
        help="Also save every validation-window prediction. This can be a large CSV.",
    )
    parser.add_argument(
        "--regularization",
        choices=["lasso", "elasticnet", "ridge"],
        default=None,
        help="Regularization family for models that support it. LEAR defaults to lasso.",
    )
    parser.add_argument(
        "--target-transform",
        choices=["raw", "asinh"],
        default=None,
        help="Target transform for models that support it. Defaults to raw where supported.",
    )
    return parser.parse_args()


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


def apply_interactive_settings(args: argparse.Namespace) -> argparse.Namespace:
    """Fill validation settings interactively."""

    args.features = ask("Feature CSV", args.features)
    args.model = ask("Model", args.model)
    args.target_option = ask_choice("Target option", ["A", "B"], args.target_option)
    if args.target_option == "A":
        args.feature_mode = ask_choice(
            "Feature mode",
            ["day_ahead_full", "period_hourly_safe", "fundamentals_calendar_only"],
            args.feature_mode,
        )
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


def load_model_module(model_name: str):
    """Import a model module from ``pipeline_helpers.modelling``."""

    return importlib.import_module(f"pipeline_helpers.modelling.{model_name}")


def build_model_options(args: argparse.Namespace) -> dict[str, object]:
    """Collect optional model settings from command-line arguments."""

    options: dict[str, object] = {}
    if args.regularization is not None:
        options["regularization"] = args.regularization
    if args.target_transform is not None:
        options["target_transform"] = args.target_transform
    return options


def output_folder_name(model_module, model_options: dict[str, object]) -> str:
    """Return the model-specific output folder name."""

    if hasattr(model_module, "output_folder_name"):
        return model_module.output_folder_name(**model_options)
    return model_module.MODEL_NAME


def summarize_validation(metrics: pd.DataFrame) -> pd.DataFrame:
    """Average validation metrics per parameter setting."""

    return (
        average_metrics_by_params(metrics)
        .merge(metrics[["model", "params"]].drop_duplicates(), on="params", how="left")
        [["model", "params", *constants.METRIC_NAMES]]
        .sort_values(constants.MODEL_SELECTION_METRICS)
        .reset_index(drop=True)
    )


def compact_metric_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Keep the command-line metric printout readable."""

    return metrics[["model", "params", *constants.METRIC_NAMES]]


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


def baseline_metric_paths(output_base_folder: Path) -> tuple[Path, Path]:
    """Return expected baseline summary and final-metrics paths."""

    baseline_folder = output_base_folder / "baseline_week_lag"
    return (
        baseline_folder / "validation_summary.csv",
        baseline_folder / "final_holdout_metrics.csv",
    )


def add_time_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    """Attach UTC year and month columns to prediction rows."""

    table = predictions.copy()
    timestamps = pd.to_datetime(table[constants.TIMESTAMP_COLUMN], utc=True)
    table["year"] = timestamps.dt.year
    table["month"] = timestamps.dt.month
    return table


def metric_breakdown_by_time(predictions: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Calculate metrics for prediction groups such as month or year."""

    rows: list[dict[str, object]] = []
    predictions = add_time_columns(predictions)
    for group_values, group in predictions.groupby(group_columns, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        metric_row = dict(zip(group_columns, group_values, strict=True))
        metric_row["n_rows"] = len(group)
        metric_row.update(calculate_metrics(group["y_true"], group["y_pred"]))
        rows.append(metric_row)
    return pd.DataFrame(rows)


def prediction_coverage_diagnostics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Extract compact prediction coverage diagnostics from metric rows."""

    columns = [
        "model",
        "params",
        "fold",
        "train_begin",
        "train_end",
        "test_begin",
        "test_end",
        "n_test_rows",
        "n_predicted_rows",
        "prediction_coverage",
    ]
    return metrics[[column for column in columns if column in metrics.columns]].copy()


def format_result_table(table: pd.DataFrame) -> pd.DataFrame:
    """Format metric tables for readable CSV and terminal output."""

    formatted = table.copy()
    if "split" in formatted.columns:
        formatted = formatted.drop(columns=["split"])

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


def model_state_diagnostics(model_state) -> dict[str, object]:
    """Extract optional diagnostics from fitted model state objects."""

    diagnostics: dict[str, object] = {}
    for attribute in ["fitted_hours", "n_train_rows_by_hour", "feature_columns"]:
        if hasattr(model_state, attribute):
            diagnostics[attribute] = getattr(model_state, attribute)
    return diagnostics


def save_final_model_artifacts(
    table: pd.DataFrame,
    feature_path: Path,
    output_folder: Path,
    model_module,
    model_options: dict[str, object],
    best_params: dict[str, object],
    feature_columns: list[str],
    target_option: str,
    feature_mode: str,
    period_days: int,
    block: str,
) -> tuple[Path, Path, Path]:
    """Train and save the final holdout model for later reuse."""

    window = build_final_holdout_window(table)
    train_data = slice_window(table, window.train_begin, window.train_end)
    training_params = best_params.copy()
    training_params["_feature_columns"] = feature_columns
    model_state = model_module.train(train_data, training_params)

    model_path = output_folder / "final_holdout_model.joblib"
    best_params_path = output_folder / "best_params.json"
    metadata_path = output_folder / "model_metadata.json"

    joblib.dump(model_state, model_path)
    write_json(best_params, best_params_path)
    write_json(
        {
            "model": model_module.MODEL_NAME,
            "model_options": model_options,
            "best_params": best_params,
            "feature_csv": str(feature_path),
            "target_option": target_option,
            "feature_mode": feature_mode,
            "period_days": period_days,
            "block": block,
            "feature_columns": feature_columns,
            "training_window": {
                "train_begin": window.train_begin,
                "train_end": window.train_end,
                "test_begin": window.test_begin,
                "test_end": window.test_end,
            },
            "training_diagnostics": model_state_diagnostics(model_state),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_note": (
                "This model is trained on the final holdout training window. "
                "For another requested delivery period, retrain on the previous "
                "rolling training window."
            ),
        },
        metadata_path,
    )
    return model_path, best_params_path, metadata_path


def main() -> None:
    """Run validation, final holdout, and write result CSVs."""

    args = parse_command_line_arguments()
    if args.interactive:
        args = apply_interactive_settings(args)
    feature_path = Path(args.features)
    output_base_folder = (
        Path(args.output_folder) if args.output_folder else default_models_base_folder(feature_path)
    )
    model_module = load_model_module(args.model)
    model_options = build_model_options(args)
    output_name_parts = [
        output_folder_name(model_module, model_options),
        f"option_{args.target_option.lower()}",
    ]
    if args.target_option == "A":
        output_name_parts.append(args.feature_mode)
    else:
        output_name_parts.append(f"{args.period_days}d_{args.block}")
    output_folder = output_base_folder / "__".join(output_name_parts)
    output_folder.mkdir(parents=True, exist_ok=True)

    feature_table = load_feature_data(feature_path)
    modelling_dataset = build_modelling_dataset(
        feature_table=feature_table,
        model_name=model_module.MODEL_NAME,
        target_option=args.target_option,
        feature_mode=args.feature_mode,
        period_days=args.period_days,
        block=args.block,
    )
    table = modelling_dataset.table
    validation_result = run_rolling_validation(
        table,
        model_module,
        model_options=model_options,
        show_progress=True,
        feature_columns=modelling_dataset.feature_columns,
    )
    final_metrics, final_predictions = run_final_holdout_test(
        table,
        model_module,
        validation_result.best_params,
        feature_columns=modelling_dataset.feature_columns,
    )
    validation_summary = summarize_validation(validation_result.metrics)
    baseline_validation_path, baseline_final_path = baseline_metric_paths(output_base_folder)
    if model_module.MODEL_NAME != "baseline_week_lag":
        validation_summary = add_relative_mae_vs_baseline(
            validation_summary,
            read_first_metric(baseline_validation_path, "mae"),
        )
        final_metrics = add_relative_mae_vs_baseline(
            final_metrics,
            read_first_metric(baseline_final_path, "mae"),
        )

    validation_metrics_path = output_folder / "validation_metrics.csv"
    validation_summary_path = output_folder / "validation_summary.csv"
    validation_predictions_path = output_folder / "validation_predictions.csv"
    final_metrics_path = output_folder / "final_holdout_metrics.csv"
    final_predictions_path = output_folder / "final_holdout_predictions.csv"
    validation_diagnostics_path = output_folder / "validation_prediction_diagnostics.csv"
    final_diagnostics_path = output_folder / "final_holdout_prediction_diagnostics.csv"
    validation_monthly_path = output_folder / "validation_monthly_metrics.csv"
    validation_yearly_path = output_folder / "validation_yearly_metrics.csv"
    final_monthly_path = output_folder / "final_holdout_monthly_metrics.csv"
    final_yearly_path = output_folder / "final_holdout_yearly_metrics.csv"

    write_result_csv(validation_result.metrics, validation_metrics_path)
    write_result_csv(validation_summary, validation_summary_path)
    write_result_csv(final_metrics, final_metrics_path)
    write_result_csv(prediction_coverage_diagnostics(validation_result.metrics), validation_diagnostics_path)
    write_result_csv(prediction_coverage_diagnostics(final_metrics), final_diagnostics_path)
    write_result_csv(
        metric_breakdown_by_time(validation_result.predictions, ["model", "params", "year", "month"]),
        validation_monthly_path,
    )
    write_result_csv(
        metric_breakdown_by_time(validation_result.predictions, ["model", "params", "year"]),
        validation_yearly_path,
    )
    write_result_csv(
        metric_breakdown_by_time(final_predictions, ["model", "params", "year", "month"]),
        final_monthly_path,
    )
    write_result_csv(
        metric_breakdown_by_time(final_predictions, ["model", "params", "year"]),
        final_yearly_path,
    )
    final_predictions.to_csv(final_predictions_path, index=False)
    model_path, best_params_path, metadata_path = save_final_model_artifacts(
        table,
        feature_path,
        output_folder,
        model_module,
        model_options,
        validation_result.best_params,
        modelling_dataset.feature_columns,
        args.target_option,
        modelling_dataset.feature_mode,
        args.period_days,
        args.block,
    )

    if args.save_validation_predictions:
        validation_result.predictions.to_csv(validation_predictions_path, index=False)

    print(f"Model: {model_module.MODEL_NAME}")
    print(f"Target option: {args.target_option}")
    print(f"Feature mode: {modelling_dataset.feature_mode}")
    print(f"Selected features: {len(modelling_dataset.feature_columns)}")
    print(f"Best params from rolling validation: {validation_result.best_params}")
    print("\nValidation summary:")
    print(format_result_table(validation_summary).to_string(index=False))
    print("\nFinal holdout metrics:")
    print(format_result_table(compact_metric_table(final_metrics)).to_string(index=False))
    print("\nSaved outputs:")
    print(f"  validation metrics: {validation_metrics_path}")
    print(f"  validation summary: {validation_summary_path}")
    print(f"  validation diagnostics: {validation_diagnostics_path}")
    print(f"  validation monthly metrics: {validation_monthly_path}")
    print(f"  validation yearly metrics: {validation_yearly_path}")
    print(f"  final holdout metrics: {final_metrics_path}")
    print(f"  final holdout predictions: {final_predictions_path}")
    print(f"  final holdout diagnostics: {final_diagnostics_path}")
    print(f"  final holdout monthly metrics: {final_monthly_path}")
    print(f"  final holdout yearly metrics: {final_yearly_path}")
    print(f"  best params: {best_params_path}")
    print(f"  saved final model: {model_path}")
    print(f"  model metadata: {metadata_path}")
    if args.save_validation_predictions:
        print(f"  validation predictions: {validation_predictions_path}")
    else:
        print("  validation predictions: not saved; use --save-validation-predictions to write them")


if __name__ == "__main__":
    main()
