"""Pipeline step: run rolling validation and final holdout testing.

Example:
    .venv/bin/python pipeline_steps/validate_model.py \
      --features data/processed/germany_modelling_2021_2026/germany_model_features.csv \
      --model baseline_week_lag

    .venv/bin/python pipeline_steps/validate_model.py \
      --features data/processed/germany_modelling_2021_2026/germany_model_features.csv \
      --model lear_model \
      --regularization elasticnet \
      --target-transform asinh

Outputs are written to a model-specific folder next to the feature CSV unless
``--output-folder`` is provided.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_helpers.modelling import constants  # noqa: E402
from pipeline_helpers.modelling.validation import (  # noqa: E402
    average_metrics_by_params,
    load_feature_data,
    run_final_holdout_test,
    run_rolling_validation,
)


def parse_command_line_arguments() -> argparse.Namespace:
    """Read model validation settings from the command line."""

    parser = argparse.ArgumentParser(description="Validate a forecasting model.")
    parser.add_argument("--features", required=True, help="Path to germany_model_features.csv.")
    parser.add_argument(
        "--model",
        default="baseline_week_lag",
        help="Model module name inside pipeline_helpers/modelling.",
    )
    parser.add_argument(
        "--output-folder",
        default=None,
        help="Base folder for validation outputs. Defaults to the feature CSV folder.",
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
        help="Target transform for models that support it. LEAR defaults to raw.",
    )
    return parser.parse_args()


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


def main() -> None:
    """Run validation, final holdout, and write result CSVs."""

    args = parse_command_line_arguments()
    feature_path = Path(args.features)
    output_base_folder = Path(args.output_folder) if args.output_folder else feature_path.parent
    model_module = load_model_module(args.model)
    model_options = build_model_options(args)
    output_folder = output_base_folder / output_folder_name(model_module, model_options)
    output_folder.mkdir(parents=True, exist_ok=True)

    table = load_feature_data(feature_path)
    validation_result = run_rolling_validation(
        table,
        model_module,
        model_options=model_options,
        show_progress=True,
    )
    final_metrics, final_predictions = run_final_holdout_test(
        table,
        model_module,
        validation_result.best_params,
    )
    validation_summary = summarize_validation(validation_result.metrics)

    validation_metrics_path = output_folder / "validation_metrics.csv"
    validation_summary_path = output_folder / "validation_summary.csv"
    validation_predictions_path = output_folder / "validation_predictions.csv"
    final_metrics_path = output_folder / "final_holdout_metrics.csv"
    final_predictions_path = output_folder / "final_holdout_predictions.csv"

    write_result_csv(validation_result.metrics, validation_metrics_path)
    write_result_csv(validation_summary, validation_summary_path)
    write_result_csv(final_metrics, final_metrics_path)
    final_predictions.to_csv(final_predictions_path, index=False)

    if args.save_validation_predictions:
        validation_result.predictions.to_csv(validation_predictions_path, index=False)

    print(f"Model: {model_module.MODEL_NAME}")
    print(f"Best params from rolling validation: {validation_result.best_params}")
    print("\nValidation summary:")
    print(format_result_table(validation_summary).to_string(index=False))
    print("\nFinal holdout metrics:")
    print(format_result_table(compact_metric_table(final_metrics)).to_string(index=False))
    print("\nSaved outputs:")
    print(f"  validation metrics: {validation_metrics_path}")
    print(f"  validation summary: {validation_summary_path}")
    print(f"  final holdout metrics: {final_metrics_path}")
    print(f"  final holdout predictions: {final_predictions_path}")
    if args.save_validation_predictions:
        print(f"  validation predictions: {validation_predictions_path}")
    else:
        print("  validation predictions: not saved; use --save-validation-predictions to write them")


if __name__ == "__main__":
    main()
