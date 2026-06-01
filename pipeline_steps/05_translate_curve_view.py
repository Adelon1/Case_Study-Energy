"""Pipeline step: translate hourly forecasts into a prompt-curve trading view.

Example:
    .venv/bin/python pipeline_steps/05_translate_curve_view.py \
      --features data/03_processed/germany_modelling_2021_2026/germany_model_features.csv \
      --start 01-10-2025 \
      --end 01-01-2026 \
      --model lear_model \
      --regularization lasso \
      --target-transform raw \
      --block baseload \
      --benchmark trailing_average

Interactive mode:
    .venv/bin/python pipeline_steps/05_translate_curve_view.py --interactive
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

constants = importlib.import_module("pipeline_helpers.03_curve_translation.00_constants")
curve_view = importlib.import_module("pipeline_helpers.03_curve_translation.02_curve_view")
forecast_blocks = importlib.import_module("pipeline_helpers.03_curve_translation.01_forecast_blocks")
period_prediction = importlib.import_module("pipeline_helpers.02_modelling.10_period_prediction")

build_curve_view = curve_view.build_curve_view
write_curve_view_outputs = curve_view.write_curve_view_outputs
parse_utc_period = forecast_blocks.parse_utc_period
build_model_options = period_prediction.build_model_options
get_or_create_predictions = period_prediction.get_or_create_predictions


DEFAULT_FEATURES = "data/03_processed/germany_modelling_2021_2026/germany_model_features.csv"


def parse_command_line_arguments() -> argparse.Namespace:
    """Read translation settings from the command line."""

    parser = argparse.ArgumentParser(description="Translate model forecasts into curve views.")
    parser.add_argument("--interactive", action="store_true", help="Ask for settings interactively.")
    parser.add_argument("--features", default=DEFAULT_FEATURES, help="Path to germany_model_features.csv.")
    parser.add_argument("--model", default=constants.DEFAULT_MODEL, help="Model module name.")
    parser.add_argument(
        "--target",
        dest="target_option",
        choices=["hourly", "period_average"],
        default="hourly",
        help="'hourly' aggregates hourly predictions; 'period_average' predicts the period average directly.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["day_ahead_full", "period_hourly_safe", "fundamentals_calendar_only"],
        default="period_hourly_safe",
        help="Feature mode used when the hourly target needs retraining.",
    )
    parser.add_argument(
        "--period-days",
        type=int,
        default=None,
        help="Period length in days for feature rules. Defaults to end-start.",
    )
    parser.add_argument(
        "--regularization",
        choices=["lasso", "elasticnet", "ridge"],
        default=None,
        help="Regularization option for models that support it.",
    )
    parser.add_argument(
        "--target-transform",
        choices=["raw", "asinh"],
        default=None,
        help="Target transform option for models that support it.",
    )
    parser.add_argument("--start", default=None, help="Delivery start date, DD-MM-YYYY.")
    parser.add_argument("--end", default=None, help="Exclusive delivery end date, DD-MM-YYYY.")
    parser.add_argument(
        "--block",
        nargs="+",
        choices=["baseload", "peakload", "offpeak", "peak_base_spread", "all"],
        default=["baseload"],
        help="Delivery block(s) to translate into signals. Use 'all' for every block.",
    )
    parser.add_argument(
        "--benchmark",
        choices=["trailing_average", "same_month_history", "manual"],
        default="trailing_average",
        help="Benchmark used as proxy curve reference.",
    )
    parser.add_argument(
        "--curve-price",
        type=float,
        default=None,
        help="Manual benchmark or observable curve price in EUR/MWh.",
    )
    parser.add_argument(
        "--output-base-folder",
        default=None,
        help="Base folder containing model outputs. Defaults to models/<dataset-name>.",
    )
    parser.add_argument(
        "--output-folder",
        default=None,
        help="Folder for curve translation outputs. Defaults under the selected model folder.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain for the requested period even if existing predictions cover it.",
    )
    parser.add_argument(
        "--ai-commentary",
        action="store_true",
        help="After writing each curve view, call the AI commentary pipeline step.",
    )
    return parser.parse_args()


def ask(prompt: str, default: str | None = None) -> str:
    """Prompt for one interactive value."""

    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or (default or "")


def ask_choice(prompt: str, choices: list[str], default: str) -> str:
    """Prompt until the user selects a valid choice."""

    choices_text = ", ".join(choices)
    while True:
        value = ask(f"{prompt} ({choices_text})", default)
        if value in choices:
            return value
        print(f"Please choose one of: {choices_text}")


def apply_interactive_settings(args: argparse.Namespace) -> argparse.Namespace:
    """Fill command-line args through an interactive terminal flow."""

    args.features = ask("Feature CSV", args.features)
    args.model = ask("Model", args.model)
    args.target_option = ask_choice("Target", ["hourly", "period_average"], args.target_option)
    if args.target_option == "hourly":
        args.feature_mode = ask_choice(
            "Feature mode",
            ["day_ahead_full", "period_hourly_safe", "fundamentals_calendar_only"],
            args.feature_mode,
        )
    if args.model == "lear_model":
        args.regularization = ask_choice(
            "Regularization",
            ["lasso", "elasticnet", "ridge"],
            args.regularization or constants.DEFAULT_REGULARIZATION,
        )
    args.target_transform = ask_choice(
        "Target transform",
        ["raw", "asinh"],
        args.target_transform or constants.DEFAULT_TARGET_TRANSFORM,
    )
    args.start = ask("Delivery start date DD-MM-YYYY", args.start)
    args.end = ask("Exclusive delivery end date DD-MM-YYYY", args.end)
    period_days_default = args.period_days
    if period_days_default is None and args.start and args.end:
        period_days_default = max(
            1,
            int(
                (
                    pd.to_datetime(args.end, format="%d-%m-%Y")
                    - pd.to_datetime(args.start, format="%d-%m-%Y")
                )
                / pd.Timedelta(days=1)
            ),
        )
    args.period_days = int(ask("Period length in days", str(period_days_default or 1)))
    args.block = ask_choice(
        "Block",
        ["baseload", "peakload", "offpeak", "peak_base_spread", "all"],
        first_block(args.block),
    )
    args.benchmark = ask_choice(
        "Benchmark",
        ["trailing_average", "same_month_history", "manual"],
        args.benchmark,
    )
    if args.benchmark == "manual":
        args.curve_price = float(ask("Manual curve/benchmark price EUR/MWh", None))
    force_retrain = ask_choice("Force retrain", ["no", "yes"], "no")
    args.force_retrain = force_retrain == "yes"
    return args


def parse_delivery_date(date_text: str) -> str:
    """Convert DD-MM-YYYY input into YYYY-MM-DD for pandas."""

    return pd.to_datetime(date_text, format="%d-%m-%Y").strftime("%Y-%m-%d")


def first_block(blocks: str | list[str]) -> str:
    """Return the first selected block for interactive defaults."""

    if isinstance(blocks, list):
        return blocks[0]
    return blocks


def normalize_blocks(blocks: str | list[str]) -> list[str]:
    """Expand command-line block input into concrete block names."""

    if isinstance(blocks, str):
        selected_blocks = [blocks]
    else:
        selected_blocks = blocks

    all_blocks = ["baseload", "peakload", "offpeak", "peak_base_spread"]
    if "all" in selected_blocks:
        return all_blocks
    return selected_blocks


def generate_ai_commentary(summary_path: Path) -> None:
    """Call the AI commentary pipeline step for one curve-view summary."""

    subprocess.run(
        [
            sys.executable,
            "pipeline_steps/06_generate_ai_commentary.py",
            "--summary",
            str(summary_path),
        ],
        check=True,
    )


def main() -> None:
    """Create a curve-translation report."""

    args = parse_command_line_arguments()
    if args.interactive:
        args = apply_interactive_settings(args)

    if not args.start or not args.end:
        raise ValueError("Delivery --start and --end are required unless provided interactively.")

    period = parse_utc_period(parse_delivery_date(args.start), parse_delivery_date(args.end))
    if args.period_days is None:
        args.period_days = max(1, int((period.end_utc - period.start_utc) / pd.Timedelta(days=1)))
    selected_blocks = normalize_blocks(args.block)
    model_options = build_model_options(args.regularization, args.target_transform)
    prediction_source = get_or_create_predictions(
        feature_path=args.features,
        model_name=args.model,
        model_options=model_options,
        period=period,
        output_base_folder=args.output_base_folder,
        force_retrain=args.force_retrain,
        target_option=args.target_option,
        feature_mode=args.feature_mode,
        period_days=args.period_days,
        block=selected_blocks[0],
    )

    predictions = pd.read_csv(prediction_source.predictions_path, sep=None, engine="python")
    feature_table = pd.read_csv(args.features)

    has_band = {"y_pred_lower", "y_pred_upper"}.issubset(predictions.columns)
    print("=== Forecast source ===")
    print(f"  Model: {args.model}")
    print(f"  Prediction source: {prediction_source.source} (retrained={prediction_source.retrained})")
    print(f"  Predictions: {prediction_source.predictions_path}")
    print(f"  Metrics: {prediction_source.metrics_path}")
    print(f"  Forecast band available: {'yes (P10-P90)' if has_band else 'no (risk-buffer proxy)'}")

    output_base_folder = (
        Path(args.output_folder)
        if args.output_folder
        else Path("outputs")
        / "curve_translation"
        / Path(args.features).resolve().parent.name
        / prediction_source.model_folder.name
        / f"{period.start_utc:%Y%m%d}_{period.end_utc:%Y%m%d}"
    )

    print("\n=== Curve views ===")
    for block in selected_blocks:
        view = build_curve_view(
            predictions=predictions,
            feature_table=feature_table,
            metrics_path=prediction_source.metrics_path,
            period=period,
            block=block,
            benchmark_method=args.benchmark,
            manual_curve_price=args.curve_price,
        )
        block_output_folder = output_base_folder / block
        summary_path, report_path = write_curve_view_outputs(view, block_output_folder)

        print(f"\n[{block}] signal: {view.signal}")
        print(
            f"  forecast {view.forecast_fair_value:.2f} "
            f"(band {view.forecast_low:.2f}-{view.forecast_high:.2f}, {view.band_source})"
        )
        print(f"  benchmark {view.benchmark_value:.2f} via {view.benchmark_method}")
        print(f"  edge {view.edge:+.2f} EUR/MWh, margin beyond band {view.signal_margin:.2f}")
        print(f"  {view.decision_rationale}")
        print(f"  desk action: {view.desk_action}")
        print(f"  coverage {view.prediction_coverage:.1%} | summary {summary_path}")

        if args.ai_commentary:
            generate_ai_commentary(summary_path)


if __name__ == "__main__":
    main()
