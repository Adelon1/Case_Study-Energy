"""Pipeline step: translate hourly forecasts into a prompt-curve trading view.

Example:
    .venv/bin/python pipeline_steps/05_translate_curve_view.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

constants = importlib.import_module("pipeline_helpers.03_curve_translation.00_constants")
curve_view = importlib.import_module("pipeline_helpers.03_curve_translation.02_curve_view")
forecast_blocks = importlib.import_module("pipeline_helpers.03_curve_translation.01_forecast_blocks")
period_prediction = importlib.import_module("pipeline_helpers.02_modelling.10_period_prediction")
ai_commentary_step = importlib.import_module("pipeline_steps.06_generate_ai_commentary")

build_curve_view = curve_view.build_curve_view
write_curve_view_outputs = curve_view.write_curve_view_outputs
parse_utc_period = forecast_blocks.parse_utc_period
build_model_options = period_prediction.build_model_options
get_or_create_predictions = period_prediction.get_or_create_predictions
write_ai_commentary = ai_commentary_step.generate_commentary


DEFAULT_FEATURES = "data/03_processed/germany_modelling_2021_2026/germany_model_features.csv"


def parse_command_line_arguments() -> SimpleNamespace:
    """Return default settings; the user edits them through prompts."""

    return SimpleNamespace(
        features=DEFAULT_FEATURES,
        model=constants.DEFAULT_MODEL,
        forecast_setup="hourly_period",
        period_days=None,
        regularization=None,
        target_transform=None,
        start=None,
        end=None,
        block=["baseload"],
        benchmark="trailing_average",
        curve_price=None,
        output_base_folder=None,
        output_folder=None,
        force_retrain=False,
        ai_commentary=False,
    )


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


def apply_interactive_settings(args: SimpleNamespace) -> SimpleNamespace:
    """Fill command-line args through an interactive terminal flow."""

    args.features = ask("Feature CSV", args.features)
    args.model = ask("Model", args.model)
    args.forecast_setup = ask_choice(
        "Forecast setup",
        ["hourly_day_ahead", "hourly_period", "period_average"],
        args.forecast_setup,
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
    args.ai_commentary = ask_choice("Generate AI commentary", ["no", "yes"], "no") == "yes"
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

    write_ai_commentary(summary_path)


def main() -> None:
    """Create a curve-translation report."""

    args = apply_interactive_settings(parse_command_line_arguments())

    if not args.start or not args.end:
        raise ValueError("Delivery start and end dates are required.")

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
        forecast_setup=args.forecast_setup,
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
