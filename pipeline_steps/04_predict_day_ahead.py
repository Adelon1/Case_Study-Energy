"""Predict one delivery day as a 24-hour day-ahead vector.

This is a thin interface over the shared Option A hourly model path. It makes
the day-ahead forecast object explicit: one command returns the full delivery
day, not one hour at a time.

Example:
    .venv/bin/python pipeline_steps/04_predict_day_ahead.py \
      --features data/03_processed/germany_modelling_2021_2026/germany_model_features.csv \
      --delivery-day 01-12-2025 \
      --model lear_model \
      --regularization lasso \
      --target-transform raw
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

period_prediction = importlib.import_module("pipeline_helpers.02_modelling.10_period_prediction")
forecast_blocks = importlib.import_module("pipeline_helpers.03_curve_translation.01_forecast_blocks")

build_model_options = period_prediction.build_model_options
get_or_create_predictions = period_prediction.get_or_create_predictions
parse_utc_period = forecast_blocks.parse_utc_period


DEFAULT_FEATURES = "data/03_processed/germany_modelling_2021_2026/germany_model_features.csv"


def parse_command_line_arguments() -> argparse.Namespace:
    """Read day-ahead prediction settings."""

    parser = argparse.ArgumentParser(description="Predict one day-ahead 24-hour vector.")
    parser.add_argument("--interactive", action="store_true", help="Ask for settings interactively.")
    parser.add_argument("--features", default=DEFAULT_FEATURES, help="Path to feature CSV.")
    parser.add_argument("--delivery-day", default=None, help="Delivery day, DD-MM-YYYY.")
    parser.add_argument("--model", default="lear_model", help="Model module name.")
    parser.add_argument(
        "--regularization",
        choices=["lasso", "elasticnet", "ridge"],
        default=None,
        help="Regularization for models that support it.",
    )
    parser.add_argument(
        "--target-transform",
        choices=["raw", "asinh"],
        default=None,
        help="Target transform for models that support it.",
    )
    parser.add_argument("--output-base-folder", default=None, help="Base model output folder.")
    parser.add_argument("--force-retrain", action="store_true", help="Retrain even if predictions exist.")
    return parser.parse_args()


def ask(prompt: str, default: str | None = None) -> str:
    """Ask one terminal question."""

    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or (default or "")


def ask_choice(prompt: str, choices: list[str], default: str) -> str:
    """Ask until the answer is valid."""

    choices_text = ", ".join(choices)
    while True:
        answer = ask(f"{prompt} ({choices_text})", default)
        if answer in choices:
            return answer
        print(f"Please choose one of: {choices_text}")


def apply_interactive_settings(args: argparse.Namespace) -> argparse.Namespace:
    """Fill missing values interactively."""

    args.features = ask("Feature CSV", args.features)
    args.delivery_day = ask("Delivery day DD-MM-YYYY", args.delivery_day)
    args.model = ask("Model", args.model)
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
    args.force_retrain = ask_choice("Force retrain", ["no", "yes"], "no") == "yes"
    return args


def parse_delivery_day(day_text: str) -> tuple[str, str]:
    """Return start and exclusive end dates in YYYY-MM-DD format."""

    start = pd.to_datetime(day_text, format="%d-%m-%Y")
    end = start + pd.Timedelta(days=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def main() -> None:
    """Create or reuse a full-day hourly forecast."""

    args = parse_command_line_arguments()
    if args.interactive:
        args = apply_interactive_settings(args)
    if not args.delivery_day:
        raise ValueError("--delivery-day is required unless provided interactively.")

    start, end = parse_delivery_day(args.delivery_day)
    period = parse_utc_period(start, end)
    model_options = build_model_options(args.regularization, args.target_transform)
    prediction_source = get_or_create_predictions(
        feature_path=args.features,
        model_name=args.model,
        model_options=model_options,
        period=period,
        output_base_folder=args.output_base_folder,
        force_retrain=args.force_retrain,
        target_option="hourly",
        feature_mode="day_ahead_full",
        period_days=1,
        block="baseload",
    )

    predictions = pd.read_csv(prediction_source.predictions_path, sep=None, engine="python")
    timestamps = pd.to_datetime(predictions["timestamp_utc"], utc=True)
    day_predictions = predictions.loc[
        (timestamps >= period.start_utc) & (timestamps < period.end_utc),
        ["timestamp_utc", "y_pred"],
    ]

    print(f"Prediction source: {prediction_source.source}")
    print(f"Retrained: {prediction_source.retrained}")
    print(f"Predictions: {prediction_source.predictions_path}")
    print("\n24-hour day-ahead vector:")
    print(day_predictions.to_string(index=False))


if __name__ == "__main__":
    main()
