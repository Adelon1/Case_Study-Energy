"""Train once on a chosen history window and predict the remaining window.

This is deliberately similar to one fold of rolling validation:

``full window = [window_start, window_end)``
``train = [window_start, window_start + train_months)``
``predict = [window_start + train_months, window_end)``

Unlike validation, this runs exactly one user-chosen split. Use it after
validation has found good hyperparameters and you want a concrete prediction
artifact for a specific date range.

Example:
    .venv/bin/python pipeline_steps/04_predict_window.py
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

window_prediction = importlib.import_module("pipeline_helpers.02_modelling.10_window_prediction")
forecast_blocks = importlib.import_module("pipeline_helpers.03_curve_translation.01_forecast_blocks")

build_model_options = window_prediction.build_model_options
model_folder_for = window_prediction.model_folder_for
train_and_predict_window = window_prediction.train_and_predict_window
parse_utc_period = forecast_blocks.parse_utc_period
TimeWindow = window_prediction.TimeWindow
add_months = window_prediction.add_months


DEFAULT_FEATURES = "data/03_processed/germany_modelling_2021_2026/germany_model_features.csv"


def parse_command_line_arguments() -> SimpleNamespace:
    """Return default settings; the user edits them through prompts."""

    return SimpleNamespace(
        features=DEFAULT_FEATURES,
        model="lear_model",
        forecast_setup="hourly_day_ahead",
        window_start=None,
        window_end=None,
        train_months=24,
        period_days=None,
        block="baseload",
        regularization=None,
        target_transform=None,
        output_base_folder=None,
    )


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


def apply_interactive_settings(args: SimpleNamespace) -> SimpleNamespace:
    """Fill missing values interactively."""

    args.features = ask("Feature CSV", args.features)
    args.model = ask("Model", args.model)
    args.forecast_setup = ask_choice(
        "Forecast setup",
        ["hourly_day_ahead", "hourly_period", "period_average"],
        args.forecast_setup,
    )
    args.window_start = ask("Full window start DD-MM-YYYY", args.window_start)
    args.window_end = ask("Full window end DD-MM-YYYY", args.window_end)
    args.train_months = int(ask("Training months", str(args.train_months)))
    if args.forecast_setup == "period_average":
        args.period_days = int(ask("Period row length in days", str(args.period_days or 15)))
        args.block = ask_choice("Block", ["baseload", "peakload", "offpeak"], args.block)
    else:
        args.period_days = None
        args.block = "baseload"
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


def parse_delivery_date(date_text: str) -> str:
    """Parse a DD-MM-YYYY date into the YYYY-MM-DD format used by UTC periods."""

    return pd.to_datetime(date_text, format="%d-%m-%Y").strftime("%Y-%m-%d")


def main() -> None:
    """Train on the first part of a user window and predict the rest."""

    args = apply_interactive_settings(parse_command_line_arguments())
    if not args.window_start or not args.window_end:
        raise ValueError("Full window start and end dates are required.")

    full_period = parse_utc_period(
        parse_delivery_date(args.window_start),
        parse_delivery_date(args.window_end),
    )
    train_end = add_months(full_period.start_utc, args.train_months)
    if train_end >= full_period.end_utc:
        raise ValueError(
            "Training months leave no prediction window. "
            f"Train would end at {train_end}, but full window ends at {full_period.end_utc}."
        )

    prediction_window = TimeWindow(
        train_begin=full_period.start_utc,
        train_end=train_end,
        test_begin=train_end,
        test_end=full_period.end_utc,
    )
    if args.period_days is None:
        args.period_days = max(
            1,
            int((prediction_window.test_end - prediction_window.test_begin) / pd.Timedelta(days=1)),
        )

    model_options = build_model_options(args.regularization, args.target_transform)
    output_folder = model_folder_for(
        feature_path=args.features,
        model_name=args.model,
        model_options=model_options,
        output_base_folder=args.output_base_folder,
        forecast_setup=args.forecast_setup,
        period_days=args.period_days,
        block=args.block,
    )
    result = train_and_predict_window(
        feature_path=args.features,
        model_name=args.model,
        model_options=model_options,
        output_folder=output_folder,
        window=prediction_window,
        forecast_setup=args.forecast_setup,
        period_days=args.period_days,
        block=args.block,
    )

    predictions = pd.read_csv(result.predictions_path, sep=None, engine="python")
    print(f"Forecast setup: {args.forecast_setup}")
    print(f"Train window: {prediction_window.train_begin} -> {prediction_window.train_end}")
    print(f"Prediction window: {prediction_window.test_begin} -> {prediction_window.test_end}")
    print(f"Rows predicted: {predictions['y_pred'].notna().sum()} / {len(predictions)}")
    print("\nSaved outputs:")
    print(f"  predictions: {result.predictions_path}")
    print(f"  metrics: {result.metrics_path}")
    print(f"  model: {result.model_path}")
    print(f"  metadata: {result.metadata_path}")


if __name__ == "__main__":
    main()
