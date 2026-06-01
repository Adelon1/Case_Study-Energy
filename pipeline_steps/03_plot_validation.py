"""Plot rolling-validation results for one model run.

Reads a model run's ``predictions.csv`` (the out-of-sample walk-forward
forecast) and writes three figures that together tell the accuracy story:

  1. forecast_vs_actual.png  - recent days of actual vs forecast with the
                               P10-P90 residual band drawn around the forecast.
  2. mae_by_hour.png         - mean absolute error per delivery hour, showing
                               which hours of the day are hardest to forecast.
  3. residual_distribution.png - spread of forecast errors with the band edges
                               and realised coverage.

Example:
    .venv/bin/python pipeline_steps/03_plot_validation.py \
      --run-folder models/germany_modelling_2021_2026/lear_model_lasso_raw__hourly__day_ahead_full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_helpers.modelling import constants  # noqa: E402

TIMESTAMP_COLUMN = constants.TIMESTAMP_COLUMN


def parse_command_line_arguments() -> argparse.Namespace:
    """Read plotting settings from the command line."""

    parser = argparse.ArgumentParser(description="Plot rolling-validation results.")
    parser.add_argument(
        "--run-folder",
        required=True,
        help="Model run folder containing predictions.csv.",
    )
    parser.add_argument(
        "--output-folder",
        default=None,
        help="Where to write figures. Defaults to outputs/figures/<run-name>.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="How many recent days to show in the forecast-vs-actual chart.",
    )
    return parser.parse_args()


def load_predictions(run_folder: Path) -> pd.DataFrame:
    """Load a run's out-of-sample predictions and parse the timestamp."""

    predictions_path = run_folder / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"No predictions.csv in {run_folder}")

    predictions = pd.read_csv(predictions_path, sep=";")
    predictions[TIMESTAMP_COLUMN] = pd.to_datetime(predictions[TIMESTAMP_COLUMN], utc=True)
    return predictions.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)


def plot_forecast_vs_actual(predictions: pd.DataFrame, days: int, path: Path) -> None:
    """Plot recent actual vs forecast prices with the forecast band."""

    last_timestamp = predictions[TIMESTAMP_COLUMN].max()
    window_start = last_timestamp - pd.Timedelta(days=days)
    recent = predictions.loc[predictions[TIMESTAMP_COLUMN] >= window_start]

    figure, axis = plt.subplots(figsize=(12, 5))
    if {"y_pred_lower", "y_pred_upper"}.issubset(recent.columns):
        axis.fill_between(
            recent[TIMESTAMP_COLUMN],
            recent["y_pred_lower"],
            recent["y_pred_upper"],
            color="tab:orange",
            alpha=0.2,
            label=(
                f"P{int(constants.BAND_LOWER_QUANTILE * 100)}-"
                f"P{int(constants.BAND_UPPER_QUANTILE * 100)} band"
            ),
        )
    axis.plot(recent[TIMESTAMP_COLUMN], recent["y_true"], color="black", linewidth=1.4, label="Actual")
    axis.plot(
        recent[TIMESTAMP_COLUMN],
        recent["y_pred"],
        color="tab:orange",
        linewidth=1.4,
        label="Forecast",
    )
    axis.set_title(f"Day-ahead price: actual vs forecast (last {days} days)")
    axis.set_ylabel("EUR / MWh")
    axis.set_xlabel("Delivery time (UTC)")
    axis.legend(loc="upper left")
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def plot_mae_by_hour(predictions: pd.DataFrame, path: Path) -> None:
    """Plot mean absolute error for each local delivery hour."""

    if "local_hour" not in predictions.columns:
        return

    errors = (predictions["y_pred"] - predictions["y_true"]).abs()
    mae_by_hour = errors.groupby(predictions["local_hour"]).mean()

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(mae_by_hour.index, mae_by_hour.values, color="tab:blue")
    axis.set_title("Mean absolute error by delivery hour")
    axis.set_ylabel("MAE (EUR / MWh)")
    axis.set_xlabel("Local delivery hour")
    axis.set_xticks(range(0, 24))
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def plot_residual_distribution(predictions: pd.DataFrame, path: Path) -> None:
    """Plot the forecast error distribution with band edges and coverage."""

    residuals = predictions["y_true"] - predictions["y_pred"]
    lower_edge = residuals.quantile(constants.BAND_LOWER_QUANTILE)
    upper_edge = residuals.quantile(constants.BAND_UPPER_QUANTILE)

    coverage_note = ""
    if {"y_pred_lower", "y_pred_upper"}.issubset(predictions.columns):
        inside = (predictions["y_true"] >= predictions["y_pred_lower"]) & (
            predictions["y_true"] <= predictions["y_pred_upper"]
        )
        coverage_note = f" - realised coverage {inside.mean():.0%}"

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(residuals, bins=60, color="tab:gray", alpha=0.8)
    axis.axvline(0.0, color="black", linewidth=1.0, label="Zero error")
    axis.axvline(lower_edge, color="tab:orange", linestyle="--", label="Band edges")
    axis.axvline(upper_edge, color="tab:orange", linestyle="--")
    axis.set_title(f"Forecast error distribution{coverage_note}")
    axis.set_ylabel("Hours")
    axis.set_xlabel("Actual - forecast (EUR / MWh)")
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def main() -> None:
    """Generate the validation figures for one model run."""

    args = parse_command_line_arguments()
    run_folder = Path(args.run_folder)
    output_folder = (
        Path(args.output_folder)
        if args.output_folder
        else PROJECT_ROOT / "outputs" / "figures" / run_folder.name
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(run_folder)

    forecast_path = output_folder / "forecast_vs_actual.png"
    hour_path = output_folder / "mae_by_hour.png"
    residual_path = output_folder / "residual_distribution.png"

    plot_forecast_vs_actual(predictions, args.days, forecast_path)
    plot_mae_by_hour(predictions, hour_path)
    plot_residual_distribution(predictions, residual_path)

    print(f"Run: {run_folder.name}")
    print("Saved figures:")
    print(f"  {forecast_path}")
    print(f"  {hour_path}")
    print(f"  {residual_path}")


if __name__ == "__main__":
    main()
