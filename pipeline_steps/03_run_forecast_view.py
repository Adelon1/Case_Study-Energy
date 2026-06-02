"""Run the full forecast-view workflow for one chosen delivery window.

This is deliberately similar to one fold of rolling validation:

``train = [test_begin - training_months, test_begin)``
``predict = [test_begin, test_end)``

Unlike validation, this runs exactly one user-chosen split. It then translates
the prediction into prompt-curve fair values, signals, plots, and optional AI
commentary.

Example:
    .venv/bin/python pipeline_steps/03_run_forecast_view.py
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

window_prediction = importlib.import_module("pipeline_helpers.02_modelling.10_window_prediction")
model_io = importlib.import_module("pipeline_helpers.02_modelling.04_model_io")
forecast_blocks = importlib.import_module("pipeline_helpers.03_curve_translation.01_forecast_blocks")
curve_view = importlib.import_module("pipeline_helpers.03_curve_translation.02_curve_view")
ai_commentary = importlib.import_module("pipeline_helpers.03_curve_translation.03_ai_commentary")

build_model_options = window_prediction.build_model_options
model_folder_for = window_prediction.model_folder_for
train_and_predict_window = window_prediction.train_and_predict_window
add_months = window_prediction.add_months
DeliveryPeriod = forecast_blocks.DeliveryPeriod
TimeWindow = window_prediction.TimeWindow
build_curve_view = curve_view.build_curve_view
write_curve_view_outputs = curve_view.write_curve_view_outputs
dataset_name_from_feature_path = model_io.dataset_name_from_feature_path
write_json = model_io.write_json
write_ai_commentary = ai_commentary.generate_commentary


DEFAULT_FEATURES = "data/03_processed/germany_modelling_2021_2026/germany_model_features.csv"
MARKET_TIMEZONE = "Europe/Berlin"


def parse_command_line_arguments() -> SimpleNamespace:
    """Return default settings; the user edits them through prompts."""

    return SimpleNamespace(
        features=DEFAULT_FEATURES,
        model="lear_model",
        forecast_setup="hourly_day_ahead",
        delivery_date=None,
        test_begin=None,
        test_end=None,
        training_months=24,
        period_days=None,
        block="all",
        benchmark="trailing_average",
        curve_price=None,
        regularization=None,
        target_transform=None,
        output_base_folder=None,
        ai_commentary=False,
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
    if args.forecast_setup == "hourly_day_ahead":
        args.delivery_date = ask("Delivery day to predict DD-MM-YYYY", args.delivery_date)
        if not args.delivery_date:
            raise ValueError("Delivery day is required for hourly_day_ahead.")
        args.test_begin = args.delivery_date
        args.test_end = next_delivery_date(args.delivery_date)
    else:
        print("Prediction window:")
        args.test_begin = ask("  test_begin DD-MM-YYYY", args.test_begin)
        args.test_end = ask("  test_end DD-MM-YYYY", args.test_end)
    args.training_months = int(ask("Training months before test_begin", str(args.training_months)))
    if args.forecast_setup == "period_average":
        args.period_days = int(ask("Period row length in days", str(args.period_days or 15)))
    else:
        args.period_days = None
    args.block = "all"
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
    args.benchmark = ask_choice(
        "Benchmark",
        ["trailing_average", "same_month_history", "manual"],
        args.benchmark,
    )
    if args.benchmark == "manual":
        args.curve_price = float(ask("Manual curve/benchmark price EUR/MWh", None))
    args.ai_commentary = ask_choice("Generate AI commentary", ["no", "yes"], "no") == "yes"
    return args


def parse_delivery_timestamp(date_text: str) -> pd.Timestamp:
    """Parse a DD-MM-YYYY German delivery boundary as a UTC timestamp.

    User-facing dates are delivery dates in German market time, not UTC dates.
    In summer, for example, 03-06-2026 starts at 2026-06-02 22:00 UTC.
    """

    local_midnight = pd.Timestamp(
        pd.to_datetime(date_text, format="%d-%m-%Y"),
        tz=MARKET_TIMEZONE,
    )
    return local_midnight.tz_convert("UTC")


def next_delivery_date(date_text: str) -> str:
    """Return the next DD-MM-YYYY date after a delivery day."""

    return (pd.to_datetime(date_text, format="%d-%m-%Y") + pd.Timedelta(days=1)).strftime("%d-%m-%Y")


def normalize_blocks(blocks: str | list[str]) -> list[str]:
    """Expand ``all`` into the standard three prompt-curve blocks."""

    selected = [blocks] if isinstance(blocks, str) else list(blocks)
    if "all" in selected:
        return ["baseload", "peakload", "offpeak"]
    return selected


def selected_curve_blocks(forecast_setup: str, blocks: str | list[str]) -> list[str]:
    """Return the curve blocks to analyze for this forecast setup."""

    return normalize_blocks(blocks)


def plot_context_label(model_label: str, prediction_window: TimeWindow) -> str:
    """Return a compact label with model and delivery period for plot titles."""

    local_start = prediction_window.test_begin.tz_convert(MARKET_TIMEZONE)
    local_end_inclusive = prediction_window.test_end.tz_convert(MARKET_TIMEZONE) - pd.Timedelta(seconds=1)
    if local_start.date() == local_end_inclusive.date():
        period_text = local_start.strftime("%Y-%m-%d")
    else:
        period_text = f"{local_start:%Y-%m-%d} to {local_end_inclusive:%Y-%m-%d}"
    return f"Model: {model_label} | Delivery: {period_text} ({MARKET_TIMEZONE})"


def period_folder_slug(start_text: str, end_text: str) -> str:
    """Use the user's date notation for human-facing output folders."""

    return f"{start_text}_{end_text}"


def output_run_folder(
    feature_path: str | Path,
    model_folder: Path,
    start_text: str,
    end_text: str,
) -> Path:
    """Build the human-facing Task 3 output folder."""

    return (
        Path("outputs")
        / dataset_name_from_feature_path(feature_path)
        / period_folder_slug(start_text, end_text)
        / model_folder.name
    )


def load_predictions_and_feature_table(predictions_path: Path, features_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read prediction and feature tables with timestamps parsed later by helpers."""

    return (
        pd.read_csv(predictions_path, sep=None, engine="python"),
        pd.read_csv(features_path),
    )


def plot_forecast_actual_band(
    predictions: pd.DataFrame,
    output_path: Path,
    context_label: str,
) -> None:
    """Plot predicted vs actual prices over time, including bands when available."""

    rows = predictions.copy()
    rows["timestamp_utc"] = pd.to_datetime(rows["timestamp_utc"], utc=True)
    rows = rows.sort_values("timestamp_utc")
    local_time = rows["timestamp_utc"].dt.tz_convert(MARKET_TIMEZONE)
    plot_time = local_time.dt.tz_localize(None)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(plot_time, rows["y_pred"], label="Predicted", linewidth=1.6)
    if "y_true" in rows.columns:
        ax.plot(plot_time, rows["y_true"], label="Actual", linewidth=1.1, alpha=0.75)
    if {"y_pred_lower", "y_pred_upper"}.issubset(rows.columns):
        ax.fill_between(
            plot_time,
            rows["y_pred_lower"],
            rows["y_pred_upper"],
            alpha=0.18,
            label="P10-P90 band",
        )
    period_start = local_time.min()
    period_end = local_time.max()
    period_label = period_start.strftime("%Y-%m-%d")
    if period_end.date() != period_start.date():
        period_label = f"{period_label} to {period_end.strftime('%Y-%m-%d')}"
    ax.set_title(f"Forecast vs Actual Price ({period_label})\n{context_label}")
    ax.set_xlabel(f"Prediction time ({MARKET_TIMEZONE})")
    ax.set_ylabel("EUR/MWh")
    ax.grid(True, alpha=0.25)
    ax.legend()
    time_span = local_time.max() - local_time.min()
    if time_span <= pd.Timedelta(days=1):
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    else:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_forecast_heatmap(
    predictions: pd.DataFrame,
    output_path: Path,
    context_label: str,
) -> None:
    """Plot predicted price shape by date and local hour."""

    rows = predictions.dropna(subset=["y_pred"]).copy()
    rows["timestamp_utc"] = pd.to_datetime(rows["timestamp_utc"], utc=True)
    local = rows["timestamp_utc"].dt.tz_convert("Europe/Berlin")
    rows["date"] = local.dt.strftime("%d-%m")
    rows["hour"] = local.dt.hour
    heatmap = rows.pivot_table(index="hour", columns="date", values="y_pred", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(max(9, 0.35 * len(heatmap.columns)), 6))
    image = ax.imshow(heatmap, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(f"Predicted Price Heatmap\n{context_label}")
    ax.set_xlabel("Delivery date Europe/Berlin")
    ax.set_ylabel("Local hour")
    ax.set_yticks(range(0, 24, 3))
    tick_step = max(1, len(heatmap.columns) // 12)
    tick_positions = list(range(0, len(heatmap.columns), tick_step))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([heatmap.columns[index] for index in tick_positions], rotation=45, ha="right", fontsize=8)
    fig.colorbar(image, ax=ax, label="EUR/MWh")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_fair_value_vs_benchmark(
    summary: pd.DataFrame,
    output_path: Path,
    context_label: str,
) -> None:
    """Plot forecast fair value against benchmark for every block."""

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(summary))
    width = 0.38
    ax.bar([value - width / 2 for value in x], summary["forecast_fair_value"], width, label="Forecast FV")
    ax.bar([value + width / 2 for value in x], summary["benchmark_value"], width, label="Benchmark")
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary["block"])
    ax.set_title(f"Fair Value vs Benchmark\n{context_label}")
    ax.set_ylabel("EUR/MWh")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_signal_by_block(
    summary: pd.DataFrame,
    output_path: Path,
    context_label: str,
) -> None:
    """Plot forecast edge and confidence context by block."""

    colors = ["#2b8a3e" if edge > 0 else "#c92a2a" if edge < 0 else "#495057" for edge in summary["edge"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(summary["block"], summary["edge"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    for index, row in summary.iterrows():
        ax.text(index, row["edge"], row["signal"], ha="center", va="bottom" if row["edge"] >= 0 else "top")
    ax.set_title(f"Signal Edge by Block\n{context_label}")
    ax.set_ylabel("Forecast FV - benchmark (EUR/MWh)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_combined_curve_report(summary: pd.DataFrame, output_path: Path) -> None:
    """Write one compact report covering all requested blocks."""

    lines = [
        "# Forecast View Report",
        "",
        "## Curve Signals",
        "",
        "| Block | FV | Benchmark | Edge | Signal | Coverage |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            "| {block} | {forecast_fair_value:.2f} | {benchmark_value:.2f} | "
            "{edge:+.2f} | {signal} | {prediction_coverage:.1%} |".format(**row)
        )
    lines.extend(["", "## Desk Interpretation", ""])
    for row in summary.to_dict("records"):
        lines.append(f"- **{row['block']}**: {row['decision_rationale']} {row['desk_action']}")
    lines.extend(["", "## Invalidation Logic", "", str(summary.iloc[0]["invalidation_logic"])])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_metadata(
    output_folder: Path,
    args: SimpleNamespace,
    model_folder: Path,
    prediction_window: TimeWindow,
    prediction_artifacts: dict[str, Path],
    curve_summary_path: Path,
    plot_paths: list[Path],
) -> Path:
    """Write a human-facing metadata file for the complete Task 3 run."""

    metadata_path = output_folder / "metadata.json"
    write_json(
        {
            "dataset": dataset_name_from_feature_path(args.features),
            "model": args.model,
            "model_folder": str(model_folder),
            "forecast_setup": args.forecast_setup,
            "window_mode": "prediction_window_with_training_months",
            "delivery_date": args.delivery_date,
            "training_months": args.training_months,
            "train_begin": prediction_window.train_begin,
            "train_end": prediction_window.train_end,
            "prediction_begin": prediction_window.test_begin,
            "prediction_end": prediction_window.test_end,
            "period_days": args.period_days,
            "blocks": selected_curve_blocks(args.forecast_setup, args.block),
            "benchmark": args.benchmark,
            "curve_price": args.curve_price,
            "artifacts": {key: str(value) for key, value in prediction_artifacts.items()},
            "curve_summary": str(curve_summary_path),
            "plots": [str(path) for path in plot_paths],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        metadata_path,
    )
    return metadata_path


def add_proxy_bands_when_missing(predictions: pd.DataFrame, metrics_path: Path) -> pd.DataFrame:
    """Add a simple MAE band when validation residual bands are unavailable."""

    if {"y_pred_lower", "y_pred_upper"}.issubset(predictions.columns):
        return predictions

    metrics = pd.read_csv(metrics_path, sep=None, engine="python")
    mae = float(pd.to_numeric(metrics["mae"], errors="coerce").dropna().iloc[0])
    result = predictions.copy()
    result["y_pred_lower"] = result["y_pred"] - mae
    result["y_pred_upper"] = result["y_pred"] + mae
    return result


def main() -> None:
    """Run prediction, curve translation, plots, metadata, and optional AI commentary."""

    args = apply_interactive_settings(parse_command_line_arguments())
    missing_dates = [
        name
        for name in ["test_begin", "test_end"]
        if not getattr(args, name)
    ]
    if missing_dates:
        raise ValueError(f"Missing required window dates: {', '.join(missing_dates)}")
    if args.training_months <= 0:
        raise ValueError("Training months must be positive.")

    test_begin = parse_delivery_timestamp(args.test_begin)
    test_end = parse_delivery_timestamp(args.test_end)

    prediction_window = TimeWindow(
        train_begin=add_months(test_begin, -args.training_months),
        train_end=test_begin,
        test_begin=test_begin,
        test_end=test_end,
    )
    if prediction_window.test_begin >= prediction_window.test_end:
        raise ValueError("test_begin must be before test_end.")
    if args.period_days is None:
        args.period_days = max(
            1,
            int((prediction_window.test_end - prediction_window.test_begin) / pd.Timedelta(days=1)),
        )

    model_options = build_model_options(args.regularization, args.target_transform)
    model_folder = model_folder_for(
        feature_path=args.features,
        model_name=args.model,
        model_options=model_options,
        output_base_folder=args.output_base_folder,
        forecast_setup=args.forecast_setup,
        period_days=args.period_days,
        block=args.block,
    )
    output_folder = output_run_folder(
        args.features,
        model_folder,
        args.test_begin,
        args.test_end,
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
        artifact_subfolder=".",
        params_folder=model_folder,
    )

    predictions, feature_table = load_predictions_and_feature_table(result.predictions_path, args.features)
    predictions = add_proxy_bands_when_missing(predictions, result.metrics_path)
    predictions.to_csv(result.predictions_path, index=False, sep=";")

    selected_blocks = selected_curve_blocks(args.forecast_setup, args.block)
    curve_views = []
    block_summary_paths = []
    curve_output_folder = output_folder / "curve_translation"
    for block in selected_blocks:
        view = build_curve_view(
            predictions=predictions,
            feature_table=feature_table,
            metrics_path=result.metrics_path,
            period=DeliveryPeriod(
                start_utc=prediction_window.test_begin,
                end_utc=prediction_window.test_end,
            ),
            block=block,
            benchmark_method=args.benchmark,
            manual_curve_price=args.curve_price,
        )
        curve_views.append(view)
        block_summary_path, _block_report_path = write_curve_view_outputs(
            view,
            curve_output_folder / block,
        )
        block_summary_paths.append(block_summary_path)

    curve_summary = pd.DataFrame([asdict(view) for view in curve_views]).round(2)
    curve_summary_path = output_folder / "curve_view_summary.csv"
    curve_report_path = output_folder / "curve_view_report.md"
    curve_summary.to_csv(curve_summary_path, index=False)
    write_combined_curve_report(pd.DataFrame([asdict(view) for view in curve_views]), curve_report_path)

    plots_folder = output_folder / "plots"
    plots_folder.mkdir(parents=True, exist_ok=True)
    plot_paths = [
        plots_folder / "forecast_actual_band.png",
        plots_folder / "fair_value_vs_benchmark.png",
        plots_folder / "signal_by_block.png",
    ]
    plot_label = plot_context_label(model_folder.name, prediction_window)
    curve_view_table = pd.DataFrame([asdict(view) for view in curve_views])
    plot_forecast_actual_band(predictions, plot_paths[0], plot_label)
    plot_fair_value_vs_benchmark(curve_view_table, plot_paths[1], plot_label)
    plot_signal_by_block(curve_view_table, plot_paths[2], plot_label)
    if args.forecast_setup != "period_average":
        heatmap_path = plots_folder / "forecast_heatmap.png"
        plot_forecast_heatmap(predictions, heatmap_path, plot_label)
        plot_paths.append(heatmap_path)

    metadata_path = write_run_metadata(
        output_folder=output_folder,
        args=args,
        model_folder=model_folder,
        prediction_window=prediction_window,
        prediction_artifacts={
            "predictions": result.predictions_path,
            "metrics": result.metrics_path,
            "model": result.model_path,
            "prediction_metadata": result.metadata_path,
        },
        curve_summary_path=curve_summary_path,
        plot_paths=plot_paths,
    )

    if args.ai_commentary:
        for summary_path in block_summary_paths:
            write_ai_commentary(summary_path)

    print(f"Forecast setup: {args.forecast_setup}")
    print(f"Train window: {prediction_window.train_begin} -> {prediction_window.train_end}")
    print(f"Prediction window: {prediction_window.test_begin} -> {prediction_window.test_end}")
    print(f"Rows predicted: {predictions['y_pred'].notna().sum()} / {len(predictions)}")
    print("\nCurve signals:")
    for view in curve_views:
        print(
            f"  {view.block}: {view.signal} | FV {view.forecast_fair_value:.2f} "
            f"vs benchmark {view.benchmark_value:.2f} | edge {view.edge:+.2f}"
        )
    print("\nSaved outputs:")
    print(f"  run folder: {output_folder}")
    print(f"  predictions: {result.predictions_path}")
    print(f"  metrics: {result.metrics_path}")
    print(f"  model: {result.model_path}")
    print(f"  metadata: {metadata_path}")
    print(f"  curve summary: {curve_summary_path}")
    print(f"  curve report: {curve_report_path}")
    print(f"  plots: {plots_folder}")


if __name__ == "__main__":
    main()
