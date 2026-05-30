"""Build risk-adjusted prompt-curve views from forecast blocks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from pipeline_helpers.curve_translation import constants
from pipeline_helpers.curve_translation.forecast_blocks import (
    DeliveryPeriod,
    calculate_block_values,
    filter_delivery_period,
    peakload_mask,
)
from pipeline_helpers.modelling import constants as modelling_constants


@dataclass(frozen=True)
class BenchmarkResult:
    """Benchmark price used as the reference for the forecast fair value."""

    method: str
    value: float
    description: str


@dataclass(frozen=True)
class CurveView:
    """Final prompt-curve trading view."""

    period_start_utc: str
    period_end_utc: str
    block: str
    forecast_fair_value: float
    benchmark_method: str
    benchmark_value: float
    edge: float
    mae: float
    tail_metric_name: str
    tail_metric_value: float
    risk_buffer: float
    confidence_score: float
    signal: str
    desk_action: str
    invalidation_logic: str
    prediction_coverage: float


def benchmark_manual(value: float) -> BenchmarkResult:
    """Use a user-provided curve or benchmark price."""

    return BenchmarkResult(
        method="manual",
        value=float(value),
        description="User-provided benchmark or observable curve price.",
    )


def benchmark_trailing_average(
    feature_table: pd.DataFrame,
    period: DeliveryPeriod,
    block: str,
    days: int = constants.TRAILING_BENCHMARK_DAYS,
) -> BenchmarkResult:
    """Use the realized average over the days before the delivery period."""

    table = feature_table.copy()
    table["timestamp_utc"] = pd.to_datetime(table["timestamp_utc"], utc=True)
    begin = period.start_utc - pd.Timedelta(days=days)
    mask = (table["timestamp_utc"] >= begin) & (table["timestamp_utc"] < period.start_utc)
    history = table.loc[mask].copy()
    if history.empty:
        raise ValueError("Not enough history to build trailing-average benchmark.")
    return BenchmarkResult(
        method="trailing_average",
        value=calculate_realized_block_value(history, block),
        description=f"Realized average price over the previous {days} days.",
    )


def benchmark_same_month_history(
    feature_table: pd.DataFrame,
    period: DeliveryPeriod,
    block: str,
) -> BenchmarkResult:
    """Use historical realized prices from the same calendar month in prior years."""

    table = feature_table.copy()
    timestamps = pd.to_datetime(table["timestamp_utc"], utc=True)
    period_month = period.start_utc.month
    history_mask = (
        (timestamps < period.start_utc)
        & (timestamps.dt.month == period_month)
    )
    history = table.loc[history_mask].copy()
    if history.empty:
        raise ValueError("Not enough history to build same-month benchmark.")
    return BenchmarkResult(
        method="same_month_history",
        value=calculate_realized_block_value(history, block),
        description="Realized average from the same calendar month in earlier years.",
    )


def calculate_realized_block_value(table: pd.DataFrame, block: str) -> float:
    """Calculate a historical realized benchmark for the selected block."""

    if block == "baseload":
        return float(table[modelling_constants.TARGET_COLUMN].mean())

    timestamps = pd.to_datetime(table["timestamp_utc"], utc=True)
    peak_mask = peakload_mask(timestamps)
    if block == "peakload":
        return float(table.loc[peak_mask, modelling_constants.TARGET_COLUMN].mean())
    if block == "offpeak":
        return float(table.loc[~peak_mask, modelling_constants.TARGET_COLUMN].mean())
    if block == "peak_base_spread":
        peakload = float(table.loc[peak_mask, modelling_constants.TARGET_COLUMN].mean())
        baseload = float(table[modelling_constants.TARGET_COLUMN].mean())
        return peakload - baseload
    raise ValueError(f"Unsupported block: {block}")


def build_benchmark(
    method: str,
    feature_table: pd.DataFrame,
    period: DeliveryPeriod,
    block: str,
    manual_value: float | None = None,
) -> BenchmarkResult:
    """Build the requested benchmark."""

    if method == "manual":
        if manual_value is None:
            raise ValueError("Manual benchmark needs --curve-price.")
        return benchmark_manual(manual_value)
    if method == "trailing_average":
        return benchmark_trailing_average(feature_table, period, block)
    if method == "same_month_history":
        return benchmark_same_month_history(feature_table, period, block)
    raise ValueError(f"Unsupported benchmark method: {method}")


def read_metric_value(metrics_path: str | Path, column: str) -> float:
    """Read one metric value from a validation or holdout metrics CSV."""

    metrics = pd.read_csv(metrics_path, sep=None, engine="python")
    if column not in metrics.columns:
        return float("nan")
    return float(pd.to_numeric(metrics[column], errors="coerce").dropna().iloc[0])


def choose_tail_metric(edge: float, metrics_path: str | Path) -> tuple[str, float]:
    """Use upside or downside stress error depending on signal direction."""

    metric_name = "top_decile_mae" if edge >= 0 else "bottom_decile_mae"
    return metric_name, read_metric_value(metrics_path, metric_name)


def choose_signal(score: float) -> str:
    """Map risk-adjusted edge into a trading signal."""

    if score >= constants.STRONG_LONG_THRESHOLD:
        return "Strong long"
    if score >= constants.LONG_THRESHOLD:
        return "Long"
    if score <= constants.STRONG_SHORT_THRESHOLD:
        return "Strong short"
    if score <= constants.SHORT_THRESHOLD:
        return "Short"
    return "Neutral"


def desk_action_for_signal(signal: str, block: str) -> str:
    """Explain how a desk could express the signal."""

    if "long" in signal.lower():
        return f"Buy or keep long exposure in the selected {block} delivery block."
    if "short" in signal.lower():
        return f"Sell or keep short exposure in the selected {block} delivery block."
    return "Do not add directional exposure; monitor until the edge clears the risk buffer."


def invalidation_logic() -> str:
    """Return generic invalidation rules for the curve view."""

    return (
        "Invalidate or resize the signal if updated load, wind, or solar forecasts "
        "materially change residual load; if outage, flow, or market-regime news "
        "changes the supply stack; if recent model error exceeds validation error; "
        "or if liquidity/execution prices differ materially from the benchmark."
    )


def build_curve_view(
    predictions: pd.DataFrame,
    feature_table: pd.DataFrame,
    metrics_path: str | Path,
    period: DeliveryPeriod,
    block: str,
    benchmark_method: str,
    manual_curve_price: float | None = None,
) -> CurveView:
    """Build one fair-value and signal view for a selected delivery block."""

    period_predictions = filter_delivery_period(predictions, period)
    if period_predictions.empty:
        raise ValueError("Prediction file has no rows for the requested period.")

    block_values = calculate_block_values(period_predictions)
    if block not in {"baseload", "peakload", "offpeak", "peak_base_spread"}:
        raise ValueError(f"Unsupported block: {block}")

    forecast_value = float(getattr(block_values, block))
    benchmark = build_benchmark(
        benchmark_method,
        feature_table,
        period,
        block,
        manual_value=manual_curve_price,
    )
    edge = forecast_value - benchmark.value
    mae = read_metric_value(metrics_path, "mae")
    tail_metric_name, tail_metric_value = choose_tail_metric(edge, metrics_path)
    risk_buffer = max(mae, constants.TAIL_RISK_WEIGHT * tail_metric_value)
    confidence_score = edge / risk_buffer if risk_buffer else float("nan")
    signal = choose_signal(confidence_score)
    prediction_coverage = block_values.predicted_row_count / block_values.row_count

    return CurveView(
        period_start_utc=str(period.start_utc),
        period_end_utc=str(period.end_utc),
        block=block,
        forecast_fair_value=forecast_value,
        benchmark_method=benchmark.method,
        benchmark_value=benchmark.value,
        edge=edge,
        mae=mae,
        tail_metric_name=tail_metric_name,
        tail_metric_value=tail_metric_value,
        risk_buffer=risk_buffer,
        confidence_score=confidence_score,
        signal=signal,
        desk_action=desk_action_for_signal(signal, block),
        invalidation_logic=invalidation_logic(),
        prediction_coverage=prediction_coverage,
    )


def write_curve_view_outputs(view: CurveView, output_folder: str | Path) -> tuple[Path, Path]:
    """Write CSV and Markdown outputs for the curve view."""

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    summary_path = output_folder / "curve_view_summary.csv"
    report_path = output_folder / "curve_view_report.md"
    pd.DataFrame([asdict(view)]).round(2).to_csv(summary_path, index=False)

    report = f"""# Prompt Curve Fair-Value View

## Delivery Period

- Period UTC: `{view.period_start_utc}` to `{view.period_end_utc}`; end is exclusive.
- Block: `{view.block}`
- Prediction coverage: `{view.prediction_coverage:.2%}`

## Fair Value

- Forecast fair value: `{view.forecast_fair_value:.2f}` EUR/MWh
- Benchmark method: `{view.benchmark_method}`
- Benchmark value: `{view.benchmark_value:.2f}` EUR/MWh
- Edge: `{view.edge:.2f}` EUR/MWh

## Risk Adjustment

- MAE: `{view.mae:.2f}` EUR/MWh
- Tail metric: `{view.tail_metric_name}` = `{view.tail_metric_value:.2f}` EUR/MWh
- Risk buffer: `{view.risk_buffer:.2f}` EUR/MWh
- Confidence score: `{view.confidence_score:.2f}`

## Signal

- Signal: **{view.signal}**
- Desk action: {view.desk_action}

## Invalidation Logic

{view.invalidation_logic}
"""
    report_path.write_text(report, encoding="utf-8")
    return summary_path, report_path
