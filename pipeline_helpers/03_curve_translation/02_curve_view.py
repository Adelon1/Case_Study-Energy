"""Build risk-adjusted prompt-curve views from forecast blocks.

Public entry points:
    ``build_curve_view(...)``
    ``write_curve_view_outputs(...)``

Everything else is a helper for one of the three translation steps:
    benchmark construction -> forecast band/risk buffer -> trading signal text.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
from pathlib import Path

import pandas as pd

constants = importlib.import_module("pipeline_helpers.03_curve_translation.00_constants")
forecast_blocks = importlib.import_module("pipeline_helpers.03_curve_translation.01_forecast_blocks")
modelling_constants = importlib.import_module("pipeline_helpers.02_modelling.00_constants")

BlockBand = forecast_blocks.BlockBand
DeliveryPeriod = forecast_blocks.DeliveryPeriod
calculate_block_band = forecast_blocks.calculate_block_band
calculate_block_values = forecast_blocks.calculate_block_values
filter_delivery_period = forecast_blocks.filter_delivery_period
peakload_mask = forecast_blocks.peakload_mask


@dataclass(frozen=True)
class BenchmarkResult:
    """Benchmark price used as the reference for the forecast fair value."""

    method: str
    value: float
    description: str


@dataclass(frozen=True)
class CurveView:
    """Final prompt-curve trading view for one delivery block."""

    period_start_utc: str
    period_end_utc: str
    block: str
    forecast_fair_value: float
    forecast_low: float
    forecast_high: float
    band_source: str
    benchmark_method: str
    benchmark_value: float
    edge: float
    signal: str
    signal_margin: float
    desk_action: str
    decision_rationale: str
    invalidation_logic: str
    mae: float
    tail_metric_name: str
    tail_metric_value: float
    prediction_coverage: float


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


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
            raise ValueError("Manual benchmark needs a curve price.")
        return benchmark_manual(manual_value)
    if method == "trailing_average":
        return benchmark_trailing_average(feature_table, period, block)
    if method == "same_month_history":
        return benchmark_same_month_history(feature_table, period, block)
    raise ValueError(f"Unsupported benchmark method: {method}")


# ---------------------------------------------------------------------------
# Risk-buffer and signal helpers
# ---------------------------------------------------------------------------


def read_metric_value(metrics_path: str | Path, column: str) -> float:
    """Read one metric value from a metrics CSV."""

    metrics = pd.read_csv(metrics_path, sep=None, engine="python")
    if column not in metrics.columns:
        return float("nan")
    return float(pd.to_numeric(metrics[column], errors="coerce").dropna().iloc[0])


def choose_tail_metric(edge: float, metrics_path: str | Path) -> tuple[str, float]:
    """Use upside or downside stress error depending on signal direction."""

    metric_name = "top_decile_mae" if edge >= 0 else "bottom_decile_mae"
    return metric_name, read_metric_value(metrics_path, metric_name)


def derive_forecast_band(
    forecast_value: float,
    block_band: BlockBand,
    risk_buffer: float,
) -> tuple[float, float, str]:
    """Return the forecast band ``(low, high, source)`` for the signal.

    Prefer the empirical P10-P90 band aggregated from validation residuals. When
    that is unavailable (e.g. a freshly retrained period, or the peak/base
    spread), fall back to a symmetric band built from the model's risk buffer.
    """

    if block_band.has_band:
        return block_band.low, block_band.high, "p10_p90_residual"
    return forecast_value - risk_buffer, forecast_value + risk_buffer, "risk_buffer_proxy"


def derive_signal(forecast_low: float, forecast_high: float, benchmark: float) -> tuple[str, float]:
    """Map the benchmark's position relative to the forecast band into a signal.

    The band is the model's expected price range. If the benchmark sits below the
    whole band the model expects higher prices (go long); above the whole band,
    lower prices (go short); inside the band there is no conviction (neutral).
    A move of more than a full band width beyond the edge is a strong signal.
    """

    band_width = forecast_high - forecast_low

    if benchmark < forecast_low:
        margin = forecast_low - benchmark
        signal = "Strong long" if margin >= band_width else "Long"
        return signal, margin
    if benchmark > forecast_high:
        margin = benchmark - forecast_high
        signal = "Strong short" if margin >= band_width else "Short"
        return signal, margin
    return "Neutral", 0.0


def build_decision_rationale(
    block: str,
    forecast_low: float,
    forecast_high: float,
    benchmark_value: float,
    signal: str,
    signal_margin: float,
) -> str:
    """Explain in plain language why the signal was produced."""

    band_text = f"{forecast_low:.2f}-{forecast_high:.2f} EUR/MWh"
    if signal == "Neutral":
        return (
            f"Benchmark {benchmark_value:.2f} sits inside the {block} forecast band "
            f"({band_text}), so there is no directional conviction."
        )
    direction = "below" if "long" in signal.lower() else "above"
    return (
        f"Benchmark {benchmark_value:.2f} sits {direction} the {block} forecast band "
        f"({band_text}) by {signal_margin:.2f} EUR/MWh, supporting a {signal.lower()} view."
    )


def desk_action_for_signal(signal: str, block: str) -> str:
    """Explain how a desk could express the signal."""

    if "long" in signal.lower():
        return f"Buy or keep long exposure in the selected {block} delivery block."
    if "short" in signal.lower():
        return f"Sell or keep short exposure in the selected {block} delivery block."
    return "Do not add directional exposure; monitor until the benchmark moves outside the forecast band."


def invalidation_logic() -> str:
    """Return generic invalidation rules for the curve view."""

    return (
        "Invalidate or resize the signal if updated load, wind, or solar forecasts "
        "materially change residual load; if outage, flow, or market-regime news "
        "changes the supply stack; if recent model error exceeds validation error; "
        "or if liquidity/execution prices differ materially from the benchmark."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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

    if block not in {"baseload", "peakload", "offpeak", "peak_base_spread"}:
        raise ValueError(f"Unsupported block: {block}")

    if (
        "prediction_granularity" in period_predictions.columns
        and (period_predictions["prediction_granularity"] == "period_average").all()
    ):
        if "block" not in period_predictions.columns:
            raise ValueError(
                "Period-average predictions need a block column with baseload/peakload/offpeak rows."
            )
        period_predictions = period_predictions.loc[period_predictions["block"] == block]
        if period_predictions.empty:
            raise ValueError(f"Period-average predictions contain no rows for block '{block}'.")
        forecast_value = float(period_predictions["y_pred"].dropna().mean())
        prediction_coverage = float(period_predictions["y_pred"].notna().mean())
        block_band = BlockBand(low=float("nan"), high=float("nan"), has_band=False)
    else:
        block_values = calculate_block_values(period_predictions)
        forecast_value = float(getattr(block_values, block))
        prediction_coverage = block_values.predicted_row_count / block_values.row_count
        block_band = calculate_block_band(period_predictions, block)

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

    forecast_low, forecast_high, band_source = derive_forecast_band(
        forecast_value, block_band, risk_buffer
    )
    signal, signal_margin = derive_signal(forecast_low, forecast_high, benchmark.value)
    decision_rationale = build_decision_rationale(
        block, forecast_low, forecast_high, benchmark.value, signal, signal_margin
    )

    return CurveView(
        period_start_utc=str(period.start_utc),
        period_end_utc=str(period.end_utc),
        block=block,
        forecast_fair_value=forecast_value,
        forecast_low=forecast_low,
        forecast_high=forecast_high,
        band_source=band_source,
        benchmark_method=benchmark.method,
        benchmark_value=benchmark.value,
        edge=edge,
        signal=signal,
        signal_margin=signal_margin,
        desk_action=desk_action_for_signal(signal, block),
        decision_rationale=decision_rationale,
        invalidation_logic=invalidation_logic(),
        mae=mae,
        tail_metric_name=tail_metric_name,
        tail_metric_value=tail_metric_value,
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
- Forecast band: `{view.forecast_low:.2f}` to `{view.forecast_high:.2f}` EUR/MWh (`{view.band_source}`)
- Benchmark method: `{view.benchmark_method}`
- Benchmark value: `{view.benchmark_value:.2f}` EUR/MWh
- Edge vs benchmark: `{view.edge:.2f}` EUR/MWh

## Signal

- Signal: **{view.signal}**
- Distance beyond band edge: `{view.signal_margin:.2f}` EUR/MWh
- Rationale: {view.decision_rationale}
- Desk action: {view.desk_action}

## Model Error Context

- MAE: `{view.mae:.2f}` EUR/MWh
- Tail metric: `{view.tail_metric_name}` = `{view.tail_metric_value:.2f}` EUR/MWh

## Invalidation Logic

{view.invalidation_logic}
"""
    report_path.write_text(report, encoding="utf-8")
    return summary_path, report_path
