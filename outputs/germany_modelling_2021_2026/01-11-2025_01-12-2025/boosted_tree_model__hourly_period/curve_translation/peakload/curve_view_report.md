# Prompt Curve Fair-Value View

## Delivery Period

- Period UTC: `2025-11-01 00:00:00+00:00` to `2025-12-01 00:00:00+00:00`; end is exclusive.
- Block: `peakload`
- Prediction coverage: `100.00%`

## Fair Value

- Forecast fair value: `119.79` EUR/MWh
- Forecast band: `104.02` to `135.56` EUR/MWh (`p10_p90_residual`)
- Benchmark method: `trailing_average`
- Benchmark value: `107.57` EUR/MWh
- Edge vs benchmark: `12.21` EUR/MWh

## Signal

- Signal: **Neutral**
- Distance beyond band edge: `0.00` EUR/MWh
- Rationale: Benchmark 107.57 sits inside the peakload forecast band (104.02-135.56 EUR/MWh), so there is no directional conviction.
- Desk action: Do not add directional exposure; monitor until the benchmark moves outside the forecast band.

## Model Error Context

- MAE: `15.77` EUR/MWh
- Tail metric: `top_decile_mae` = `43.72` EUR/MWh

## Invalidation Logic

Invalidate or resize the signal if updated load, wind, or solar forecasts materially change residual load; if outage, flow, or market-regime news changes the supply stack; if recent model error exceeds validation error; or if liquidity/execution prices differ materially from the benchmark.
