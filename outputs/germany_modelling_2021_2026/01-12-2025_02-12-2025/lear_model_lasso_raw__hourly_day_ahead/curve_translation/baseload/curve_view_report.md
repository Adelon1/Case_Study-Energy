# Prompt Curve Fair-Value View

## Delivery Period

- Period UTC: `2025-12-01 00:00:00+00:00` to `2025-12-02 00:00:00+00:00`; end is exclusive.
- Block: `baseload`
- Prediction coverage: `100.00%`

## Fair Value

- Forecast fair value: `96.23` EUR/MWh
- Forecast band: `87.20` to `105.26` EUR/MWh (`p10_p90_residual`)
- Benchmark method: `trailing_average`
- Benchmark value: `101.93` EUR/MWh
- Edge vs benchmark: `-5.70` EUR/MWh

## Signal

- Signal: **Neutral**
- Distance beyond band edge: `0.00` EUR/MWh
- Rationale: Benchmark 101.93 sits inside the baseload forecast band (87.20-105.26 EUR/MWh), so there is no directional conviction.
- Desk action: Do not add directional exposure; monitor until the benchmark moves outside the forecast band.

## Model Error Context

- MAE: `9.03` EUR/MWh
- Tail metric: `bottom_decile_mae` = `4.87` EUR/MWh

## Invalidation Logic

Invalidate or resize the signal if updated load, wind, or solar forecasts materially change residual load; if outage, flow, or market-regime news changes the supply stack; if recent model error exceeds validation error; or if liquidity/execution prices differ materially from the benchmark.
