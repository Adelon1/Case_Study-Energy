# Prompt Curve Fair-Value View

## Delivery Period

- Period UTC: `2026-06-02 22:00:00+00:00` to `2026-06-03 22:00:00+00:00`; end is exclusive.
- Block: `baseload`
- Prediction coverage: `100.00%`

## Fair Value

- Forecast fair value: `101.60` EUR/MWh
- Forecast band: `94.85` to `108.35` EUR/MWh (`p10_p90_residual`)
- Benchmark method: `trailing_average`
- Benchmark value: `105.53` EUR/MWh
- Edge vs benchmark: `-3.93` EUR/MWh

## Signal

- Signal: **Neutral**
- Distance beyond band edge: `0.00` EUR/MWh
- Rationale: Benchmark 105.53 sits inside the baseload forecast band (94.85-108.35 EUR/MWh), so there is no directional conviction.
- Desk action: Do not add directional exposure; monitor until the benchmark moves outside the forecast band.

## Model Error Context

- MAE: `6.75` EUR/MWh
- Tail metric: `bottom_decile_mae` = `6.22` EUR/MWh

## Invalidation Logic

Invalidate or resize the signal if updated load, wind, or solar forecasts materially change residual load; if outage, flow, or market-regime news changes the supply stack; if recent model error exceeds validation error; or if liquidity/execution prices differ materially from the benchmark.
