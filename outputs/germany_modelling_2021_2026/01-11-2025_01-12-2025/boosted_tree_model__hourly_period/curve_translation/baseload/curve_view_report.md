# Prompt Curve Fair-Value View

## Delivery Period

- Period UTC: `2025-11-01 00:00:00+00:00` to `2025-12-01 00:00:00+00:00`; end is exclusive.
- Block: `baseload`
- Prediction coverage: `100.00%`

## Fair Value

- Forecast fair value: `103.66` EUR/MWh
- Forecast band: `87.89` to `119.43` EUR/MWh (`p10_p90_residual`)
- Benchmark method: `trailing_average`
- Benchmark value: `83.27` EUR/MWh
- Edge vs benchmark: `20.39` EUR/MWh

## Signal

- Signal: **Long**
- Distance beyond band edge: `4.62` EUR/MWh
- Rationale: Benchmark 83.27 sits below the baseload forecast band (87.89-119.43 EUR/MWh) by 4.62 EUR/MWh, supporting a long view.
- Desk action: Buy or keep long exposure in the selected baseload delivery block.

## Model Error Context

- MAE: `15.77` EUR/MWh
- Tail metric: `top_decile_mae` = `43.72` EUR/MWh

## Invalidation Logic

Invalidate or resize the signal if updated load, wind, or solar forecasts materially change residual load; if outage, flow, or market-regime news changes the supply stack; if recent model error exceeds validation error; or if liquidity/execution prices differ materially from the benchmark.
