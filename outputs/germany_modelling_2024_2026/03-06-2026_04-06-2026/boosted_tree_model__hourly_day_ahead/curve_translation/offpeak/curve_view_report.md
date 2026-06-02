# Prompt Curve Fair-Value View

## Delivery Period

- Period UTC: `2026-06-02 22:00:00+00:00` to `2026-06-03 22:00:00+00:00`; end is exclusive.
- Block: `offpeak`
- Prediction coverage: `100.00%`

## Fair Value

- Forecast fair value: `122.55` EUR/MWh
- Forecast band: `115.80` to `129.30` EUR/MWh (`p10_p90_residual`)
- Benchmark method: `trailing_average`
- Benchmark value: `115.79` EUR/MWh
- Edge vs benchmark: `6.75` EUR/MWh

## Signal

- Signal: **Long**
- Distance beyond band edge: `0.00` EUR/MWh
- Rationale: Benchmark 115.79 sits below the offpeak forecast band (115.80-129.30 EUR/MWh) by 0.00 EUR/MWh, supporting a long view.
- Desk action: Buy or keep long exposure in the selected offpeak delivery block.

## Model Error Context

- MAE: `6.75` EUR/MWh
- Tail metric: `top_decile_mae` = `4.98` EUR/MWh

## Invalidation Logic

Invalidate or resize the signal if updated load, wind, or solar forecasts materially change residual load; if outage, flow, or market-regime news changes the supply stack; if recent model error exceeds validation error; or if liquidity/execution prices differ materially from the benchmark.
