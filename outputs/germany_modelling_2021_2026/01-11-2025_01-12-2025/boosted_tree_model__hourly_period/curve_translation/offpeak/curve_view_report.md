# Prompt Curve Fair-Value View

## Delivery Period

- Period UTC: `2025-11-01 00:00:00+00:00` to `2025-12-01 00:00:00+00:00`; end is exclusive.
- Block: `offpeak`
- Prediction coverage: `100.00%`

## Fair Value

- Forecast fair value: `95.60` EUR/MWh
- Forecast band: `79.83` to `111.37` EUR/MWh (`p10_p90_residual`)
- Benchmark method: `trailing_average`
- Benchmark value: `69.21` EUR/MWh
- Edge vs benchmark: `26.39` EUR/MWh

## Signal

- Signal: **Long**
- Distance beyond band edge: `10.62` EUR/MWh
- Rationale: Benchmark 69.21 sits below the offpeak forecast band (79.83-111.37 EUR/MWh) by 10.62 EUR/MWh, supporting a long view.
- Desk action: Buy or keep long exposure in the selected offpeak delivery block.

## Model Error Context

- MAE: `15.77` EUR/MWh
- Tail metric: `top_decile_mae` = `43.72` EUR/MWh

## Invalidation Logic

Invalidate or resize the signal if updated load, wind, or solar forecasts materially change residual load; if outage, flow, or market-regime news changes the supply stack; if recent model error exceeds validation error; or if liquidity/execution prices differ materially from the benchmark.
