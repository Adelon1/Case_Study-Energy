# Prompt Curve Fair-Value View

## Delivery Period

- Period UTC: `2025-12-01 00:00:00+00:00` to `2025-12-02 00:00:00+00:00`; end is exclusive.
- Block: `peakload`
- Prediction coverage: `100.00%`

## Fair Value

- Forecast fair value: `107.15` EUR/MWh
- Forecast band: `98.12` to `116.18` EUR/MWh (`p10_p90_residual`)
- Benchmark method: `trailing_average`
- Benchmark value: `128.66` EUR/MWh
- Edge vs benchmark: `-21.51` EUR/MWh

## Signal

- Signal: **Short**
- Distance beyond band edge: `12.48` EUR/MWh
- Rationale: Benchmark 128.66 sits above the peakload forecast band (98.12-116.18 EUR/MWh) by 12.48 EUR/MWh, supporting a short view.
- Desk action: Sell or keep short exposure in the selected peakload delivery block.

## Model Error Context

- MAE: `9.03` EUR/MWh
- Tail metric: `bottom_decile_mae` = `4.87` EUR/MWh

## Invalidation Logic

Invalidate or resize the signal if updated load, wind, or solar forecasts materially change residual load; if outage, flow, or market-regime news changes the supply stack; if recent model error exceeds validation error; or if liquidity/execution prices differ materially from the benchmark.
