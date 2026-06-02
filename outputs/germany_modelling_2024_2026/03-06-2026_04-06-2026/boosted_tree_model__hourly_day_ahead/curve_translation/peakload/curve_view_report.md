# Prompt Curve Fair-Value View

## Delivery Period

- Period UTC: `2026-06-02 22:00:00+00:00` to `2026-06-03 22:00:00+00:00`; end is exclusive.
- Block: `peakload`
- Prediction coverage: `100.00%`

## Fair Value

- Forecast fair value: `80.65` EUR/MWh
- Forecast band: `73.90` to `87.40` EUR/MWh (`p10_p90_residual`)
- Benchmark method: `trailing_average`
- Benchmark value: `87.79` EUR/MWh
- Edge vs benchmark: `-7.14` EUR/MWh

## Signal

- Signal: **Short**
- Distance beyond band edge: `0.39` EUR/MWh
- Rationale: Benchmark 87.79 sits above the peakload forecast band (73.90-87.40 EUR/MWh) by 0.39 EUR/MWh, supporting a short view.
- Desk action: Sell or keep short exposure in the selected peakload delivery block.

## Model Error Context

- MAE: `6.75` EUR/MWh
- Tail metric: `bottom_decile_mae` = `6.22` EUR/MWh

## Invalidation Logic

Invalidate or resize the signal if updated load, wind, or solar forecasts materially change residual load; if outage, flow, or market-regime news changes the supply stack; if recent model error exceeds validation error; or if liquidity/execution prices differ materially from the benchmark.
