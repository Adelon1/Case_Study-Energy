# Forecast View Report

## Curve Signals

| Block | FV | Benchmark | Edge | Signal | Coverage |
| --- | ---: | ---: | ---: | --- | ---: |
| baseload | 101.60 | 105.53 | -3.93 | Neutral | 100.0% |
| peakload | 80.65 | 87.79 | -7.14 | Short | 100.0% |
| offpeak | 122.55 | 115.79 | +6.75 | Long | 100.0% |

## Desk Interpretation

- **baseload**: Benchmark 105.53 sits inside the baseload forecast band (94.85-108.35 EUR/MWh), so there is no directional conviction. Do not add directional exposure; monitor until the benchmark moves outside the forecast band.
- **peakload**: Benchmark 87.79 sits above the peakload forecast band (73.90-87.40 EUR/MWh) by 0.39 EUR/MWh, supporting a short view. Sell or keep short exposure in the selected peakload delivery block.
- **offpeak**: Benchmark 115.79 sits below the offpeak forecast band (115.80-129.30 EUR/MWh) by 0.00 EUR/MWh, supporting a long view. Buy or keep long exposure in the selected offpeak delivery block.

## Invalidation Logic

Invalidate or resize the signal if updated load, wind, or solar forecasts materially change residual load; if outage, flow, or market-regime news changes the supply stack; if recent model error exceeds validation error; or if liquidity/execution prices differ materially from the benchmark.
