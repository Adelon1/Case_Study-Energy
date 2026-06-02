# Forecast View Report

## Curve Signals

| Block | FV | Benchmark | Edge | Signal | Coverage |
| --- | ---: | ---: | ---: | --- | ---: |
| baseload | 96.23 | 101.93 | -5.70 | Neutral | 100.0% |
| peakload | 107.15 | 128.66 | -21.51 | Short | 100.0% |
| offpeak | 85.31 | 88.56 | -3.25 | Neutral | 100.0% |

## Desk Interpretation

- **baseload**: Benchmark 101.93 sits inside the baseload forecast band (87.20-105.26 EUR/MWh), so there is no directional conviction. Do not add directional exposure; monitor until the benchmark moves outside the forecast band.
- **peakload**: Benchmark 128.66 sits above the peakload forecast band (98.12-116.18 EUR/MWh) by 12.48 EUR/MWh, supporting a short view. Sell or keep short exposure in the selected peakload delivery block.
- **offpeak**: Benchmark 88.56 sits inside the offpeak forecast band (76.28-94.34 EUR/MWh), so there is no directional conviction. Do not add directional exposure; monitor until the benchmark moves outside the forecast band.

## Invalidation Logic

Invalidate or resize the signal if updated load, wind, or solar forecasts materially change residual load; if outage, flow, or market-regime news changes the supply stack; if recent model error exceeds validation error; or if liquidity/execution prices differ materially from the benchmark.
