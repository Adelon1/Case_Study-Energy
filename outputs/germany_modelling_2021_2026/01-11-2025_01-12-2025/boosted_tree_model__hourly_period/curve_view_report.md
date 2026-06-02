# Forecast View Report

## Curve Signals

| Block | FV | Benchmark | Edge | Signal | Coverage |
| --- | ---: | ---: | ---: | --- | ---: |
| baseload | 103.66 | 83.27 | +20.39 | Long | 100.0% |
| peakload | 119.79 | 107.57 | +12.21 | Neutral | 100.0% |
| offpeak | 95.60 | 69.21 | +26.39 | Long | 100.0% |

## Desk Interpretation

- **baseload**: Benchmark 83.27 sits below the baseload forecast band (87.89-119.43 EUR/MWh) by 4.62 EUR/MWh, supporting a long view. Buy or keep long exposure in the selected baseload delivery block.
- **peakload**: Benchmark 107.57 sits inside the peakload forecast band (104.02-135.56 EUR/MWh), so there is no directional conviction. Do not add directional exposure; monitor until the benchmark moves outside the forecast band.
- **offpeak**: Benchmark 69.21 sits below the offpeak forecast band (79.83-111.37 EUR/MWh) by 10.62 EUR/MWh, supporting a long view. Buy or keep long exposure in the selected offpeak delivery block.

## Invalidation Logic

Invalidate or resize the signal if updated load, wind, or solar forecasts materially change residual load; if outage, flow, or market-regime news changes the supply stack; if recent model error exceeds validation error; or if liquidity/execution prices differ materially from the benchmark.
