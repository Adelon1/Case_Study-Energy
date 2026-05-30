# European Power Fair Value Case Study Report

Name: Rawad Batous  
Market: Germany/Luxembourg bidding zone  
Target: Hourly day-ahead electricity price

## 1. Data Ingestion and QA

Data source:

- ENTSO-E Transparency Platform REST API

Datasets:

- Hourly day-ahead prices
- Load forecast
- Solar forecast
- Wind onshore forecast
- Wind offshore forecast

Timezone handling:

- `timestamp_utc` is the canonical join key.
- German local time is used for reporting and curve block definitions.
- UTC avoids duplicate/missing timestamp keys across DST changes.

QA outputs:

- Missing values by field
- Duplicate timestamps
- Obvious outlier flags
- Coverage
- Timestamp alignment
- Leakage-safe `t-24h` imputation for missing forecast fundamentals

Generated QA file:

```text
data/processed/.../data_qa_report.md
```

## 2. Forecasting and Validation

Target approach:

- Option A: forecast hourly day-ahead prices and aggregate to curve-relevant delivery periods.

Feature design:

- German local calendar features are used for hour, weekday, month, and annual cycles.
- UTC remains the canonical timestamp, and full daily price-curve lag features are UTC-based to avoid DST holes.
- LEAR-style price curve lags include `d-1`, `d-2`, `d-3`, and `d-7`.

Models:

- Seasonal naive baseline using lagged prices.
- LEAR-style 24-hour regularised ARX model.
- Histogram gradient boosting nonlinear benchmark.

Validation:

- Rolling-window time-series validation.
- Final validation cadence uses monthly steps.
- Final holdout test from the end of the available complete data.
- No random K-fold split.
- Scalers are fitted inside model training folds through sklearn pipelines.

Metrics:

- MAE
- RMSE
- Bias
- Top-decile MAE
- Bottom-decile MAE
- Negative-price MAE
- Scarcity-price MAE
- Relative MAE versus baseline where baseline outputs are available

Selected model:

- LASSO raw-price LEAR-style model, selected by rolling validation.
- ElasticNet and asinh transforms were considered; raw LASSO was kept because validation performance was better on this dataset.
- Histogram gradient boosting is kept as a nonlinear benchmark, including absolute-error loss candidates.

## 3. Prompt Curve Translation

Method:

1. Use hourly predictions for a requested delivery period.
2. Aggregate into baseload, peakload, offpeak, or peak/base spread.
3. Compare forecast fair value against a proxy benchmark or manual curve price.
4. Convert edge into a risk-adjusted signal using MAE and tail metric buffers.

Important limitation:

- The default benchmark is not a traded forward price. It is a realised day-ahead proxy. The same code can compare against a real prompt-week or prompt-month forward price if supplied manually.

Signals:

- Strong long
- Long
- Neutral
- Short
- Strong short

Invalidation examples:

- Major load, wind, or solar forecast revision
- Outage or plant-return news
- Interconnector or flow disruption
- Recent model error exceeding validation levels
- Liquidity or execution price far from benchmark

## 4. AI-Accelerated Workflow

Implemented component:

- Programmatic AI commentary generation from curve-view summary tables.

Properties:

- Calls an LLM from code through the OpenAI API.
- Loads API key from environment variables only.
- Logs prompts.
- Logs outputs.
- Logs failures.
- Writes deterministic fallback commentary if the API call fails.

Files:

```text
pipeline_steps/generate_ai_commentary.py
```

Outputs:

```text
ai_commentary.md
ai_logs/*_prompt.json
ai_logs/*_output.json
ai_logs/*_failure.json
```

## Limitations and Future Work

- Add longer final holdout only if evaluating broader robustness rather than one-month operational forecasting.
- Add neighbouring market prices, fuel, carbon, and flow features if public data access is available.
