# European Power Fair Value Case Study

Rawad Batous  
rawad.batous2006@gmail.com

## Executive Summary

This project builds a prototype workflow for a European power fair-value case study. The chosen market is the Germany/Luxembourg bidding zone. The prototype ingests public ENTSO-E data, constructs an hourly modelling dataset, validates baseline and improved day-ahead price models, translates model output into prompt-curve trading views, and includes a programmatic AI commentary component.

The project should be interpreted as a **daily rolling day-ahead forecasting workflow**, not as a one-shot model that can forecast an entire future month from today without updating. This distinction is critical. A daily day-ahead model may use yesterday's realised prices as lagged inputs. A one-shot next-month model cannot use future realised prices inside the delivery month unless those values are recursively predicted or explicitly treated as a perfect-foresight/oracle assumption.

A critical leakage audit found that the current feature builder still creates rolling price features using `price.shift(1).rolling(...)`. That is leakage for a full next-day auction forecast, because hour 23 of a delivery day would indirectly use hour 22 of the same delivery day. This must be fixed before final performance numbers are trusted. The correct conservative version is `price.shift(24).rolling(...)`, or removal of those rolling features.

The rest of the pipeline remains valuable: data ingestion, QA, feature construction, validation scaffolding, model interfaces, curve translation, and AI commentary are reusable. The main correction is to tighten the forecast information set and rebuild/revalidate after fixing leakage.

## Assignment Requirements

The case study asks for four must-have components:

1. Public data ingestion and data quality.
2. Forecasting and model validation.
3. Prompt curve translation.
4. AI-accelerated workflow.

The assignment recommends forecasting hourly day-ahead prices and deriving next-week or next-month averages from them. It explicitly requires time-series validation, leakage avoidance, MAE/RMSE, and at least one tail or stress metric if extremes are considered. It also requires a concrete translation from forecasts into long/neutral/short curve views and one LLM component called from code with prompt/output/failure logging.

Reference: `TASK.md`, sections 1-4.

## Market and Data Source Decision

Germany was chosen because it has a rich market structure, strong public data availability, and several fallback sources. The reviewed sources were:

- ENTSO-E Transparency Platform.
- Open Power System Data.
- SMARD / Bundesnetzagentur.
- Fraunhofer Energy-Charts API.

The final source hierarchy was:

- Primary source: ENTSO-E Transparency Platform.
- Fallback/cross-check: Fraunhofer Energy-Charts API.
- Germany-specific documentation and figures: SMARD / Bundesnetzagentur.
- Fast historical bootstrap option: Open Power System Data.

The decision to use ENTSO-E was based on its official public European transparency data and availability of day-ahead prices, load forecasts, wind forecasts, and solar forecasts. This decision is documented in `Document.md`, "Germany Data Source Decision".

## Public Data Ingestion

### Data Source

The implemented pipeline uses the ENTSO-E Transparency Platform REST API:

```text
https://web-api.tp.entsoe.eu/api
```

The project stores the ENTSO-E token in `.env`:

```text
ENTSOE_API_KEY=...
```

`.env.example` documents how to request the token using the ENTSO-E guide. This follows the decision in `Document.md`, "ENTSO-E Security Token Request".

### Dataset Choices

The main dataset uses:

- `day_ahead_prices`
- `load_forecast`
- `solar_forecast`
- `wind_onshore_forecast`
- `wind_offshore_forecast`

This satisfies the minimum requirement of hourly day-ahead prices plus at least two fundamental drivers. It uses forecasted fundamentals as the primary tradable input set, consistent with the methodological conclusion in `session.md`: the main tradable model should use ex-ante load/wind/solar forecasts.

### Storage Layout

The intended storage design is layered:

```text
data/
  raw/
  interim/
  processed/
```

Raw files preserve API responses, interim files contain one parsed CSV per dataset, and processed files contain the combined hourly table, feature table, QA report, model outputs, and curve reports. This layout follows `Document.md`, "Data Storage Layout".

Current code creates matched run folders through:

```text
pipeline_helpers/entsoe_data/dataset_folders.py
```

and the main data builder is:

```text
pipeline_steps/build_dataset.py
```

## Ingestion Pipeline Code

The ingestion pipeline is implemented as:

```text
pipeline_steps/build_dataset.py
```

It performs these stages:

1. Parse command-line arguments: selected datasets, local start/end dates, mode, `.env` path.
2. Convert German local date input into UTC API windows.
3. Split the requested window into monthly chunks.
4. Download ENTSO-E XML files for each dataset and chunk.
5. Parse XML files into one standardized interim CSV per dataset.
6. Combine all interim CSVs into one hourly dataset.
7. Apply leakage-safe missing-value imputation for forecast fundamentals.
8. Write a data QA report.
9. Build the feature table.

Important helper files:

- `pipeline_helpers/entsoe_data/entsoe_api.py`: sends ENTSO-E GET requests and saves XML responses.
- `pipeline_helpers/entsoe_data/entsoe_xml_to_csv.py`: parses ENTSO-E XML periods and points into standardized timestamp/value rows.
- `pipeline_helpers/entsoe_data/combine_dataset_csvs.py`: joins datasets by `timestamp_utc`, aggregates to hourly means, imputes missing forecast fundamentals.
- `pipeline_helpers/entsoe_data/build_features.py`: builds modelling features.
- `pipeline_helpers/entsoe_data/date_windows.py`: handles local date to UTC window conversion.

## Timezone and DST Handling

The canonical timestamp is:

```text
timestamp_utc
```

The project uses UTC for joining, filtering, validation, and model window splitting. This avoids duplicate or missing timestamp keys during daylight-saving-time transitions.

Local German time is still important for interpretation and trading blocks. The combined dataset includes:

```text
timestamp_local
```

derived from UTC using `Europe/Berlin`.

The project learned an important DST lesson during feature engineering. Full daily price-curve features were initially built using local market dates/hours. That created missing cells on spring DST days, where local hour 02:00 does not exist. The code was changed to build full daily price-curve lag features in UTC. This avoids 23/25-hour local-day matrix problems. This decision is not directly in `Document.md`, but it follows the same UTC-canonical design recorded in `Document.md`, "Data Storage Layout".

The later improvement added local calendar features only:

- `local_hour`
- `local_weekday`
- `local_month`
- `local_day_of_year`
- local cyclic encodings
- local weekday dummies

This is DST-safe because these are row-wise timestamp features, not full local-day pivots.

## Missing Data and Imputation

The ENTSO-E source can contain missing periods for some forecast series. We observed a concrete example where `load_forecast` was missing more than expected for a February 2022 window. Investigation showed this was source-side missing data rather than parser error.

The implemented imputation is in:

```text
pipeline_helpers/entsoe_data/combine_dataset_csvs.py
```

It applies after hourly assembly and before feature generation:

```text
missing value at time t = same column value at t - 24h
```

Only forecast driver columns are imputed:

- `load_forecast_mw`
- `solar_forecast_mw`
- `wind_onshore_forecast_mw`
- `wind_offshore_forecast_mw`

If `t-24h` is unavailable, rows with remaining missing forecast fundamentals are dropped before feature generation. This avoids future-value imputation and keeps the procedure leakage-safe for fundamentals.

## Data QA Report

`build_dataset.py` writes:

```text
data/processed/.../data_qa_report.md
```

The report includes:

- dataset run name
- source endpoint
- included datasets
- requested local delivery window
- API UTC window
- parsed input frequency
- final assembled frequency
- expected versus actual row count
- coverage
- duplicate timestamps
- missing values
- imputation counts
- simple outlier checks
- timestamp alignment explanation
- DST handling explanation
- known limitations

This satisfies the Task 1 requirement for generated QA output.

## Forecasting Methodology

### Target

The target is:

```text
day_ahead_price_eur_per_mwh
```

The chosen assignment approach is Option A: forecast hourly day-ahead prices and aggregate them into curve-relevant blocks. This is the assignment's recommended approach and is more flexible than directly forecasting a monthly average.

### Forecast Information Set

This is the most important modelling concept in the project.

The primary model should represent a tradable ex-ante information set:

- load forecast
- wind forecast
- solar forecast
- lagged prices known before the forecast
- calendar features

It should not use future actual load/wind/solar generation, and it should not use same-delivery-day actual prices when forecasting the full next delivery day.

This follows the methodological conclusion in `session.md`: the primary model should be a forecast-input/tradable/MOS-style model. Actual-input models may be useful only as structural/oracle benchmarks.

### Actual Inputs Versus Forecast Inputs

The external discussion in `session.md` framed this as:

- Perfect Prog / structural / oracle model: train on actual fundamentals, possibly deploy with forecasts.
- MOS-style / tradable model: train and deploy on forecasted fundamentals.

The main project should use the tradable forecast-input model:

```text
forecasted load, forecasted wind, forecasted solar -> price
```

because these are available before the auction. Actual realised fundamentals should only be presented as an oracle upper bound or structural benchmark if implemented.

This decision is supported by:

- `session.md`, "Recommended Two-Model Design".
- `session.md`, "Strong Final Position".
- `Document.md`, "External Discussion Notes in session.md".

## Literature and Paper Findings

The literature review in `session.md` shaped the modelling decisions.

### MOS versus Perfect Prog

Marzban, Sandgathe & Kalnay, "MOS, Perfect Prog, and Reanalysis" explains the statistical distinction between training on observations and training on model forecasts. In this project:

- training on actual fundamentals maps to Perfect Prog;
- training on forecasted fundamentals maps to MOS-style modelling.

Brunet, Verret & Yacowar and Wilson & Vallée were discussed as empirical comparisons. The conclusion was not that one approach universally dominates; rather, the correct choice depends on forecast horizon, forecast archive quality, updateability, and train-live mismatch.

### Forecast Error in Energy Problems

Fay & Ringwood support the argument that training on actual weather can be reasonable in energy forecasting, because forecast archives are noisy and weather models improve over time. But they also warn that live forecast errors can hurt operational performance.

Wang et al. and Runge & Saloux reinforce the distinction between prediction with actual contemporaneous inputs and true forecasting with future forecast inputs.

Fildes, Randall & Stubbs show that exogenous weather variables improve utility demand forecasting, while also emphasizing that ex-ante evaluation must account for whether explanatory variables are themselves forecasts.

### Electricity Price Forecasting Literature

Maciejowska, Nitka & Weron are especially relevant for Germany. They show that load, wind, and solar forecasts are biased and that improving these forecasts can improve electricity price forecasting. This supports using public forecast fundamentals while leaving separate fundamental forecast enhancement as future work.

Uniejewski & Ziel show that probabilistic forecasts of load, solar, and wind can improve electricity price forecasting. This supports the importance of fundamental forecast uncertainty, but it is beyond this prototype's scope.

Kulakov & Ziel and Goodarzi, Perera & Bunn show that renewable forecast errors affect intraday/spot prices and imbalance outcomes. This confirms that forecast errors are economically meaningful.

Beran, Vogler & Weber emphasize respecting information availability in German multi-day-ahead price forecasting. This is directly relevant to the leakage concern.

Lago et al. and Weron support the project's validation choices: baseline models, time-series splits, transparent metrics, leakage control, and reproducibility.

## Models Implemented

### Baseline: Seasonal Lag Model

File:

```text
pipeline_helpers/modelling/baseline_week_lag.py
```

The model predicts using a configured lag:

- 24 hours
- 48 hours
- 168 hours

It has a `train()` function for interface consistency, but no fitted state.

Purpose:

- simple benchmark;
- sanity check;
- relative MAE denominator for improved models.

### LEAR-Style Regularised ARX Model

File:

```text
pipeline_helpers/modelling/lear_model.py
```

This is described as a **LEAR-style 24-hour regularised ARX model**, not an exact reproduction of LEAR. It fits 24 separate regularised linear models, one per local delivery hour.

Supported regularisation:

- LASSO
- ElasticNet
- Ridge

Supported target transforms:

- raw
- asinh

The current project view is:

- keep asinh support in code;
- do not treat it as the final default if rolling validation shows raw prices perform better;
- describe asinh as tested/rejected if results deteriorate.

Main feature groups:

- forecast fundamentals;
- residual load and renewable shares;
- German local calendar features;
- local weekday dummies;
- lagged prices;
- UTC daily price-curve lags `d-1`, `d-2`, `d-3`, `d-7`.

Training diagnostics now include:

- fitted hours;
- number of training rows by hour;
- feature columns.

The model raises an error if not all 24 hourly models fit.

### Histogram Gradient Boosting Model

File:

```text
pipeline_helpers/modelling/hist_gradient_boosting.py
```

This is a nonlinear benchmark using sklearn's `HistGradientBoostingRegressor`.

The grid includes:

- `absolute_error` loss candidates;
- `squared_error` loss candidates;
- different learning rates, iterations, leaf counts, minimum leaf sizes, and L2 regularisation.

Internal early stopping is disabled because external rolling validation controls model selection.

HGB is allowed to receive NaNs in feature columns because sklearn's histogram gradient boosting can handle missing feature values. It still drops rows where the target is missing.

## Feature Engineering

File:

```text
pipeline_helpers/entsoe_data/build_features.py
```

Current feature construction:

1. Fundamental features:
   - total wind forecast;
   - renewable total forecast;
   - residual load forecast;
   - wind share of load;
   - solar share of load;
   - renewable share of load.

2. Calendar features:
   - UTC hour/weekday/month/day-of-year;
   - local German hour/weekday/month/day-of-year;
   - local cyclic encodings;
   - local weekday dummies.

3. Price lags:
   - `price_lag_24`;
   - `price_lag_48`;
   - `price_lag_168`.

4. Rolling price features:
   - `price_rolling_mean_24`;
   - `price_rolling_std_24`;
   - `price_rolling_mean_168`;
   - `price_rolling_std_168`.

5. Daily UTC price-curve lags:
   - `price_d1_h00` ... `price_d1_h23`;
   - `price_d2_h00` ... `price_d2_h23`;
   - `price_d3_h00` ... `price_d3_h23`;
   - `price_d7_h00` ... `price_d7_h23`;
   - `price_d1_min`, `price_d1_max`, `price_d1_mean`;
   - `price_d7_min`, `price_d7_max`, `price_d7_mean`.

## Critical Leakage Audit

This section is intentionally explicit because it is the largest current modelling risk.

The feature builder currently creates rolling price features from:

```python
shifted_price = price.shift(1)
features["price_rolling_mean_24"] = shifted_price.rolling(24).mean()
features["price_rolling_std_24"] = shifted_price.rolling(24).std()
features["price_rolling_mean_168"] = shifted_price.rolling(168).mean()
features["price_rolling_std_168"] = shifted_price.rolling(168).std()
```

This is safe for sequential one-hour-ahead forecasting after the previous hour's actual price is known. It is **not safe** for a day-ahead full-delivery-day forecast made before the auction, because hour 23 of the delivery day may use hour 22 of the same delivery day through the rolling window.

Therefore, current validation results using these rolling features may be overstated.

The conservative fix is:

```python
safe_price = price.shift(24)
features["price_rolling_mean_24"] = safe_price.rolling(24).mean()
features["price_rolling_std_24"] = safe_price.rolling(24).std()
features["price_rolling_mean_168"] = safe_price.rolling(168).mean()
features["price_rolling_std_168"] = safe_price.rolling(168).std()
```

or remove rolling price features from model feature sets entirely.

This issue is conceptually related to `session.md`, especially Beran, Vogler & Weber and the repeated emphasis on respecting information available at or before forecast time.

## Validation Design

Files:

```text
pipeline_helpers/modelling/validation.py
pipeline_steps/validate_model.py
```

The model interface is:

```python
def train(train_data, params):
    ...

def predict(model_state, test_data, params):
    ...
```

This was decided in `Document.md`, "Modelling Validation Design".

The validation engine:

1. Loads the feature table.
2. Infers complete calendar-month windows.
3. Reserves the final test window from the end of the data.
4. Builds rolling train/test windows before the final holdout.
5. Loops over each model's parameter grid.
6. Scores each fold.
7. Chooses best parameters by MAE with stress metrics as tie-breakers.
8. Evaluates the selected parameters on the final holdout.
9. Saves metrics, predictions, diagnostics, and model artifacts.

Current constants:

```python
TRAIN_MONTHS = 24
TEST_MONTHS = 1
STEP_MONTHS = 1
```

Earlier discussion used `STEP_MONTHS = 3` for faster experiments and `TEST_MONTHS = 1` to match near-term operational forecasting. The current setting of `STEP_MONTHS = 1` is intended for final validation coverage, while a larger step can still be used for quick experiments. This connects to `Document.md`, "Modelling Validation Design".

Validation outputs include:

- `validation_metrics.csv`
- `validation_summary.csv`
- `final_holdout_metrics.csv`
- `final_holdout_predictions.csv`
- prediction coverage diagnostics
- monthly metric breakdowns
- yearly metric breakdowns
- saved model state
- best parameter JSON
- model metadata JSON

## Metrics

Metrics include:

- MAE: average absolute hourly price error.
- RMSE: root mean squared error, more sensitive to large misses.
- Bias: average signed error.
- Top-decile MAE: MAE during highest-price hours.
- Bottom-decile MAE: MAE during lowest-price hours.
- Negative-price MAE.
- Scarcity-price MAE.
- Relative MAE versus baseline, when baseline outputs exist.

Tail metrics are required by the assignment if modelling extremes. The current top/bottom decile and scarcity/negative-price metrics satisfy this requirement at a basic level.

Known improvement:

- current top/bottom thresholds are based on each test window's true distribution;
- an even stricter ex-ante stress design would compute thresholds from the training window.

## Model Persistence

Validation now saves reusable model artifacts:

```text
final_holdout_model.joblib
best_params.json
model_metadata.json
```

`joblib` stores Python/sklearn model objects. It is not for manual viewing in Excel; it is loaded from Python with:

```python
import joblib
model_state = joblib.load("final_holdout_model.joblib")
```

Period-specific prediction training is handled by:

```text
pipeline_helpers/modelling/period_prediction.py
```

It trains on the previous rolling window and predicts a requested period, saving:

```text
period_predictions/YYYYMMDD_YYYYMMDD/
  predictions.csv
  metrics.csv
  model.joblib
  metadata.json
```

## Prompt Curve Translation

Files:

```text
pipeline_helpers/curve_translation/
pipeline_steps/translate_curve_view.py
```

The curve translation layer converts hourly predictions into:

- baseload;
- peakload;
- offpeak;
- peak/base spread.

Peakload/offpeak use German local market time:

```python
local = pd.to_datetime(timestamps, utc=True).dt.tz_convert("Europe/Berlin")
is_weekday = local.dt.weekday < 5
is_peak_hour = (local.dt.hour >= 8) & (local.dt.hour < 20)
```

This is correct for German market interpretation and does not affect UTC timestamp joins.

## Forward Prices and Benchmarks

The assignment says forward price data is not required. A forward price is a traded price today for electricity delivered in a future period, such as German next-month baseload.

Because reliable forward curve data is often paid/vendor data, the current implementation supports proxy benchmarks:

- trailing realised day-ahead average;
- same-month historical average;
- manual curve price if the user has one.

The report must be explicit:

> In absence of public forward price data, trailing realised day-ahead average is used as a proxy benchmark. The same method can compare forecast fair value against traded prompt-week/month forwards if available.

## Signal Logic

The curve view computes:

```text
edge = forecast fair value - benchmark
risk buffer = max(MAE, 0.5 * relevant tail MAE)
confidence score = edge / risk buffer
```

Signals:

- strong long;
- long;
- neutral;
- short;
- strong short.

Desk action text is generated from the signal. Invalidation logic includes:

- large load/wind/solar forecast revisions;
- outages or plant returns;
- flow disruptions;
- market regime changes;
- recent model error exceeding validation error;
- liquidity or execution constraints.

## Important Prompt-Month Interpretation

This project should not claim to forecast a full future month from today using future lagged actual prices.

There are two different meanings:

1. Daily rolling day-ahead workflow:
   - each day, forecast tomorrow;
   - yesterday's realised price curve is known;
   - prompt-week/month views are refreshed daily or historically simulated by aggregating rolling daily forecasts.

2. One-shot next-month forecast:
   - forecast the whole month from today;
   - future `d-1`, `d-2`, `d-7` actual prices inside the month are not known;
   - requires recursive predicted lags or removal of future-dependent lag features.

The defensible interpretation for this prototype is the first one: daily rolling day-ahead fair-value generation. The curve translation demonstrates how such hourly/daily forecasts can be aggregated into curve-relevant views.

This must be clearly stated in final materials.

## AI-Accelerated Workflow

File:

```text
pipeline_steps/generate_ai_commentary.py
```

The AI component:

1. Reads `curve_view_summary.csv`.
2. Builds a constrained prompt using only computed values.
3. Calls the OpenAI Responses API using `OPENAI_API_KEY`.
4. Logs the prompt.
5. Logs the output.
6. Logs failures.
7. Writes `ai_commentary.md`.
8. Writes deterministic fallback commentary if the API fails or quota is unavailable.

Environment variables:

```text
OPENAI_API_KEY
OPENAI_MODEL
```

This satisfies Task 4's minimum requirements. It is intentionally not a manual chat transcript.

## Code Map

Top-level pipeline scripts:

- `pipeline_steps/build_dataset.py`: download, parse, combine, QA, feature table.
- `pipeline_steps/validate_model.py`: rolling validation, final holdout, metrics, model artifacts.
- `pipeline_steps/translate_curve_view.py`: curve fair-value translation and optional AI commentary.
- `pipeline_steps/generate_ai_commentary.py`: standalone AI commentary generation.

Data helpers:

- `pipeline_helpers/entsoe_data/constants.py`
- `pipeline_helpers/entsoe_data/date_windows.py`
- `pipeline_helpers/entsoe_data/dataset_folders.py`
- `pipeline_helpers/entsoe_data/entsoe_api.py`
- `pipeline_helpers/entsoe_data/entsoe_xml_to_csv.py`
- `pipeline_helpers/entsoe_data/combine_dataset_csvs.py`
- `pipeline_helpers/entsoe_data/build_features.py`

Modelling helpers:

- `pipeline_helpers/modelling/constants.py`
- `pipeline_helpers/modelling/metrics.py`
- `pipeline_helpers/modelling/validation.py`
- `pipeline_helpers/modelling/baseline_week_lag.py`
- `pipeline_helpers/modelling/lear_model.py`
- `pipeline_helpers/modelling/hist_gradient_boosting.py`
- `pipeline_helpers/modelling/period_prediction.py`

Curve translation helpers:

- `pipeline_helpers/curve_translation/constants.py`
- `pipeline_helpers/curve_translation/forecast_blocks.py`
- `pipeline_helpers/curve_translation/curve_view.py`

Period prediction and prediction reuse are handled in:

- `pipeline_helpers/modelling/period_prediction.py`

## Reproducible Run Order

1. Create `.env` from `.env.example`.
2. Fill `ENTSOE_API_KEY`.
3. Optionally fill `OPENAI_API_KEY`.
4. Build dataset:

```bash
.venv/bin/python pipeline_steps/build_dataset.py \
  --datasets day_ahead_prices load_forecast solar_forecast wind_onshore_forecast wind_offshore_forecast \
  --start 01-01-2021 \
  --end 02-01-2026 \
  --mode modelling
```

5. Validate baseline:

```bash
.venv/bin/python pipeline_steps/validate_model.py \
  --features data/processed/germany_modelling_2021_2026/germany_model_features.csv \
  --model baseline_week_lag
```

6. Validate LEAR-style model:

```bash
.venv/bin/python pipeline_steps/validate_model.py \
  --features data/processed/germany_modelling_2021_2026/germany_model_features.csv \
  --model lear_model \
  --regularization lasso \
  --target-transform raw
```

7. Translate curve view:

```bash
.venv/bin/python pipeline_steps/translate_curve_view.py \
  --features data/processed/germany_modelling_2021_2026/germany_model_features.csv \
  --start 01-11-2025 \
  --end 01-12-2025 \
  --model lear_model \
  --regularization lasso \
  --target-transform raw \
  --block all \
  --benchmark trailing_average
```

8. Generate AI commentary:

```bash
.venv/bin/python pipeline_steps/generate_ai_commentary.py \
  --summary data/processed/germany_modelling_2021_2026/lear_model_lasso_raw/curve_translation/20251101_20251201/baseload/curve_view_summary.csv
```

## What Is Complete

Completed or structurally implemented:

- ENTSO-E ingestion with secrets in `.env`.
- Raw/interim/processed data layout.
- XML parsing to standardized CSVs.
- Hourly aggregation.
- Forecast fundamental imputation.
- QA report generation.
- Baseline model.
- LEAR-style regularised model.
- HGB nonlinear benchmark.
- Rolling validation and final holdout framework.
- Metrics including tail/stress metrics.
- Model artifact saving.
- Prompt-curve translation.
- Manual and proxy benchmarks.
- Local-time peak/offpeak.
- AI commentary code with prompt/output/failure logging and fallback.

## Critical Errors and Corrections Needed

### 1. Rolling price feature leakage

Current rolling price features use `price.shift(1)`. This leaks same-delivery-day prices for a full day-ahead forecast. Fix to `price.shift(24)` or remove these features.

Priority: highest.

### 2. Forecast contract must be explicit

The project must state that it is a daily rolling day-ahead workflow, not a one-shot full next-month forecast. Otherwise daily lag features become invalid for prompt-month interpretation.

Priority: highest.

### 3. Rebuild features after feature changes

The code now expects local calendar features and `d-3` curves. Existing feature CSVs created before these changes are stale. Rebuild the feature dataset before running validation.

Priority: high.

### 4. Re-run all validation after leakage fix

Any MAE values observed before fixing `shift(1)` rolling leakage should be considered provisional. Rebuild features and rerun baseline, LEAR, and HGB validation.

Priority: high.

### 5. Avoid overstating prompt-month live capability

The current curve translation can aggregate historical period predictions. It does not yet implement recursive one-shot next-month forecasting. The write-up must not imply that future month prices can be forecast from today using future lagged actuals.

Priority: high.

### 6. Forward benchmark limitation

The default benchmark is not a real forward price. It is a realised day-ahead proxy. The report must say this clearly.

Priority: medium.

### 7. Local-time features are safe only because full pivots remain UTC

Adding local calendar features is fine. Rebuilding full local daily price-curve pivots would reintroduce DST missing-hour problems unless 23/25-hour days are explicitly handled.

Priority: medium.

## Suggested Next Steps

1. Fix rolling price features to use `price.shift(24)` or remove them.
2. Rebuild `germany_model_features.csv`.
3. Rerun baseline validation.
4. Rerun LEAR validation.
5. Rerun HGB validation only if time allows.
6. Regenerate curve translation outputs.
7. Regenerate AI commentary or fallback.
8. Update final report with post-fix metrics.

## Final Position

The project is salvageable and structurally strong, but the current performance numbers should not be presented as final until the rolling price leakage is fixed. The correct framing is:

> This is a daily rolling German day-ahead price forecasting prototype. It uses public ex-ante forecast fundamentals, lagged historical prices, and calendar features to produce hourly price forecasts. These forecasts are then aggregated into curve-relevant fair-value views. Prompt-week/month views are interpreted as rolling daily fair-value updates or historical simulations, not one-shot forecasts that know future realised lagged prices.

That framing is consistent with the assignment, the data available, and the literature reviewed in `session.md`.
