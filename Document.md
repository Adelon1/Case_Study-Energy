# Project Documentation Log

## 2026-05-28 01: Discussion: Germany Data Source Decision

**Question / topic:**  
Which public databases should we use for the European power fair value case study, and should we focus on Germany?

**Answer / decision:**  
We decided to focus on **Germany** because it has several strong public data fallbacks and a rich market story around load, wind, solar, residual load, imports/exports, and negative price events.

For Germany, the reviewed sources were:

- **ENTSO-E Transparency Platform**
- **Open Power System Data**
- **SMARD / Bundesnetzagentur**
- **Fraunhofer Energy-Charts API**

The final preferred setup is:

- **Primary source:** ENTSO-E Transparency Platform  
- **Fallback / cross-check:** Fraunhofer Energy-Charts API  
- **Germany-specific documentation and figures:** SMARD / Bundesnetzagentur  
- **Fast historical bootstrap option:** Open Power System Data

We chose **ENTSO-E as the primary source** because it provides official public European transparency data, including day-ahead prices and leakage-safe day-ahead fundamentals such as load, wind, and solar forecasts. Energy-Charts and SMARD remain useful as fallbacks and validation sources.

**Follow-up notes:**  
The pipeline should load API keys and secrets from `.env`, commit only `.env.example`, and document how to obtain the ENTSO-E API token.

## 2026-05-28 01: ENTSO-E Security Token Request

**Question / topic:**  
How was the ENTSO-E API token requested?

**Answer / decision:**  
The ENTSO-E security token request was started by following the official guide:

https://transparencyplatform.zendesk.com/hc/en-us/articles/12845911031188-How-to-get-security-token

Access can take around **2 to 3 days**, so the project should continue with code scaffolding while waiting for the token.

**Follow-up notes:**  
The repository should include a local `.env` file for the real token, a committed `.env.example` file showing the expected layout, and an ENTSO-E API download module that reads `ENTSOE_API_KEY` from the environment.

## 2026-05-28 16:25: External Discussion Notes in session.md

**Question / topic:**  
How should important discussions from outside this chat be shared with the project?

**Answer / decision:**  
The file `session.md` will be used as a handover file for important external discussions. When it is updated, it should be read and treated as project context.

**Follow-up notes:**  
The current `session.md` discussion refined the modelling methodology:

- The **main tradable model** should train and evaluate on ex-ante forecast inputs such as load, wind, and solar forecasts.
- An **actual-input model** can still be useful as a structural / Perfect Prog / oracle benchmark.
- If data and time allow, compare three variants: actual inputs to actual test inputs, forecast inputs to forecast test inputs, and actual-trained model evaluated with forecast inputs.
- The write-up should explain that realised fundamentals are not used as live trading inputs unless clearly labelled as a perfect-foresight or structural limitation.

## 2026-05-28 16:46: Data Storage Layout

**Question / topic:**  
How should the project store raw, interim, and processed data?

**Answer / decision:**  
Use a layered data layout:

```text
data/
  raw/
    entsoe/
      day_ahead_prices_202401010000_202501010000.xml
    energy_charts/
      price_2024-01-01_2025-01-01.json

  interim/
    entsoe_day_ahead_prices_hourly.csv
    entsoe_load_forecast_hourly.csv
    entsoe_solar_forecast_hourly.csv
    entsoe_wind_forecast_hourly.csv

  processed/
    germany_hourly_model_dataset.csv
```

Raw API responses should be preserved in their native format: XML for ENTSO-E and JSON for Energy-Charts. Parsed hourly source tables should be stored as CSV in `data/interim/`. The final joined modelling table should be stored as CSV in `data/processed/`.

**Follow-up notes:**  
The canonical timestamp should be UTC, with local German market time stored as an additional column for reporting and DST checks. Generated data files stay out of git; only directory placeholders should be committed.

## 2026-05-29 17:32: Modelling Validation Design

**Question / topic:**  
How should the baseline model, rolling validation, hyperparameter tuning, and final holdout test be structured?

**Answer / decision:**  
The next stage should introduce a new modelling helper area, for example:

```text
pipeline_helpers/
  modelling/
    constants.py
    metrics.py
    validation.py
    baseline_week_lag.py
```

Each model should expose the same simple interface:

```python
PARAM_GRID = [...]

def train(train_data, params):
    ...

def predict(model_state, test_data, params):
    ...
```

The validation code should not know model-specific hyperparameters. Instead, each model declares its own `PARAM_GRID`, and validation loops over those parameter settings. This lets the same validation engine work for the weekly baseline, LEAR-style regression, and gradient boosting.

The baseline model should be implemented first as a seasonal naive forecast, mainly `price_lag_168`, meaning the prediction for an hour is the price from the same hour one week earlier. Its `train` function can return no fitted state, but it should still exist to keep the interface consistent.

**Validation purpose:**  
Rolling validation is used to choose model settings and compare model families. It answers whether a model works repeatedly across different historical market periods, not just in one lucky train/test split.

**Final holdout decision:**  
The final holdout should not be hardcoded by date. Instead, validation should infer it from the data:

- the last `TEST_MONTHS` of the available dataset are reserved as the untouched final holdout;
- rolling validation/tuning must only use data before that final holdout;
- final testing trains on the previous `TRAIN_MONTHS` immediately before the holdout and tests on the final `TEST_MONTHS`.

Use one consistent forecasting task:

```python
TRAIN_MONTHS = 24
TEST_MONTHS = 1
STEP_MONTHS = 3
```

So validation repeatedly does:

```text
train previous 24 months -> test next 1 month -> step forward 3 months
```

And final holdout does:

```text
train previous 24 months before the final month -> test the final month
```

**Follow-up notes:**  
This keeps validation and final testing aligned with the operational goal: forecast the near future, specifically the next month, from a fixed recent training window. Longer final holdout tests were considered, but rejected for now because they would answer a slightly different question than the rolling validation setup.
