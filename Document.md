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
