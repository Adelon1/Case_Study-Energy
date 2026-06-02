# Data Ingestion and QA Report

## Dataset Run

- Dataset folder: `germany_modelling_2024_2026`
- Stage-3 CSV: `data/03_processed/germany_modelling_2024_2026/germany_model_dataset.csv`
- Feature CSV: `data/03_processed/germany_modelling_2024_2026/germany_model_features.csv`
- Data source: ENTSO-E Transparency Platform REST API
- API endpoint: `https://web-api.tp.entsoe.eu/api`
- Market: Germany/Luxembourg bidding zone (`DE-LU`)
- Timezone convention: ENTSO-E XML timestamps are parsed as UTC. `timestamp_utc` is the canonical join key. `timestamp_local` is derived from UTC using `Europe/Berlin` for reporting and delivery-period interpretation.
- Requested local delivery window: `2024-05-20 00:00:00+02:00` to `2026-06-05 00:00:00+02:00`; end is exclusive.
- API UTC window: `202405192200` to `202606042200`
- Final timestamp UTC range: `2024-05-19 22:00:00+00:00 to 2026-06-03 21:00:00+00:00`
- Final timestamp local range: `2024-05-20 00:00:00+02:00 to 2026-06-03 23:00:00+02:00`

## Included Data

| Dataset name | Dataset description | Final column | ENTSO-E request params |
| --- | --- | --- | --- |
| day_ahead_prices | Sequence 1 SDAC day-ahead electricity prices for DE-LU. | day_ahead_price_eur_per_mwh | {'documentType': 'A44', 'in_Domain': '10Y1001A1001A82H', 'out_Domain': '10Y1001A1001A82H', 'contract_MarketAgreement.type': 'A01'} |
| load_forecast | Day-ahead total load forecast for DE-LU. | load_forecast_mw | {'documentType': 'A65', 'processType': 'A01', 'outBiddingZone_Domain': '10Y1001A1001A82H'} |
| solar_forecast | Day-ahead solar generation forecast for DE-LU. | solar_forecast_mw | {'documentType': 'A69', 'processType': 'A01', 'in_Domain': '10Y1001A1001A82H', 'psrType': 'B16'} |
| wind_offshore_forecast | Day-ahead offshore wind generation forecast for DE-LU. | wind_offshore_forecast_mw | {'documentType': 'A69', 'processType': 'A01', 'in_Domain': '10Y1001A1001A82H', 'psrType': 'B18'} |
| wind_onshore_forecast | Day-ahead onshore wind generation forecast for DE-LU. | wind_onshore_forecast_mw | {'documentType': 'A69', 'processType': 'A01', 'in_Domain': '10Y1001A1001A82H', 'psrType': 'B19'} |

## Frequency and Coverage

| Dataset | Parsed input frequency before assembly |
| --- | --- |
| day_ahead_prices | 0 days 00:15:00 |
| load_forecast | 0 days 00:15:00 |
| solar_forecast | 0 days 00:15:00 |
| wind_offshore_forecast | 0 days 00:15:00 |
| wind_onshore_forecast | 0 days 00:15:00 |

- Final assembled frequency: hourly mean.
- Expected hourly rows: `17904`
- Actual hourly rows: `17880`
- Coverage: `99.87%`
- Duplicate `timestamp_utc` rows: `0`
- UTC timestamps monotonic increasing: `True`

## Missing Data

| Column | Missing values |
| --- | --- |
| day_ahead_price_eur_per_mwh | 0 |
| load_forecast_mw | 0 |
| solar_forecast_mw | 0 |
| wind_offshore_forecast_mw | 0 |
| wind_onshore_forecast_mw | 0 |

## Leakage-Safe Imputation

- Applied after hourly assembly and before feature generation.
- Columns considered: `['load_forecast_mw', 'solar_forecast_mw', 'wind_onshore_forecast_mw', 'wind_offshore_forecast_mw']`
- Fill rule: a missing value at time `t` is filled from the same column at `t - 24h`.
- The pipeline does not use future values, interpolation across future points, or backfill.
- Rows dropped after the fill because forecast driver values were still missing: `0`

| Column | Missing before fill | Filled from t-24h | Missing after fill |
| --- | --- | --- | --- |
| load_forecast_mw | 1 | 1 | 0 |
| solar_forecast_mw | 2 | 2 | 0 |
| wind_onshore_forecast_mw | 2 | 2 | 0 |
| wind_offshore_forecast_mw | 0 | 0 | 0 |

## Obvious Outlier Checks

Outlier checks are QA flags only; the pipeline does not remove values automatically.

Rules:

- `day_ahead_price_eur_per_mwh`: below -500 or above 1000
- `load_forecast_mw`: <= 0 or above 120000
- `solar_forecast_mw`, `wind_onshore_forecast_mw`: below 0 or above 90000
- `wind_offshore_forecast_mw`: below 0 or above 30000

| Column | Obvious outlier count |
| --- | --- |
| day_ahead_price_eur_per_mwh | 0 |
| load_forecast_mw | 0 |
| solar_forecast_mw | 0 |
| wind_offshore_forecast_mw | 0 |
| wind_onshore_forecast_mw | 0 |

## Feature Table Inventory

| Feature inventory item | Count |
| --- | --- |
| Total feature table columns | 135 |
| Timestamp columns | 2 |
| Target columns | 1 |
| Candidate feature columns | 132 |
| Fundamental/source/derived columns | 10 |
| Calendar columns | 20 |
| Daily price curve columns | 96 |
| Daily price summary columns | 6 |

Feature groups:

- Passthrough ENTSO-E columns: `['day_ahead_price_eur_per_mwh', 'load_forecast_mw', 'solar_forecast_mw', 'wind_offshore_forecast_mw', 'wind_onshore_forecast_mw']`
- Fundamental derived columns: `['load_forecast_mw', 'renewable_share_of_load', 'renewable_total_forecast_mw', 'residual_load_forecast_mw', 'solar_forecast_mw', 'solar_share_of_load', 'wind_offshore_forecast_mw', 'wind_onshore_forecast_mw', 'wind_share_of_load', 'wind_total_forecast_mw']`
- Local calendar columns: `['is_holiday', 'local_day_of_year', 'local_day_of_year_cos', 'local_day_of_year_sin', 'local_hour', 'local_hour_cos', 'local_hour_sin', 'local_month', 'local_month_cos', 'local_month_sin', 'local_weekday', 'local_weekday_0', 'local_weekday_1', 'local_weekday_2', 'local_weekday_3', 'local_weekday_4', 'local_weekday_5', 'local_weekday_6', 'local_weekday_cos', 'local_weekday_sin']`
- Daily price summary columns: `['price_d1_max', 'price_d1_mean', 'price_d1_min', 'price_d7_max', 'price_d7_mean', 'price_d7_min']`

Daily price curve lag columns:

| Lag group | Column count | Hour columns |
| --- | --- | --- |
| previous 1 day(s) | 24 | h00...h23 |
| previous 2 day(s) | 24 | h00...h23 |
| previous 3 day(s) | 24 | h00...h23 |
| previous 7 day(s) | 24 | h00...h23 |

## Timestamp Alignment

- Datasets are joined by `timestamp_utc`.
- Any timestamp alignment problem appears as missing values after the join.
- Local timestamps are derived after joining and hourly aggregation, not used as the primary key.

## DST Handling

The pipeline does not join on local clock time. ENTSO-E XML timestamps are UTC, which is unique across daylight-saving-time transitions. The requested delivery dates are provided as German local dates (`DD-MM-YYYY`) and converted once to UTC using pandas/zoneinfo timezone rules. This means normal days, 23-hour spring DST days, and 25-hour autumn DST days produce the correct UTC window length.

## Known Limitations

- The report uses simple rule-based outlier checks; it does not classify market-valid scarcity or negative-price events as errors unless they exceed the stated thresholds.
- ENTSO-E may return full market documents that overlap the request window. The final combined dataset is filtered to the requested local delivery window after parsing.
- Final hourly values are arithmetic means of the parsed ENTSO-E resolution. This is appropriate for the current price and MW forecast series, but should be reviewed if future datasets represent totals rather than average levels.
- API availability, revisions, and publication timing are controlled by ENTSO-E and TSOs.
