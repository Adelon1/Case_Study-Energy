# European Power Fair Value — German Day-Ahead Forecasting

A reproducible prototype that forecasts **German/Luxembourg hourly day-ahead
electricity prices** from public ENTSO-E fundamentals, attaches calibrated
uncertainty bands, and translates the result into **prompt-curve trading views**
(baseload / peakload / offpeak / peak–base spread) with a long / neutral / short
signal, desk action, and invalidation logic. A single programmatic LLM step generates
grounded commentary.

The default workflow is a **daily rolling day-ahead forecast**: each delivery day is
forecast from information known *before* the auction (forecast load/wind/solar, recent
daily price-curve shapes, calendar). It is not a one-shot
forecast of future realised prices.

The same code also supports a second target on request — a **period-average forecast**,
where a single value is predicted for a whole delivery period of length `|P|` days
(e.g. a month-ahead baseload or peakload average) instead of 24 hourly values. The two
targets are selected with `--target hourly` (default) or `--target period_average`.

> **Documentation**
> - [`TASK.md`](TASK.md) — the original case-study brief.
> - [`REPORT.md`](REPORT.md) — full report: design decisions, literature, code walk-through, results.

## Features

- ENTSO-E ingestion with raw → interim → processed data layout and a generated QA report.
- Correct UTC/DST handling (UTC join key, local German calendar features, 24-column daily lags across 23/25-hour days).
- Leakage-safe feature engineering with previous UTC daily price curves and local German calendar features.
- Three models behind one interface: seasonal-lag baseline, **LEAR-style 24-hour regularised ARX** (headline), and a histogram gradient-boosting benchmark.
- Rolling-origin validation (24m train / 1m test / 1m step, 35 folds) with MAE, RMSE, bias, and tail/scarcity metrics.
- Empirical **P10–P90 residual bands** per hour (validation coverage ≈ 80%).
- Band-driven curve signal with benchmark comparison, desk action, and invalidation rules.
- One LLM commentary step (OpenAI Responses API) with prompt/output/failure logging and a deterministic fallback.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in secrets
```

Environment variables (loaded from `.env`, never committed):

| Variable | Required | Purpose |
| --- | --- | --- |
| `ENTSOE_API_KEY` | yes | ENTSO-E Transparency Platform token ([how to get one](https://transparencyplatform.zendesk.com/hc/en-us/articles/12845911031188-How-to-get-security-token)) |
| `OPENAI_API_KEY` | optional | enables real LLM commentary (falls back to deterministic text otherwise) |
| `OPENAI_MODEL` | optional | commentary model name (default `gpt-4.1-mini`) |

## Quick start

```bash
# 1. Build the dataset (download, parse, combine, QA, features)
.venv/bin/python pipeline_steps/01_build_dataset.py \
  --datasets day_ahead_prices load_forecast solar_forecast wind_onshore_forecast wind_offshore_forecast \
  --start 01-01-2021 --end 02-01-2026 --mode modelling

# 2. Validate the baseline
.venv/bin/python pipeline_steps/02_validate_model.py \
  --features data/processed/germany_modelling_2021_2026/germany_model_features.csv \
  --model baseline_week_lag

# 3. Validate the headline LEAR model
.venv/bin/python pipeline_steps/02_validate_model.py \
  --features data/processed/germany_modelling_2021_2026/germany_model_features.csv \
  --model lear_model --regularization lasso --target-transform raw \
  --target hourly --feature-mode day_ahead_full

# 4. Generate validation figures
.venv/bin/python pipeline_steps/03_plot_validation.py \
  --run-folder models/germany_modelling_2021_2026/lear_model_lasso_raw__hourly__day_ahead_full

# 5. Translate a period into curve views
.venv/bin/python pipeline_steps/05_translate_curve_view.py \
  --features data/processed/germany_modelling_2021_2026/germany_model_features.csv \
  --start 01-11-2025 --end 01-12-2025 \
  --model lear_model --regularization lasso --target-transform raw \
  --target hourly --feature-mode period_hourly_safe --block all --benchmark trailing_average

# 6. (optional) AI commentary for one curve view
.venv/bin/python pipeline_steps/06_generate_ai_commentary.py \
  --summary outputs/curve_translation/germany_modelling_2021_2026/<run>/20251101_20251201/baseload/curve_view_summary.csv
```

Forecast a single delivery day as a 24-hour vector:

```bash
.venv/bin/python pipeline_steps/04_predict_day_ahead.py \
  --features data/processed/germany_modelling_2021_2026/germany_model_features.csv \
  --delivery-day 01-12-2025 \
  --model lear_model --regularization lasso --target-transform raw
```

## Repository layout

```text
pipeline_steps/                  # runnable entry points, numbered in run order
  01_build_dataset.py            # download → parse → combine → QA → features
  02_validate_model.py           # rolling validation, metrics, bands, artifacts
  03_plot_validation.py          # forecast-vs-actual, MAE-by-hour, residual figures
  04_predict_day_ahead.py        # one delivery day as a 24h vector
  05_translate_curve_view.py     # hourly forecast → curve views + signal
  06_generate_ai_commentary.py   # LLM commentary with logging + fallback

pipeline_helpers/
  entsoe_data/                   # ingestion, parsing, QA, feature engineering
  modelling/                     # models, validation, metrics, prediction bands
  curve_translation/             # blocks, benchmarks, band-driven signal

data/        raw/ interim/ processed/   (generated; not tracked)
models/      validation runs and saved artifacts per model
outputs/     figures/ and curve_translation/ reports
```

## Outputs

| Path | Contents |
| --- | --- |
| `data/processed/.../germany_model_dataset.csv` | clean hourly modelling dataset |
| `data/processed/.../germany_model_features.csv` | feature table |
| `data/processed/.../data_qa_report.md` | ingestion + data QA report |
| `models/<dataset>/<run>/validation_summary.csv` | averaged metrics per parameter set |
| `models/<dataset>/<run>/validation_metrics.csv` | one row per parameter × fold |
| `models/<dataset>/<run>/predictions.csv` | out-of-sample predictions with `y_pred_lower/upper` bands |
| `models/<dataset>/<run>/model.joblib`, `metadata.json` | saved model + run metadata |
| `outputs/figures/<run>/*.png` | validation figures |
| `outputs/curve_translation/.../curve_view_report.md` | prompt-curve fair-value view + signal |
| `outputs/curve_translation/.../ai_commentary.md` | optional LLM (or fallback) commentary |

## Headline result

LEAR (LASSO, raw target), 35 monthly folds, Feb-2023 → Dec-2025:

| MAE | RMSE | Bias | Top-decile MAE | P10–P90 band coverage |
| --- | --- | --- | --- | --- |
| 17.30 | 25.56 | +0.41 | 30.34 | 79.9% |

(€/MWh; coverage target is 80%.)

## Notes & conventions

- **UTC** is the canonical timestamp and join key; German local time is used only for calendar features and peak/offpeak blocks.
- Only **load, solar, and wind** fundamentals are used — no fuel, carbon, or flows (by design).
- The benchmark is a realised-price **proxy**, not a traded forward curve; a manual curve price can be supplied.
- Secrets are read from environment variables only; `.env` is never committed.

## License / data

Data is sourced from the public [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/). Respect ENTSO-E's terms of use when redistributing data.
