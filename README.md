# European Power Fair Value — German Day-Ahead Forecasting

A reproducible prototype that forecasts **German/Luxembourg hourly day-ahead
electricity prices** from public ENTSO-E fundamentals, attaches calibrated
uncertainty bands, and translates the result into **prompt-curve trading views**
(baseload / peakload / offpeak) with a long / neutral / short
signal, desk action, and invalidation logic. A single programmatic LLM step generates
grounded commentary.

The default workflow is a **daily rolling day-ahead forecast**: each delivery day is
forecast from information known before the auction (forecast load/wind/solar, recent
daily price-curve shapes, calendar). It is not a one-shot forecast of future realised
prices.

The same code also supports period-oriented setups: `hourly_period` predicts hourly
prices over a longer window with price-history features removed, while `period_average`
predicts baseload, peakload, and offpeak averages for a whole delivery period. The
setup is selected inside the interactive validation and forecast-view prompts.

> **Documentation**
> - [`TASK.md`](TASK.md) — the original case-study brief.
> - [`submission_report.tex`](submission_report.tex) / `submission_report.pdf` — submission report.
> - [`REPORT.md`](REPORT.md) — long-form internal report: design decisions, literature, code walk-through, results.

## Features

- ENTSO-E ingestion with 01_raw → 02_interim → 03_processed data layout and a generated QA report.
- Correct UTC/DST handling (UTC join key, German-local calendar features, UTC daily lag curves across 23/25-hour local days).
- Leakage-safe feature engineering with previous UTC daily price curves and local German calendar features.
- Five models behind one interface: row-lag baseline, **LEAR-style 24-hour regularised ARX**, histogram gradient boosting, Theil-Sen, and RANSAC-LASSO robust linear checks.
- Rolling-origin validation (24m train / 1m test / 1m step) with MAE, RMSE, bias, and tail/scarcity metrics.
- Empirical **P10-P90 residual bands** from validation residuals (coverage ≈ 80%).
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

## Workflow

All runnable scripts are interactive by default; run them without command-line options and answer the prompts.

### Recommended Workflow

| Step | Command | Use |
| --- | --- | --- |
| 1 | `.venv/bin/python pipeline_steps/01_build_dataset.py` | Build raw/interim/processed data, QA report, and feature table. |
| 2 | `.venv/bin/python pipeline_steps/02_validate_model.py` | Validate one model/setup and save best parameters, metrics, bands, and artifacts. |
| 3 | `.venv/bin/python pipeline_steps/03_run_forecast_view.py` | Train/predict one requested delivery period and create curve view, plots, report, and optional AI commentary. |

### Possible Workflows

| Workflow | What to run | When to use |
| --- | --- | --- |
| Full rebuild | `01_build_dataset.py` → `02_validate_model.py` → `03_run_forecast_view.py` | Use after changing data dates, feature engineering, or model logic. |
| Model comparison | Run `02_validate_model.py` multiple times | Use to compare baseline, LEAR, boosted trees, Theil-Sen, or RANSAC-LASSO. |
| Trading view only | Run `03_run_forecast_view.py` | Use after validation already produced a model folder with best parameters. |
| AI commentary only | Enable AI in `03_run_forecast_view.py` | Use when you want a logged commentary from computed curve-view numbers. |

### Forecast Setups

| Setup | Meaning | Use |
| --- | --- | --- |
| `hourly_day_ahead` | Predict one delivery day as 24 hourly prices. | Recommended for next-day DA forecasting. |
| `hourly_period` | Predict hourly prices over a longer period without price-lag features. | Use for multi-day period views from fundamentals. |
| `period_average` | Predict baseload, peakload, and offpeak averages for each delivery period. | Use for direct multi-block period forecasts; all three blocks are built together. |

### Main Files

| File | Role |
| --- | --- |
| `pipeline_steps/01_build_dataset.py` | Main data-building entry point. |
| `pipeline_steps/02_validate_model.py` | Main model-validation entry point. |
| `pipeline_steps/03_run_forecast_view.py` | Main Task-3 forecast-to-curve workflow. |
| `pipeline_helpers/02_modelling/feature_sets.json` | User-editable mapping from model/setup to feature bundles. |
| `.env` | User-editable secrets and API model settings. |
| `REPORT.md` | Long-form explanation of methodology, implementation details, and results. |

### What A Normal User May Change

| File / Prompt | What to change |
| --- | --- |
| `.env` | Add `ENTSOE_API_KEY`, optional `OPENAI_API_KEY`, and optional `OPENAI_MODEL`. |
| `01_build_dataset.py` prompts | Change data mode, datasets, and start/end dates. |
| `02_validate_model.py` prompts | Change model, forecast setup, regularization, and period length. |
| `03_run_forecast_view.py` prompts | Change delivery day/window, training months, benchmark, and AI commentary. |
| `pipeline_helpers/02_modelling/feature_sets.json` | Add/remove feature bundles or assign feature sets to models. |
| `pipeline_helpers/02_modelling/00_constants.py` | Change validation windows, metric thresholds, or model grids. |

### What A Normal User Should Not Change

| File / Folder | Why not |
| --- | --- |
| `pipeline_helpers/01_entsoe_data/03_entsoe_api.py` | API mechanics should stay stable unless ENTSO-E changes. |
| `pipeline_helpers/01_entsoe_data/04_entsoe_xml_to_csv.py` | XML parsing is low-level and easy to break. |
| `pipeline_helpers/02_modelling/09_validation.py` | Validation engine is shared by every model. |
| `pipeline_helpers/02_modelling/10_window_prediction.py` | Window prediction is shared by Task-3 workflows. |
| `pipeline_helpers/03_curve_translation/*.py` | Curve translation logic should change only when the trading methodology changes. |
| `data/02_interim`, `data/03_processed`, `models/`, `outputs/` | Submission artifacts; inspect them, but regenerate rather than editing by hand. |

### Model Names

| Model | Use |
| --- | --- |
| `baseline_model` | Naive benchmark that must be beaten. |
| `lear_model` | Main LEAR-style regularised linear model. |
| `boosted_tree_model` | Nonlinear improved model. |
| `theil_sen_model` | Robust linear check on compact features. |
| `ransac_lasso_model` | Robust sparse linear check. |

## Repository layout

```text
pipeline_steps/                  # runnable entry points, numbered in run order
  01_build_dataset.py            # download → parse → combine → QA → features
  02_validate_model.py           # rolling validation, metrics, bands, artifacts
  03_run_forecast_view.py        # train/predict window → curve view → plots → optional AI

pipeline_helpers/
  01_entsoe_data/                   # ingestion, parsing, QA, feature engineering
  02_modelling/                     # models, validation, metrics, prediction bands
  03_curve_translation/             # blocks, benchmarks, signal, AI commentary helper

data/        01_raw/ 02_interim/ 03_processed/   (raw is ignored; interim/processed are submission artifacts)
models/      validation runs and saved artifacts per model
outputs/     human-facing forecast views, plots, reports, and commentary
```

## Outputs

| Path | Contents |
| --- | --- |
| `data/03_processed/.../germany_model_dataset.csv` | clean hourly modelling dataset |
| `data/03_processed/.../germany_model_features.csv` | feature table |
| `data/03_processed/.../data_qa_report.md` | ingestion + data QA report |
| `models/<dataset>/<run>/validation_summary.csv` | averaged metrics per parameter set |
| `models/<dataset>/<run>/validation_metrics.csv` | one row per parameter × fold |
| `models/<dataset>/<run>/predictions.csv` | out-of-sample predictions with `y_pred_lower/upper` bands |
| `models/<dataset>/<run>/model.joblib`, `metadata.json` | saved model + run metadata |
| `outputs/<dataset>/<period>/<model_setup>/predictions.csv` | one chosen train/predict window |
| `outputs/<dataset>/<period>/<model_setup>/curve_view_summary.csv` | fair value, benchmark, edge, signal |
| `outputs/<dataset>/<period>/<model_setup>/curve_view_report.md` | prompt-curve fair-value report |
| `outputs/<dataset>/<period>/<model_setup>/plots/*.png` | forecast/band, fair-value, signal, and optional heatmap figures |
| `outputs/<dataset>/<period>/<model_setup>/curve_translation/<block>/ai_commentary.md` | optional LLM or fallback commentary |

## Headline result

Selected validation results on `germany_modelling_2021_2026`:

| Run | MAE | RMSE | Bias | P10-P90 coverage |
| --- | ---: | ---: | ---: | ---: |
| `boosted_tree_model__hourly_day_ahead` | 15.28 | 23.94 | +2.17 | 79.9% |
| `ransac_lasso_model_raw__hourly_day_ahead` | 15.58 | 24.22 | -0.96 | 79.9% |
| `lear_model_elasticnet_raw__hourly_day_ahead` | 16.23 | 26.35 | +1.75 | 79.9% |
| `boosted_tree_model__hourly_period` | 28.30 | 40.14 | +16.06 | 79.9% |
| `theil_sen_model_raw__hourly_period` | 29.85 | 39.57 | +17.42 | 79.9% |
| `baseline_model__hourly_period` | 61.86 | 74.83 | +42.33 | 79.9% |
| `lear_model_lasso_raw__period_average__15d` | 17.13 | 19.80 | +5.47 | 79.7% |
| `baseline_model__period_average__15d` | 18.16 | 21.25 | +0.40 | 79.7% |
| `baseline_model__hourly_day_ahead` | 26.64 | 39.53 | +0.03 | 79.9% |

(€/MWh; coverage target is 80%.)

## Notes & conventions

- **UTC** is the canonical timestamp and join key; German local time is used only for calendar features and peak/offpeak blocks.
- Only **load, solar, and wind** fundamentals are used — no fuel, carbon, or flows (by design).
- The benchmark is a realised-price **proxy**, not a traded forward curve; a manual curve price can be supplied.
- `data/01_raw/` is ignored because raw API downloads are reproducible and bulky; processed QA/results artifacts are intended for submission.
- Secrets are read from environment variables only; `.env` is never committed.

## License / data

Data is sourced from the public [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/). Respect ENTSO-E's terms of use when redistributing data.
