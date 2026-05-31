# European Power Fair Value Case Study

Prototype for forecasting German/Luxembourg hourly day-ahead power prices and translating forecasts into prompt-curve fair-value views.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the environment template and fill local secrets:

```bash
cp .env.example .env
```

Required or optional variables:

- `ENTSOE_API_KEY`: ENTSO-E Transparency Platform security token.
- `OPENAI_API_KEY`: optional OpenAI API key for AI commentary.
- `OPENAI_MODEL`: optional model name for commentary generation.

Do not commit `.env`.

## Pipeline Order

Build data from ENTSO-E:

```bash
.venv/bin/python pipeline_steps/build_dataset.py \
  --datasets day_ahead_prices load_forecast solar_forecast wind_onshore_forecast wind_offshore_forecast \
  --start 01-01-2021 \
  --end 02-01-2026 \
  --mode modelling
```

Validate a baseline:

```bash
.venv/bin/python pipeline_steps/validate_model.py \
  --features data/processed/germany_modelling_2021_2026/germany_model_features.csv \
  --model baseline_week_lag
```

Validate the selected LEAR-style model:

```bash
.venv/bin/python pipeline_steps/validate_model.py \
  --features data/processed/germany_modelling_2021_2026/germany_model_features.csv \
  --model lear_model \
  --regularization lasso \
  --target-transform raw \
  --target-option A \
  --feature-mode day_ahead_full
```

Predict one delivery day as a 24-hour vector:

```bash
.venv/bin/python pipeline_steps/predict_day_ahead.py \
  --features data/processed/germany_modelling_2021_2026/germany_model_features.csv \
  --delivery-day 01-12-2025 \
  --model lear_model \
  --regularization lasso \
  --target-transform raw
```

Validate a direct period-average model, Option B:

```bash
.venv/bin/python pipeline_steps/validate_model.py \
  --features data/processed/germany_modelling_2021_2026/germany_model_features.csv \
  --model lear_model \
  --regularization lasso \
  --target-transform raw \
  --target-option B \
  --period-days 30 \
  --block baseload
```

Translate a forecast period into curve views:

```bash
.venv/bin/python pipeline_steps/translate_curve_view.py \
  --features data/processed/germany_modelling_2021_2026/germany_model_features.csv \
  --start 01-11-2025 \
  --end 01-12-2025 \
  --model lear_model \
  --regularization lasso \
  --target-transform raw \
  --target-option A \
  --feature-mode period_hourly_safe \
  --block all \
  --benchmark trailing_average
```

Generate AI commentary for one curve view:

```bash
.venv/bin/python pipeline_steps/generate_ai_commentary.py \
  --summary outputs/curve_translation/germany_modelling_2021_2026/<model-run>/20251101_20251201/baseload/curve_view_summary.csv
```

## Main Outputs

- `data/processed/.../germany_model_dataset.csv`: clean hourly modelling dataset.
- `data/processed/.../germany_model_features.csv`: feature table.
- `data/processed/.../data_qa_report.md`: ingestion and data QA report.
- `models/<dataset>/<model-run>/`: trained model artifacts and validation outputs.
- `outputs/curve_translation/...`: prompt-curve reports and AI commentary logs.
- `validation_summary.csv`: rolling validation averages by parameter setting.
- `final_holdout_metrics.csv`: final holdout performance.
- `*_prediction_diagnostics.csv`: coverage diagnostics.
- `*_monthly_metrics.csv`, `*_yearly_metrics.csv`: time breakdowns.
- `curve_view_report.md`: prompt-curve fair-value view.
- `ai_commentary.md`: optional AI-generated commentary or deterministic fallback.

## Notes

- UTC is the canonical timestamp and join key.
- Modelling includes German local calendar features while full daily lag curves remain UTC-based.
- Curve peakload/offpeak blocks use German local market time.
- The default benchmark is a proxy, not a traded forward curve. A manual curve price can be supplied when available.
- API keys are loaded from environment variables only.
