# European Power Fair Value — Full Project Report

Rawad Batous  
rawad.batous2006@gmail.com

> This is the long-form internal report. It is deliberately more detailed than the
> 1–3 page submission write-up. It records *what* was built, *how* the code works,
> and *why* every important decision was made — including the literature each idea
> came from. It absorbs and supersedes the earlier discussion logs (`Document.md`
> and `session.md`), so it is meant to be read as a single coherent story rather
> than a set of disconnected notes.

---

## 1. The one-sentence version

The project is a **daily rolling German day-ahead electricity price forecasting
prototype** that turns public ENTSO-E fundamentals into hourly price forecasts,
attaches an empirical uncertainty band to those forecasts, aggregates them into
prompt-curve blocks (baseload, peakload, offpeak, peak/base spread), and converts
the result into a long / neutral / short trading view with a desk action, an
uncertainty-aware signal, and explicit invalidation logic — with one programmatic
LLM commentary step bolted on top.

Everything below is an unfolding of that sentence. The red line that connects all
sections is a single question that dominated the design: **what information is the
model actually allowed to know at the moment it forecasts a delivery day, and how do
we stay honest about it?** That question drives the market choice, the data layout,
the feature engineering, the model family, the validation design, the band logic,
and the trading translation.

---

## 2. Choosing the market: why Germany

The case study allows one European market (DE, FR, NL, GB). Germany was chosen
because it gives the richest market story for the smallest data effort: very high
wind and solar penetration, a volatile post-2021 price regime, frequent negative
prices, and several overlapping public data sources that can cross-check each other.

The data sources reviewed were:

- **ENTSO-E Transparency Platform** — official pan-European transparency data.
- **Open Power System Data** — fast historical bootstrap.
- **SMARD / Bundesnetzagentur** — German regulator data and figures.
- **Fraunhofer Energy-Charts API** — convenient cross-check / fallback.

The final hierarchy is: **ENTSO-E as the primary source**, Energy-Charts as a
fallback/cross-check, SMARD for German-specific documentation, and Open Power System
Data as a historical bootstrap option. ENTSO-E was chosen as primary because it is
the official public European source and, crucially, it publishes **leakage-safe
day-ahead forecasts** of load, wind, and solar — exactly the ex-ante information a
trader has before the auction. The German market is also the one the electricity
price literature studies most heavily, so the modelling choices below can lean on
papers that used the same market and the same ENTSO-E inputs.

A hard scope rule was fixed at this point and kept throughout: **the only
fundamentals used are load, solar, and wind. No fuel prices, no carbon, no
cross-border flows.** This keeps the project auditable and the feature set honest.

---

## 3. The question that shaped the whole project: actual inputs vs forecast inputs

Before writing any model, one methodological question had to be settled, because it
decides what "good performance" even means.

There are two very different models you can build:

1. Train on **actual realised** load/wind/solar. Clean, unbiased inputs. But you
   never have the realised values before the auction, so this is not tradable.
2. Train on **forecasted** load/wind/solar — the values published *before* the
   auction. Noisier, biased, but this is the real information set a desk trades on.

It turns out this is a 30-year-old debate in meteorological post-processing, where it
is called **Perfect Prog vs MOS**. The foundational reference is Marzban, Sandgathe &
Kalnay, *MOS, Perfect Prog, and Reanalysis*
([pdf](https://faculty.washington.edu/marzban/mos.pdf)). Mapping their framework onto
this project:

- training on **actual** fundamentals = **Perfect Prog** (structural / oracle);
- training on **forecast** fundamentals = **MOS** (tradable / ex-ante).

Their conclusion — and the reason this matters — is that there is no universal winner.
MOS learns the forecast's own bias, but ties you to one forecast provider; Perfect
Prog is simpler but suffers when live forecast error differs from training conditions.
The empirical follow-ups confirm both sides: Brunet, Verret & Yacowar
([1988](https://journals.ametsoc.org/view/journals/wefo/3/4/1520-0434_1988_003_0273_aocomo_2_0_co_2.xml))
found Perfect Prog competitive at short range, while Wilson & Vallée
([2003](https://journals.ametsoc.org/view/journals/wefo/18/2/1520-0434_2003_018_0288_tcumos_2_0_co_2.xml))
showed an *updateable* MOS can beat both — which matters here because the validation
design retrains every month.

The same tension shows up directly in energy load forecasting. Fay & Ringwood
([2010](https://ee.maynoothuniversity.ie/jringwood/Respubs/J156DFPS.pdf)) argue you can
reasonably train on actual weather (forecast archives are noisy and weather models
keep improving) — but they warn that live forecast error, unseen in training, can hurt
disproportionately. Wang et al.
([2020](https://www.sciencedirect.com/science/article/am/pii/S0306261920301951)) show
empirically that "train on actuals, test on forecasts" degrades some models badly,
which is a concrete caution against the naive Perfect Prog route. Runge & Saloux
([2023](https://www.sciencedirect.com/science/article/abs/pii/S0360544223000555)) give
the clean vocabulary the report uses: *prediction* uses contemporaneous actual inputs,
*forecasting* uses future forecast inputs — they are not the same task. Fildes, Randall
& Stubbs ([1997](https://www.jstor.org/stable/3009939)) add the evaluation rule that
ex-ante performance must account for the fact that the explanatory variables are
themselves forecasts.

For electricity prices specifically, the literature has effectively already settled
the question in favour of the tradable, forecast-input model:

- Maciejowska, Nitka & Weron
  ([2021](https://www.sciencedirect.com/science/article/abs/pii/S014098832100178X))
  — the most directly relevant paper — show that TSO load/wind/solar forecasts are
  biased, that serious price papers nonetheless use *forecasted* fundamentals, and
  that improving those fundamentals improves price accuracy.
- Uniejewski & Ziel
  ([2025](https://arxiv.org/abs/2501.06180)) push this further with probabilistic
  fundamental forecasts.
- Kulakov & Ziel ([2019](https://arxiv.org/abs/1903.09641)) and Goodarzi, Perera &
  Bunn ([2019](https://www.sciencedirect.com/science/article/abs/pii/S0301421519304057))
  show renewable forecast errors have real, economically meaningful price impact — so
  the gap between an oracle and a tradable model is not a rounding error.
- Beran, Vogler & Weber
  ([2021](https://ideas.repec.org/p/dui/wpaper/2102.html)) emphasise respecting the
  information cutoff in German multi-day-ahead forecasting — which is exactly the
  leakage discipline applied below.

**The resolution adopted here:** the *primary* model is the tradable, forecast-input
(MOS-style) model. An actual-input model is only ever legitimate as a labelled
oracle / perfect-foresight benchmark, never as the headline result. The performance
gap between them is itself the interesting output — it quantifies the cost of not
knowing the true future fundamentals. The implemented prototype focuses on the
tradable model; the oracle is documented as the natural next comparison.

This single decision is the spine of the rest of the report.

---

## 4. Data ingestion: how ENTSO-E XML becomes a clean hourly table

The ingestion code lives in `pipeline_helpers/01_entsoe_data/` and is driven by
`pipeline_steps/01_build_dataset.py`. It implements the layered data design that keeps
stage-1 responses, parsed tables, and the modelling table separate:

```text
data/
  01_raw/         # native API responses, preserved untouched (ENTSO-E XML)
  02_interim/     # one parsed CSV per dataset
  03_processed/   # combined hourly table + feature table + QA report
```

The pipeline runs end to end as:

1. **Parse CLI arguments** — which datasets, the German local start/end dates, mode,
   and the `.env` path (`01_build_dataset.py`).
2. **Convert local dates to UTC API windows** (`01_date_windows.py`). The user thinks in
   German delivery dates; ENTSO-E wants UTC windows.
3. **Split into monthly chunks** so each download stays small and resumable.
4. **Download XML** per dataset and chunk (`03_entsoe_api.py`), reading the token from
   `ENTSOE_API_KEY` in the environment and saving XML under `data/01_raw/`.
5. **Parse XML to standardized CSV** (`04_entsoe_xml_to_csv.py`) — one
   `timestamp_utc, value` table per dataset in `data/02_interim/`.
6. **Combine datasets** on `timestamp_utc`, aggregate to hourly means, and impute
   missing forecast fundamentals (`05_combine_dataset_csvs.py`).
7. **Write the data QA report** (`data_qa_report.md`).
8. **Build the feature table** (`pipeline_helpers/01_entsoe_data/06_build_features.py`).

The main dataset uses `day_ahead_prices`, `load_forecast`, `solar_forecast`,
`wind_onshore_forecast`, and `wind_offshore_forecast` — hourly day-ahead prices plus
four ex-ante fundamental drivers, comfortably satisfying the "price + at least two
fundamentals" requirement and matching the tradable-model decision from Section 3.

### Timezone and DST — the canonical-UTC rule

The canonical timestamp is `timestamp_utc`, used for every join, filter, validation
split, and model window. German local time is carried as `timestamp_local` (derived
via `Europe/Berlin`) purely for interpretation and trading blocks. Using UTC as the
join key is what makes the 23-hour and 25-hour DST days harmless: there are never
duplicate or missing key timestamps.

A concrete DST lesson is baked into the feature code. The full daily price-curve lag
features (the 24-hour shapes) are built on **UTC** hours, not local hours, so every
day has exactly 24 columns even across spring/autumn clock changes. Local time is used
only for *row-wise* calendar features (hour, weekday, month), which are DST-safe
because they never pivot a whole local day.

### Missing data and leakage-safe imputation

ENTSO-E forecast series occasionally have gaps (a real example was a February 2022
`load_forecast` gap, confirmed source-side, not a parser bug). Imputation in
`05_combine_dataset_csvs.py` fills a missing forecast value at time *t* with the same
column at *t − 24h*, and only for the four forecast driver columns. The fill is
applied **repeatedly, up to 7 times**, so a value can be recovered from *t − 24h*,
then *t − 48h*, and so on up to a week back — this recovers multi-day outages while
still only ever copying an *older* value, never a future one. Any rows still missing a
forecast driver after the seven passes are dropped before features are built. Because
every fill reaches strictly backward in time, the procedure stays leakage-safe.

### The QA report

`01_build_dataset.py` writes `data/03_processed/.../data_qa_report.md` covering: run name,
source endpoint, included datasets, requested local window, API UTC window, parsed vs
final frequency, expected vs actual row counts, coverage, duplicate timestamps,
missing values, imputation counts, outlier checks, timestamp-alignment explanation,
DST handling, and known limitations. This is the Task-1 generated QA artifact.

---

## 5. Feature engineering: rich, but leakage-safe by construction

Features are built in `pipeline_helpers/01_entsoe_data/06_build_features.py` in four blocks.
The governing principle is the Section-3 information cutoff: a full next-day forecast
must never peek at another hour of the same delivery day.

1. **Fundamental features** — total wind (onshore + offshore), renewable total,
   residual load (load − solar − wind), and wind/solar/renewable shares of load. These
   encode the merit-order intuition that price is driven by *residual* load.
2. **Calendar features** — local German hour, weekday, month, day-of-year,
   holiday flags (German nationwide holidays via the `holidays` package), cyclic
   sine/cosine encodings, and weekday dummies. All row-wise, all DST-safe.
3. **Daily UTC price-curve lags** — the full 24-hour price shapes from d-1, d-2, d-3,
   and d-7, plus min/max/mean summaries for d-1 and d-7 only. This is the LEAR-style
   structure that captures intraday and weekly seasonality.

There are deliberately **no plain same-hour price lags** (an earlier version carried
`price_lag_24/48/168`). They were removed because they are exact duplicates of the
daily price-curve columns: `price_lag_24` is identically the d-1 diagonal
`price_d1_h{H}`, `price_lag_48` the d-2 diagonal, and `price_lag_168` the d-7 diagonal.
Keeping both fed perfectly collinear columns into the linear model, which makes its
coefficients unstable without adding any information. Dropping the scalars keeps the
feature set honest and the LEAR weights well-posed; the same lagged information is still
present, hour-by-hour, in the price-curve block.

`reindex_to_full_hourly_grid` re-inserts any hours the combine stage dropped as empty
rows, so the table sits on a gap-free hourly UTC clock before lagged daily price curves
are constructed. The empty filler rows are removed again by the final `dropna` on the
required columns, so they never reach the model.

### 5.1 From feature store to model-ready table (`01_modelling_dataset.py`)

`06_build_features.py` writes one wide feature store with every candidate column. Which of
those columns a model is actually allowed to see is decided separately by
`pipeline_helpers/02_modelling/01_modelling_dataset.py`, using the editable policy file
`pipeline_helpers/02_modelling/feature_sets.json`. Keeping feature selection out of the
models means the leakage rules live in exactly one place, and every model is guaranteed
to receive the same safe column set for the same forecast setup.

Three **forecast setups** are supported:

- **`hourly_day_ahead`** — one row per delivery hour, target = hourly price, for a
  next-day forecast. Price curves from previous UTC delivery days are allowed.
- **`hourly_period`** — one row per delivery hour, target = hourly price, for a
  multi-day period view. Price-history features are removed, so the period view does not
  depend on prices from inside the period being forecast.
- **`period_average`** — one row per delivery period and block, target =
  the average price over that period for baseload, peakload, and offpeak.
  Fundamentals become per-period mean/min/max/std features, and price history becomes
  previous-period target lags.

The feature policy is selected centrally from the forecast setup and model family:

- LEAR / RANSAC-LASSO use broad linear-safe feature sets. For `hourly_day_ahead` this
  includes previous-day price curves and price summaries; for `hourly_period` it removes
  all price history.
- boosted trees / Theil-Sen use compact feature sets. For `hourly_day_ahead` this is
  load, solar, wind total, plus previous UTC daily price curves; for `hourly_period` it is
  only load, solar, and wind total.
- for `period_average`, LEAR / RANSAC-LASSO use all numeric period features except raw
  calendar integers, while boosted trees / Theil-Sen use load mean, solar mean, wind mean,
  block dummies, `target_lag_1`, and `target_lag_2`.

The module is strict about its contract, and the errors it raises are part of the design:

- `require_columns` raises *"Feature table is missing required features: …"* if the
  calendar columns a model needs are absent — a malformed feature store fails loudly
  instead of training on a silently shorter feature set.
- an unknown forecast setup raises *"Unsupported forecast setup: …"*, so a typo can never fall through
  to a wrong-but-plausible run.
- `period_average` automatically builds baseload, peakload, and offpeak targets, so the
  user no longer has to rerun the same model per block.

The result is a small `ModellingDataset` (table, target column, selected feature
columns, forecast setup, and feature-policy label) that the validation engine consumes
without ever needing to know how the columns were chosen.

---

## 6. The models

Each model lives in `pipeline_helpers/02_modelling/<name>.py` and exposes the same tiny
contract so the validation engine never needs model-specific knowledge:

```python
MODEL_NAME
build_param_grid(**options)   # list of parameter dicts to try
train(train_data, params)     # returns fitted state
predict(state, test_data, params)
output_folder_name(**options) # optional, for tidy artifact folders
```

Models are loaded dynamically by `model_support.load_model_module` via `importlib`,
so adding a model is just dropping in a new file. This uniform interface was a
deliberate early design decision: the same engine then drives the baseline, the LEAR
model, and the boosted trees.

### 6.1 Baseline — seasonal lag (`30_baseline_model.py`)

The baseline predicts the target from one of its own past values, with no fitted state —
`train()` returns `None` purely to honour the interface, and `predict()` just reads a
lagged column. The lag is measured in **rows of the modelling table**, and because a row
means a different span of time for each target, `build_param_grid` returns a different
grid per target:

- **`hourly`** target — a row is one hour, so the candidate lags are 24h (same hour
  yesterday), 48h (two days ago), and 168h (same hour last week). The headline baseline
  is the same-hour-last-week lag of 168 rows.
- **`period_average`** target — a row is one whole delivery period and block, so the
  candidate lags are the previous 1, 2, 4, 7, and 12 periods of the same block. The `period_days` setting does
  not change the grid: they only change what "one row" already means, so a row-based lag
  scales automatically.

Its job is to be the honest denominator: an improved model that cannot beat last week's
price is not worth deploying. Lago et al. explicitly warn against weak benchmarks, so a
real seasonal naive is used rather than a trivial one.

### 6.2 Headline model — LEAR-style 24-hour regularised ARX (`31_lear_model.py`)

This is the project's main forecaster: a **LEAR-style** model, i.e. 24 separate
regularised linear models, one per local delivery hour. It is described as
"LEAR-style" rather than an exact reproduction of any one paper. The lineage:

- Ziel ([2016](https://arxiv.org/abs/1509.01966)) defines LEAR as a LASSO-estimated
  autoregressive model with exogenous variables and justifies the rich d-1 / d-7 lag
  structure.
- The `epftoolbox` reference implementation
  ([docs](https://epftoolbox.readthedocs.io/en/latest/modules/lear_model.html),
  [repo](https://github.com/jeslago/epftoolbox)) confirms the 24-hourly-model design.
- Ziel & Weron ([2018](https://arxiv.org/abs/1805.06649)) show hour-specific
  (univariate) structures are not dominated by pooled models — so 24 models is
  defensible, not just convenient.
- Uniejewski & Weron ([2018](https://www.mdpi.com/1996-1073/11/8/2039)) motivate the
  optional `asinh` price transform for spikes; Uniejewski
  ([2024](https://arxiv.org/abs/2404.03968)) motivates ElasticNet as an alternative to
  pure LASSO and confirms cross-validation works for tuning.

Implementation details that matter:

- Each hourly model is a `Pipeline([StandardScaler, fixed-penalty regressor])`, so
  scaling is fitted inside each fold with no leakage.
- The outer rolling validation tests the regularisation grid directly. LEAR then
  chooses the best validated setting separately for each delivery hour from the
  out-of-sample validation predictions. This keeps hyperparameter selection visible in
  the main validation tables and still gives genuinely **24 different alphas** when
  different hours need different regularisation.
- The fitted state records `alpha_by_hour`, `fitted_hours`, and
  `n_train_rows_by_hour`, and the model raises if any of the 24 hourly fits fail, so a
  silently half-trained model can never be scored.
- Target transforms `raw` and `asinh` are both supported; `raw` is the default and is
  kept unless validation shows `asinh` helps.

### 6.3 Nonlinear benchmark — boosted trees (`32_boosted_tree_model.py`)

The nonlinear counterpart is a single pooled `HistGradientBoostingRegressor`. The
motivation is that trees capture interactions (hour × residual load, weekday ×
renewable share) automatically. The choice of boosted trees over deep nets is
deliberate and literature-backed: Xie et al.
([2022](https://link.springer.com/article/10.1007/s00202-021-01410-6)) and a recent
short-training-window ENTSO-E study
([2025](https://arxiv.org/html/2506.10536v1)) show gradient-boosted trees are
competitive-to-superior for day-ahead prices, while Lago, De Ridder & De Schutter
([2018](https://www.sciencedirect.com/science/article/pii/S030626191830196X)) show deep
learning works but adds heavy tuning. For a case study, boosted trees give the better
performance/interpretability/delivery trade-off. Kernel methods and neural nets were
explicitly considered and rejected for cost and tuning burden.

Implementation notes: the model declares `local_hour`, `local_weekday`, and
`local_month` as native categoricals; internal early stopping is **off** because the
external rolling validation owns model selection; `build_param_grid` returns four
configurations (loss × learning-rate / depth variants).

### 6.4 Robust linear check — Theil-Sen (`33_theil_sen_model.py`)

The Theil-Sen model is included as a robustness experiment. It is a pooled robust
linear regression fitted on the feature set selected by `feature_sets.json`. In testing
it behaved best on compact, low-correlation feature sets: load, solar, wind total, and
where relevant the previous UTC daily price curves or period target lags. It is not the
headline model because Theil-Sen scales poorly in high dimension and does not perform
LASSO-style feature selection.

### 6.5 Robust sparse check — RANSAC-LASSO (`34_ransac_lasso_model.py`)

RANSAC-LASSO is another robustness experiment. For hourly targets it also fits 24
separate models, one per delivery hour; for period-average targets it uses one pooled
model. It wraps `Lasso` inside `RANSACRegressor`: the model repeatedly fits sparse
linear regressions on random training subsets, chooses an inlier set, and refits on
that subset. This can improve normal-regime MAE when a few spikes dominate training,
but it can also classify real scarcity or negative-price events as outliers. For that
reason it should be judged by MAE **and** the tail metrics, not by average error alone.

---

## 7. Validation: rolling-origin, leakage-free, operationally realistic

Validation lives in `pipeline_helpers/02_modelling/09_validation.py` and is driven by
`pipeline_steps/02_validate_model.py`. It is the part Lago et al.
([2021](https://arxiv.org/abs/2008.08004)) care about most, and their best-practice
review shaped every choice here: **no random K-fold** (it leaks the future into the
past), proper time-series blocking, a strong baseline, transparent metrics, and a
reproducible dataset.

The fixed forecasting task is:

```text
train previous 24 months  ->  test next 1 month  ->  step forward 1 month
```

That yields **35 monthly folds** spanning Feb-2023 to Dec-2025. A fixed-length rolling
window (not an expanding one) is used on purpose: an expanding window would
under-train early folds and over-train late ones, confounding "model quality" with
"training-set age". Monthly retraining also mirrors how a desk would actually operate,
which is the *updateable MOS* idea from Wilson & Vallée made concrete.

The engine: load features → infer complete calendar-month windows → build rolling
train/test splits → loop each model's `build_param_grid` → score every fold →
select the best parameters → save artifacts. A fold is only scored if its prediction
coverage clears `MIN_PREDICTION_COVERAGE = 0.99`, so a fold with missing predictions
cannot flatter the average. Selection ranks on `MODEL_SELECTION_METRICS = [mae,
top_decile_mae, bottom_decile_mae, rmse]` — MAE first, with the tail metrics as
tie-breakers so a model that wins on the mean but collapses on spikes does not get
picked.

Outputs per run (under `models/<dataset>/<run-name>/`):

- `validation_summary.csv` — averaged metrics per parameter setting.
- `validation_metrics.csv` — one row per parameter × fold.
- `predictions.csv` — slim, out-of-sample predictions across **all** folds for the
  selected parameters (`timestamp; local_hour; fold; y_true; y_pred; y_pred_lower;
  y_pred_upper`).
- `model.joblib` + `metadata.json` — the last-fold model and full run metadata.

---

## 8. Metrics: six, and only six

`02_metrics.py` was deliberately trimmed from a sprawling 14 columns to the six that
actually inform a decision (`constants.METRIC_NAMES`):

| Metric | Meaning |
| --- | --- |
| `mae` | average absolute hourly error — the headline |
| `rmse` | penalises large misses more heavily |
| `bias` | average signed error (systematic over/under) |
| `top_decile_mae` | accuracy in the highest-price (scarcity-adjacent) hours |
| `bottom_decile_mae` | accuracy in the lowest-price / negative hours |
| `scarcity_price_mae` | error above `SCARCITY_PRICE_THRESHOLD` (150 €/MWh) |

MAE and RMSE satisfy the required level metrics; the decile and scarcity metrics are
the required tail/stress metrics. The trimming was a clean-code decision: every column
that nobody read (raw counts, redundant thresholds) was removed so the metric tables
stay legible.

---

## 9. Forecast uncertainty: empirical P10–P90 residual bands

A point forecast alone cannot drive a risk-aware trade, so `03_prediction_bands.py` adds
an empirical band:

- `residual_quantiles_by_hour(predictions, lower=0.10, upper=0.90)` computes, **per
  local hour**, the 10th and 90th percentile of the validation residual
  (`residual = y_true − y_pred`).
- `add_prediction_bands(...)` applies those offsets to produce `y_pred_lower` and
  `y_pred_upper`.
- `band_coverage(...)` reports the share of actuals that fall inside the band.

The band quantiles are `BAND_LOWER_QUANTILE = 0.10` and `BAND_UPPER_QUANTILE = 0.90`,
so a well-calibrated band should cover ≈ 80% of outcomes.

There is a subtle but important honesty point here. The bands are estimated **only
from out-of-sample validation residuals**. For a forward delivery period the model
retrains on that period's own window, and using that period's residuals to build its
own band would be leakage — so forward periods fall back to a risk-buffer proxy
(Section 11). This is the same information-cutoff discipline from Section 3, applied to
uncertainty rather than to point forecasts.

---

## 10. Results, figures, and what they say

The validated model set covers the required baseline, regularised linear model,
nonlinear boosted-tree model, robust linear checks, and direct period-average target.
The most important runs are:

| Run | Forecast setup | MAE | RMSE | Bias | Band coverage |
| --- | --- | ---: | ---: | ---: | ---: |
| `baseline_model__hourly_day_ahead` | hourly next-day | 26.64 | 39.53 | +0.03 | 79.9% |
| `boosted_tree_model__hourly_day_ahead` | hourly next-day | 15.28 | 23.94 | +2.17 | 79.9% |
| `ransac_lasso_model_raw__hourly_day_ahead` | hourly next-day | 15.58 | 24.22 | -0.96 | 79.9% |
| `lear_model_elasticnet_raw__hourly_day_ahead` | hourly next-day | 16.23 | 26.35 | +1.75 | 79.9% |
| `boosted_tree_model__hourly_period` | hourly period view | 28.30 | 40.14 | +16.06 | 79.9% |
| `theil_sen_model_raw__hourly_period` | hourly period view | 29.85 | 39.57 | +17.42 | 79.9% |
| `lear_model_lasso_raw__period_average__15d` | direct 15d average | 17.13 | 19.80 | +5.47 | 79.7% |

The band coverage stays very close to the 80% target, which is the important calibration
check for Task 3. The table also documents a useful modelling lesson: models using
previous price curves are strongest for true next-day forecasting, while period views
must remove those price dependencies unless the period is already partly known. For
longer period views, compact fundamental models and direct period-average models become
more honest even when their hourly MAE is worse.

`pipeline_steps/03_run_forecast_view.py` writes the Task-3 forecast-view figures into
the selected output folder:

- `forecast_actual_band.png` — forecast line, actuals, and the P10–P90 band as a shaded
  region over a recent window.
- `fair_value_vs_benchmark.png` — forecast fair value versus benchmark by block.
- `signal_by_block.png` — edge versus benchmark, coloured by direction.
- `forecast_heatmap.png` — optional hourly forecast shape by date and local delivery hour
  for hourly setups.

These satisfy the "at least two figures" requirement and visually demonstrate the band
calibration, block-level fair values, and trading signal.

---

## 11. Prompt-curve translation: from hourly forecast to a tradable view

This is Task 3, implemented in `pipeline_helpers/03_curve_translation/` and driven by
`pipeline_steps/03_run_forecast_view.py`. It is where the forecast becomes a trade.

### 11.1 Blocks

`01_forecast_blocks.py` aggregates the hourly forecast into the three standard
curve-relevant blocks: **baseload**, **peakload**, and **offpeak**.
Peak/offpeak is defined on German local market time (peak = weekday hours 08:00–20:00
local), which is correct for German market interpretation and does not disturb the UTC
joins. `calculate_block_band` aggregates the per-hour P10–P90 band up to a block-level
band when band columns are present.

### 11.2 Where the forecast comes from

`10_window_prediction.py` trains exactly one user-selected split: a training window
immediately before the requested prediction window. It first reads validated
hyperparameters from the matching model folder (`best_params.json`, `metadata.json`, or
`validation_summary.csv`); if none exist, it prints a clear warning and falls back to the
first grid row. The output is one concrete prediction package under `outputs/`, including
`predictions.csv`, `metrics.csv`, `model.joblib`, and metadata describing the model,
forecast setup, training dates, prediction dates, and parameter source.

### 11.3 The band-driven signal

`02_curve_view.py` was rewritten so the **band drives the signal** instead of a hand-tuned
threshold. The old `confidence_score` / `risk_buffer` thresholds were removed entirely.
The logic is now:

- `derive_forecast_band` prefers the empirical `p10_p90_residual` band; if unavailable
  (forward retrain), it falls back to a `risk_buffer_proxy` of
  `forecast ± max(MAE, 0.5 × tail_metric)`.
- `derive_signal` compares the benchmark to the band: benchmark **below** the band →
  **Long**, **above** → **Short**, **inside** → **Neutral**. If the benchmark is more
  than one band-width beyond the edge, the signal becomes **Strong**. It returns the
  signal plus the margin beyond the edge.
- `build_decision_rationale` writes a plain-language explanation of why.

This is a genuine improvement over a fixed threshold: the conviction now scales with
the model's own demonstrated uncertainty. A worked example makes the point — for
November 2025 baseload, a symmetric risk-buffer proxy says **Long**, but the wider
empirical P10–P90 band says **Neutral** because the trailing benchmark (83.27 €/MWh)
sits *inside* the forecast band (78.98–131.49 €/MWh). The empirical band prevents an
over-confident trade. That is the band earning its place.

### 11.4 Benchmark, desk action, invalidation

Forward price data is not required by the brief and is often paid, so the benchmark is
a proxy chosen from: a trailing realised day-ahead average, a same-month historical
average, or a manually supplied curve price. The report is explicit that this is a
proxy, not a traded forward.

Each curve view (`curve_view_report.md` + `curve_view_summary.csv`) states the fair
value with its band, the benchmark and edge, the signal with its rationale, the
**desk action** (e.g. "do not add directional exposure; monitor until the edge clears
the risk buffer"), the model-error context (MAE + tail metric), and explicit
**invalidation logic**: material load/wind/solar forecast revisions, outages or plant
returns, flow disruptions, market-regime changes, recent model error exceeding validation
error, or liquidity/execution constraints. Together these satisfy the Task-3
requirements for a concrete signal, a trading interpretation, and invalidation rules.

`03_run_forecast_view.py` prints the forecast setup, train/prediction windows, predicted
row coverage, and a per-block signal summary (signal, fair value, benchmark, edge). It
also writes `curve_view_summary.csv`, `curve_view_report.md`, per-block reports under
`curve_translation/<block>/`, and the plots listed above.

---

## 12. AI-accelerated workflow

Task 4 is implemented as the helper `pipeline_helpers/03_curve_translation/03_ai_commentary.py`
and is called programmatically from `03_run_forecast_view.py` when requested. It reads
the computed curve-summary rows, builds a constrained prompt that is forbidden from
inventing numbers (it may only use the computed values), calls the **OpenAI Responses
API** with the key read from `OPENAI_API_KEY` (environment only), and writes
`ai_commentary.md`. Every run logs the prompt, the output, and any failure as JSON under
`ai_logs/`. If the API is unavailable or out of quota, a **deterministic fallback**
writes a templated commentary from the same numbers, so the pipeline never breaks. The
model name comes from `OPENAI_MODEL` (default `gpt-4.1-mini`). This meets the minimum AI
requirements — called from code, prompts / outputs / failures logged, no committed
secrets — and is intentionally kept minimal.

---

## 13. Code map (the structure of the whole code)

Top-level pipeline steps (`pipeline_steps/`, numbered in run order):

- `01_build_dataset.py` — download, parse, combine, QA, feature table.
- `02_validate_model.py` — rolling validation, metrics, bands, artifacts.
- `03_run_forecast_view.py` — train once on a chosen window, translate it into curve views,
  write plots/reports, and optionally call AI commentary.

Data helpers (`pipeline_helpers/01_entsoe_data/`): `00_constants.py`, `01_date_windows.py`,
`02_dataset_folders.py`, `03_entsoe_api.py`, `04_entsoe_xml_to_csv.py`,
`05_combine_dataset_csvs.py`, `06_build_features.py`.

Modelling helpers (`pipeline_helpers/02_modelling/`): `00_constants.py`, `02_metrics.py`,
`03_prediction_bands.py`, `09_validation.py`, `30_baseline_model.py`, `31_lear_model.py`,
`32_boosted_tree_model.py`, `33_theil_sen_model.py`, `34_ransac_lasso_model.py`,
`10_window_prediction.py`, plus
`05_model_support.py`, `04_model_io.py`,
`01_modelling_dataset.py`.

Curve-translation helpers (`pipeline_helpers/03_curve_translation/`): `00_constants.py`,
`01_forecast_blocks.py`, `02_curve_view.py`, `03_ai_commentary.py`.

The shape of the repository mirrors the red line: `entsoe_data` answers "what do we
know and when", `modelling` answers "what is the price given that knowledge, and how
uncertain are we", and `curve_translation` answers "what does the desk do about it".

---

## 14. Reproducible run order

```bash
# 0. environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill ENTSOE_API_KEY (and optional OPENAI_API_KEY)

# 1. build the dataset
.venv/bin/python pipeline_steps/01_build_dataset.py

# 2. validate the baseline
.venv/bin/python pipeline_steps/02_validate_model.py

# 3. validate the headline LEAR model
.venv/bin/python pipeline_steps/02_validate_model.py

# 4. forecast view: prediction -> curve translation -> plots -> optional AI
.venv/bin/python pipeline_steps/03_run_forecast_view.py
```

---

## 15. Scope, honest limitations, and future work

**What the prototype is.** A daily rolling day-ahead fair-value workflow. Prompt-week /
prompt-month views are interpreted as rolling daily fair-value updates or historical
simulations, *not* as a one-shot forecast that somehow knows future realised prices
inside the delivery month.

**Honest limitations.**

- The period-oriented hourly setup assumes load/wind/solar forecasts are available over
  the requested future window. For the proof of concept this is acceptable, but a
  production curve model should add a separate fundamental-forecast layer for horizons
  beyond the next day.
- The actual-input oracle model from Section 3 is argued for but not implemented; the
  oracle-vs-tradable gap is therefore described, not yet measured.
- The benchmark is a realised-price proxy, not a traded forward curve; a manual traded
  curve price can be supplied when available.
- Fundamentals are restricted to load/solar/wind by design — no fuel, carbon, or flows.
- Training is still mostly sequential; parallelising validation across parameter grids,
  folds, and hours is the highest-value engineering improvement.

**Future work, with its sources.** Build separate fundamental forecasts (Maciejowska,
Nitka & Weron); add probabilistic fundamental forecasts (Uniejewski & Ziel); run the
boosted-tree and oracle comparisons; extend to multi-day-ahead horizons (Beran, Vogler
& Weber). Each of these is a clean next step, not a gap in the current logic.

---

## 16. Closing position

> This is a daily rolling German day-ahead price forecasting prototype. It uses public
> ex-ante forecast fundamentals (load, wind, solar), lagged historical prices, and
> calendar features to produce hourly price forecasts with calibrated P10–P90
> uncertainty bands. Those forecasts are aggregated into curve-relevant fair-value
> views, and the band — not a hand-tuned threshold — drives a long / neutral / short
> signal with an explicit desk action and invalidation logic. Prompt-week / month views
> are rolling daily updates, never one-shot forecasts of future realised prices.

That framing is internally consistent, matches the assignment, respects the data that
is actually available before the auction, and is grounded throughout in the electricity
price forecasting literature reviewed above.

---

## References

1. Marzban, Sandgathe & Kalnay (2006). *MOS, Perfect Prog, and Reanalysis.* https://faculty.washington.edu/marzban/mos.pdf
2. Brunet, Verret & Yacowar (1988). *An Objective Comparison of MOS and Perfect Prog Systems.* https://journals.ametsoc.org/view/journals/wefo/3/4/1520-0434_1988_003_0273_aocomo_2_0_co_2.xml
3. Wilson & Vallée (2003). *The Canadian Updateable MOS (UMOS) System: Validation against Perfect Prog.* https://journals.ametsoc.org/view/journals/wefo/18/2/1520-0434_2003_018_0288_tcumos_2_0_co_2.xml
4. Fay & Ringwood (2010). *On the Influence of Weather Forecast Errors in Short-Term Load Forecasting Models.* https://ee.maynoothuniversity.ie/jringwood/Respubs/J156DFPS.pdf
5. Wang et al. (2020). *Building Thermal Load Prediction through Shallow ML and Deep Learning.* https://www.sciencedirect.com/science/article/am/pii/S0306261920301951
6. Runge & Saloux (2023). *A Comparison of Prediction and Forecasting AI Models for District Heating Demand.* https://www.sciencedirect.com/science/article/abs/pii/S0360544223000555
7. Fildes, Randall & Stubbs (1997). *One Day Ahead Demand Forecasting in the Utility Industries.* https://www.jstor.org/stable/3009939
8. Maciejowska, Nitka & Weron (2021). *Enhancing Load, Wind and Solar Generation for Day-Ahead Forecasting of Electricity Prices.* https://www.sciencedirect.com/science/article/abs/pii/S014098832100178X
9. Uniejewski & Ziel (2025). *Probabilistic Forecasts of Load, Solar and Wind for Electricity Price Forecasting.* https://arxiv.org/abs/2501.06180
10. Kulakov & Ziel (2019). *The Impact of Renewable Energy Forecasts on Intraday Electricity Prices.* https://arxiv.org/abs/1903.09641
11. Goodarzi, Perera & Bunn (2019). *The Impact of Renewable Energy Forecast Errors on Imbalance Volumes and Electricity Spot Prices.* https://www.sciencedirect.com/science/article/abs/pii/S0301421519304057
12. Beran, Vogler & Weber (2021). *Multi-Day-Ahead Electricity Price Forecasting: Fundamental, Econometric and Hybrid Models.* https://ideas.repec.org/p/dui/wpaper/2102.html
13. Lago, Marcjasz, De Schutter & Weron (2021). *Forecasting Day-Ahead Electricity Prices: A Review of State-of-the-Art Algorithms, Best Practices and an Open-Access Benchmark.* https://arxiv.org/abs/2008.08004
14. Weron (2014). *Electricity Price Forecasting: A Review of the State-of-the-Art with a Look into the Future.* https://doi.org/10.1016/j.ijforecast.2014.08.008
15. Ziel (2016). *Forecasting Electricity Spot Prices using Lasso: On Capturing the Autoregressive Intraday Structure.* https://arxiv.org/abs/1509.01966
16. Ziel & Weron (2018). *Day-Ahead Electricity Price Forecasting with High-Dimensional Structures: Univariate vs. Multivariate.* https://arxiv.org/abs/1805.06649
17. Uniejewski & Weron (2018). *Efficient Forecasting of Electricity Spot Prices with Expert and LASSO Models.* https://www.mdpi.com/1996-1073/11/8/2039
18. Uniejewski (2024). *Regularization for Electricity Price Forecasting.* https://arxiv.org/abs/2404.03968
19. Xie, Chen, Lai, Ma & Huang (2022). *Forecasting the Clearing Price in the Day-Ahead Spot Market using eXtreme Gradient Boosting.* https://link.springer.com/article/10.1007/s00202-021-01410-6
20. *Data-driven Day-Ahead Market Prices Forecasting: A Focus on Short Training Set Windows* (2025). https://arxiv.org/html/2506.10536v1
21. Lago, De Ridder & De Schutter (2018). *Forecasting Spot Electricity Prices using Deep Learning.* https://www.sciencedirect.com/science/article/pii/S030626191830196X
22. epftoolbox — LEAR reference implementation. https://epftoolbox.readthedocs.io/en/latest/modules/lear_model.html · https://github.com/jeslago/epftoolbox

*Data source: ENTSO-E Transparency Platform (https://web-api.tp.entsoe.eu/api). Token guide: https://transparencyplatform.zendesk.com/hc/en-us/articles/12845911031188-How-to-get-security-token.*
