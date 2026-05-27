# European Power Fair Value Case Study

**Theme:** European Power Fair Value: Forecasting Day-Ahead and Translating to Prompt Curve Views

**Full Name:** `[Insert full name]`  
**Email Address:** `[Insert email address]`  
**Deadline:** **Thursday, 4 June 2026** — one week from the assignment date.

---

## Objective

Build a prototype that produces a **daily fair-value view** for one European power market, grounded in public fundamentals, and demonstrates how that view can inform **prompt curve positioning and risk**.

The prototype should cover the trading workflow from **Day-Ahead price forecasting** through to **curve-relevant views** such as prompt week, prompt month, prompt quarter, shape, or spread positioning.

---

## 1. Public Data Ingestion and Data Quality — **Must Have**

Choose **one European power market** such as:

- Germany — **DE**
- France — **FR**
- Netherlands — **NL**
- Great Britain — **GB**

Use **only publicly accessible sources**. Do **not** use synthetic data or paid data.

### Minimum Data Requirements

Your dataset must include:

1. **Hourly Day-Ahead prices** for the chosen market.
2. At least **two fundamental drivers** at matching granularity.

Examples of acceptable fundamental drivers:

- **Load**
- **Wind generation or forecast**
- **Solar generation or forecast**
- **Nuclear availability proxy**
- **Net imports / interconnector flows**

### Required Ingestion Documentation

Document all public data sources and endpoints used.

Acceptable examples include:

- **ENTSO-E Transparency API documentation**
- ENTSO-E Postman Documenter pages
- National TSO public APIs or downloadable datasets
- Exchange or market operator public data, where freely accessible

For each source, document:

- **Dataset name**
- **Endpoint or download URL**
- **Fields used**
- **Timezone convention**
- **Data frequency**
- **Known limitations**

### Timezone and DST Handling

You must handle **timezone and daylight saving time correctly**.

Your pipeline should explicitly address:

- Local market time versus UTC
- 23-hour and 25-hour DST days
- Duplicate or missing local timestamps during clock changes
- Alignment between prices and fundamentals

### Required Data QA Checks

Implement and report QA checks covering:

- **Missingness** by field and timestamp
- **Duplicate timestamps**
- **Obvious outliers**
- **Coverage by field and time period**
- Timestamp alignment across price and fundamentals

The repo must include a generated **data QA output** such as a `.csv`, `.json`, `.md`, or notebook output.

---

## 2. Forecasting and Model Validation — **Must Have**

Create a model that produces a **Day-Ahead to curve-relevant forecast**.

Choose one of the following target approaches and justify your choice.

---

### Option A — **Recommended**

Forecast **next-day hourly Day-Ahead prices**, or peak/base blocks, and then derive:

- **Next-week expected average**
- **Next-month expected average**
- Forecast distribution bands, if available

This option is recommended because it naturally links hourly market fundamentals to prompt curve delivery averages.

---

### Option B

Forecast the **delivery-period average** directly from fundamentals.

Examples:

- Next-week baseload average
- Next-month baseload average
- Peakload average

---

### Required Models

You must include:

1. At least **one baseline model**, such as:
   - Seasonal naive
   - Last-week-same-day
   - Simple linear regression

2. At least **one improved model**, such as:
   - Gradient boosting
   - Regularised regression with engineered features
   - Simple structural hybrid model

### Required Validation Approach

Use **time-series appropriate validation**.

Acceptable methods include:

- Walk-forward validation
- Expanding-window validation
- Rolling-window validation
- Blocked cross-validation

Avoid leakage by ensuring that no future information is used in feature generation, scaling, model fitting, or target construction.

### Required Metrics

Report numerical performance metrics including:

- **MAE** for price levels
- **RMSE** for price levels
- At least one **tail or stress metric** if modelling extremes

Examples of tail metrics:

- MAE on top-decile price hours
- Error during negative price hours
- Error during scarcity hours
- Pinball loss for quantile forecasts

---

## 3. Prompt Curve Translation — **Must Have**

Show how the forecast informs **Day-Ahead to curve trading**.

You do not need to use forward price data, but you must provide a concrete method to translate forecast output into a tradable view.

### Required Translation Method

Include at least one method such as:

- Expected delivery-period mean
- Forecast distribution bands
- Risk premium proxy
- Confidence-weighted signal
- Forecast versus historical seasonal fair value
- Prompt week or prompt month fair-value estimate

Example workflow:

1. Forecast next-day hourly prices.
2. Aggregate simulated or rolling forecasts into **next-week** and **next-month baseload averages**.
3. Compare forecast fair value against observable curve prices, if available, or against a proxy benchmark.
4. Produce a signal such as:
   - **Long prompt month**
   - **Short prompt week**
   - **Long peak versus base**
   - **Spread between adjacent delivery periods**

### Required Trading Explanation

Explain what the desk would do with the signal.

Examples:

- Express via **prompt month exposure**
- Express via **prompt quarter exposure**
- Trade **peak/base shape**
- Trade **week-ahead versus month-ahead spread**
- Use the view for risk sizing or hedging decisions

### Required Invalidation Logic

Include what would invalidate the signal.

Examples:

- Sudden outage or plant return
- Large weather forecast revision
- Flow or interconnector disruption
- Market regime shift
- Model error outside expected confidence bands
- Liquidity or execution constraints

---

## 4. AI-Accelerated Workflow — **Must Have**

Implement one **programmatic AI or LLM component** that measurably reduces manual work.

This must be integrated into the codebase. It should not be a manual chat transcript.

### Acceptable AI Components

Examples include:

1. **LLM-driven data QA rules and tests**
   - The model receives a schema and sample rows.
   - It proposes validation rules.
   - The pipeline executes those rules.
   - A QA report is produced.

2. **Automated drivers commentary**
   - The model generates a daily explanation using only computed metrics.
   - The commentary must not invent numbers.
   - It should link back to underlying tables or outputs.

3. **Config generation for ingestion**
   - The model converts documentation or field definitions into structured ingestion config.
   - The pipeline uses the generated config.

### Minimum AI Requirements

Your implementation must show that:

- The **LLM is called from code** using an API or local model.
- **Prompts are logged**.
- **Outputs are logged**.
- **Failure modes are logged**.
- **No secrets are committed**.
- API keys are loaded from **environment variables only**.

---

## Submission Requirements

Submit a short write-up of **1–3 pages** in either:

- **PDF**, or
- **Markdown**

The write-up must include your **full name and email address at the top**.

You must also submit a **repo or zipped folder** containing:

- Reproducible pipeline code, scripts, and/or notebooks
- **README** with setup and run instructions
- **requirements.txt** or **pyproject.toml**
- Data QA output
- At least **2 figures or tables**
- AI component implementation, including prompts, code, and logged outputs

---

## Optional but Strongly Encouraged

Include a `submission.csv` containing out-of-sample predictions for a clearly defined test window.

Required columns:

```csv
id,y_pred
```

Where:

- `id` is a timestamp or timestamp plus hour identifier
- `y_pred` is the predicted value

---

## Evaluation Criteria

The submission will be assessed on:

### Dataset Correctness

- Proper source usage
- Correct timestamp alignment
- Correct DST handling
- Defensible cleaning choices

### Forecasting Rigor

- Appropriate validation design
- Baseline model included
- Improved model included
- Leakage avoidance
- Numerical metrics reported

### Trading Relevance

- Clear Day-Ahead to curve linkage
- Tradable prompt curve signal
- Sensible risk and invalidation logic

### Engineering Quality

- Reproducible structure
- Clear README
- Clean code organisation
- Documented assumptions
- Saved outputs and artifacts

### AI / LLM Productivity Lever

- Programmatic AI usage
- Controlled prompts
- Logged outputs
- Auditable failure handling
- No committed secrets

---

## Suggested Repo Structure

```text
power-fair-value-case-study/
├── README.md
├── TASK.md
├── requirements.txt
├── .env.example
├── configs/
│   └── market_config.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── ingestion/
│   │   └── fetch_public_data.py
│   ├── quality/
│   │   └── qa_checks.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── baseline.py
│   │   └── improved_model.py
│   ├── ai/
│   │   ├── prompts.py
│   │   └── llm_qa_rules.py
│   └── reporting/
│       └── build_report.py
├── outputs/
│   ├── qa_report.md
│   ├── metrics.csv
│   ├── figures/
│   ├── tables/
│   ├── ai_logs/
│   └── submission.csv
└── report/
    └── case_study_writeup.md
```

---

## Final Deliverables Checklist

Before submission, confirm that the package contains:

- [ ] **1–3 page write-up** in PDF or Markdown
- [ ] **Full name and email** at the top of the write-up
- [ ] Public source documentation
- [ ] Hourly Day-Ahead price data
- [ ] At least **two fundamental drivers**
- [ ] Correct timezone and DST treatment
- [ ] Data QA checks and output
- [ ] Baseline model
- [ ] Improved model
- [ ] Walk-forward or blocked validation
- [ ] MAE and RMSE metrics
- [ ] Tail or stress metric, if modelling extremes
- [ ] Prompt curve translation method
- [ ] Trading interpretation and invalidation logic
- [ ] Programmatic AI component
- [ ] Logged prompts, outputs, and failure modes
- [ ] README with setup and run instructions
- [ ] `requirements.txt` or `pyproject.toml`
- [ ] At least **2 figures or tables**
- [ ] Optional `submission.csv` with `id` and `y_pred`
