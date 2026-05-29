# Handover Summary: Actual Inputs vs Forecasted Inputs in Electricity Price Forecasting

## Context

We are working on a European power fair-value case study. The goal is to forecast hourly day-ahead electricity prices for one European market, probably Germany, using public data such as day-ahead prices, load, wind, solar, and other fundamentals.

The main methodological question we discussed was:

**Should the electricity price model be trained on actual realised fundamentals, or on forecasted fundamentals?**

Example:

For delivery day **X**, the target is the actual day-ahead electricity price for each hour.

Possible inputs:

1. **Actual fundamentals at day X**

   * actual wind generation
   * actual solar generation
   * actual load
   * actual weather

2. **Forecasted fundamentals available at day X − 1**

   * wind forecast for day X, published before the day-ahead auction
   * solar forecast for day X, published before the day-ahead auction
   * load forecast for day X, published before the day-ahead auction
   * weather forecast for day X, published before the day-ahead auction

The user’s argument was:

> Train the price model on actual realised inputs, so the model learns the true relationship between physical fundamentals and price. Then, in production, feed the model the best available forecasts for tomorrow. This makes the model independent of any specific forecasting provider and allows it to benefit automatically if better wind/load/solar forecasts become available later.

This is a reasonable argument. It corresponds to what meteorology/statistics literature calls **Perfect Prog** or **Perfect Prognosis**.

The opposite approach is:

> Train the price model directly on historical forecasted fundamentals, because in production the model will also receive forecasted fundamentals.

This corresponds to **MOS**, or **Model Output Statistics**, in weather-forecast postprocessing terminology.

---

# Key Concepts

## 1. Actual-input model / structural model / oracle model / Perfect Prog

This model is trained like:

```text
day-ahead price at hour t
= f(actual load at t,
    actual wind at t,
    actual solar at t,
    calendar features,
    lagged prices)
```

Then, for live forecasting, it is used like:

```text
future day-ahead price at hour t
= f(forecasted load at t,
    forecasted wind at t,
    forecasted solar at t,
    calendar features,
    lagged prices)
```

This is the user’s preferred logic.

### Advantages

* Learns the cleaner physical relationship between fundamentals and price.
* Does not depend too strongly on one historical forecast provider.
* If future wind/load/solar forecasts become much better, the price model can benefit without necessarily being retrained.
* Avoids learning the quirks, biases, and noise of one specific forecast model.
* Useful as a **structural price-response model**.
* Useful as an **upper-bound / oracle benchmark** if evaluated with actual inputs.

### Disadvantages

* There is a **train-live mismatch**: the model is trained on clean actuals but used with noisy forecasts.
* Forecasted variables can be smoother, biased, and less extreme than actuals.
* Forecast errors may be largest exactly during price spikes.
* Backtest performance using actual inputs may overstate live trading performance.
* If evaluated only on actuals, it can look like leakage because actual future wind/solar/load would not have been known before the auction.

### Best label

Call this the:

**“Structural / perfect-foresight / Perfect Prog model.”**

---

## 2. Forecast-input model / tradable model / MOS-style model

This model is trained and used like:

```text
day-ahead price at hour t
= f(load forecast for t,
    wind forecast for t,
    solar forecast for t,
    calendar features,
    lagged prices)
```

The important thing is that the forecasted fundamentals must have been available before the day-ahead auction.

### Advantages

* Training input distribution matches live input distribution.
* More realistic for trading.
* Can learn systematic bias/error patterns in the fundamental forecasts.
* Avoids look-ahead bias.
* Easier to defend in a day-ahead price forecasting case study.
* Aligns with the standard methodology in serious electricity-price-forecasting papers.

### Disadvantages

* Tied to the specific forecast provider or forecast methodology used historically.
* If the forecast provider changes model, the learned relationship may shift.
* If the old forecast model was bad, the price model may learn around its badness.
* If a future near-oracle forecast model becomes available, the price model might need retesting or retraining.
* Requires historical forecast archives, which are often harder to obtain than realised actuals.

### Best label

Call this the:

**“Tradable / ex-ante / MOS-style model.”**

---

# Main Conclusion

The best methodology is **not** to pick only one side.

The strongest solution is:

## Proposed solution: build and report two models

### Model A: Structural / oracle model

Train on **actual realised fundamentals**.

Purpose:

* learn physical price response
* show upper-bound performance
* show how much of price variation can be explained by actual fundamentals
* defend the user’s argument

### Model B: Tradable / ex-ante model

Train on **forecasted fundamentals available before the auction**.

Purpose:

* show realistic live trading performance
* avoid leakage
* follow electricity price forecasting literature
* produce the actual model used for the case study’s trading signal

Then compare:

```text
Performance gap = oracle model error - tradable model error
```

Actually, more clearly:

```text
Forecast-input performance loss
= tradable model MAE - oracle model MAE
```

Interpretation:

> The gap measures how much accuracy is lost because we do not know true future fundamentals and must rely on load/wind/solar forecasts.

---

# Recommended Wording for the Case Study

Use wording like this:

> I estimate two feature sets. The primary tradable model uses ex-ante fundamental forecasts available before the day-ahead auction, such as load, wind, and solar forecasts. This avoids look-ahead bias and matches the information set available to a trader. In addition, I estimate a realised-fundamentals model using actual load, wind, and solar generation. This is reported as a structural, perfect-foresight benchmark, analogous to the Perfect Prog approach in meteorological postprocessing. The difference between the two models’ out-of-sample performance quantifies the cost of fundamental forecast error.

And:

> I do not attempt to build separate wind, solar, or load forecasting models. These are treated as external public inputs. The project focuses on the price-response model, not on building a full fundamental forecasting stack.

---

# What the Literature Says

## Big picture

Most serious electricity price forecasting papers use **forecasted fundamentals** when they claim to build a tradable day-ahead forecast.

That means they usually try to match the real information set:

```text
At forecast time, only use what was known before the market cleared.
```

So, for day-ahead power price forecasting, they often use:

* day-ahead load forecast
* day-ahead wind forecast
* day-ahead solar forecast
* residual load forecast
* lagged prices
* calendar features
* sometimes fuel, carbon, outage, and cross-border flow data

However, several papers also work on **improving the fundamental forecasts themselves**, such as enhancing TSO load/wind/solar forecasts or converting point forecasts into probabilistic forecasts. For our case study, we probably do **not** have time to do this. We can simply use public forecasts as inputs and state that improving fundamental forecasts is out of scope.

---

# Papers and Studies Discussed

## 1. Marzban, Sandgathe & Kalnay — “MOS, Perfect Prog, and Reanalysis”

**Title:**
MOS, Perfect Prog, and Reanalysis

**Authors:**
C. Marzban, S. Sandgathe, E. Kalnay

**Year:**
2006

**Link:**
https://journals.ametsoc.org/view/journals/mwre/134/2/mwr3088.1.pdf

**Alternative link:**
https://faculty.washington.edu/marzban/mos.pdf

**Field:**
Meteorological postprocessing / weather forecasting

**Why it matters here:**
This is the clean theoretical framing of our debate.

It compares:

* **Perfect Prog:** train statistical relationship using observations / actuals, then feed model forecasts at prediction time.
* **MOS:** train statistical relationship directly using model forecasts as predictors.

This maps almost exactly to our electricity price question:

* train on actual wind/load/solar, then deploy with forecast wind/load/solar = Perfect Prog
* train on forecast wind/load/solar, then deploy with forecast wind/load/solar = MOS

**Conclusion:**
The paper explains that MOS is often expected to outperform Perfect Prog under certain linear assumptions because it directly learns the bias and error structure of the forecast model. But Perfect Prog has practical advantages: it is simpler, can be trained using observations, and is less tied to one specific numerical weather prediction model. The paper also notes that earlier work found Perfect Prog can outperform MOS in short-term forecasts, while updateable MOS can outperform both in some settings.

**Interpretation for us:**
This supports the idea that there is no universal answer. If we train on actual fundamentals, we are doing a Perfect Prog-style structural model. If we train on forecasted fundamentals, we are doing a MOS-style tradable model.

---

## 2. Brunet, Verret & Yacowar — “An Objective Comparison of Model Output Statistics and ‘Perfect Prog’ Systems in Producing Numerical Weather Element Forecasts”

**Title:**
An Objective Comparison of Model Output Statistics and “Perfect Prog” Systems in Producing Numerical Weather Element Forecasts

**Authors:**
N. Brunet, R. Verret, N. Yacowar

**Year:**
1988

**Link:**
https://journals.ametsoc.org/view/journals/wefo/3/4/1520-0434_1988_003_0273_aocomo_2_0_co_2.xml

**Field:**
Weather forecasting

**Why it matters here:**
This is an empirical comparison of MOS versus Perfect Prog.

**Conclusion:**
The study compared MOS and Perfect Prog systems for several weather elements. It is often cited as showing that Perfect Prog outperformed MOS for short-term forecasts in that setup.

**Interpretation for us:**
This supports the user’s intuition that training on actual inputs can be defensible, especially for short-horizon structural relationships. But it is not proof that actual-input training is always best for electricity prices.

---

## 3. Wilson & Vallée — “The Canadian Updateable Model Output Statistics (UMOS) System: Validation against Perfect Prog”

**Title:**
The Canadian Updateable Model Output Statistics (UMOS) System: Validation against Perfect Prog

**Authors:**
L. J. Wilson, M. Vallée

**Year:**
2003

**Link:**
https://journals.ametsoc.org/view/journals/wefo/18/2/1520-0434_2003_018_0288_tcumos_2_0_co_2.xml

**Field:**
Weather forecasting / postprocessing

**Why it matters here:**
This is another MOS vs Perfect Prog comparison, but with an updateable MOS system.

**Conclusion:**
The study validated updateable MOS against Perfect Prog and operational model forecasts. Marzban et al. summarize it as finding that updateable MOS outperformed both standard Perfect Prog and standard MOS for some weather variables such as 2-meter temperature, 10-meter wind direction/speed, and precipitation probability.

**Interpretation for us:**
If you can regularly update/retrain the model and keep forecast archives, MOS-style training can be very strong. But it requires historical forecast data and maintenance.

---

## 4. Fay & Ringwood — “On the Influence of Weather Forecast Errors in Short-Term Load Forecasting Models”

**Title:**
On the Influence of Weather Forecast Errors in Short-Term Load Forecasting Models

**Authors:**
Damien Fay, John V. Ringwood

**Year:**
2010

**Link:**
https://ee.maynoothuniversity.ie/jringwood/Respubs/J156DFPS.pdf

**Field:**
Short-term electricity load forecasting

**Why it matters here:**
This is one of the closest papers to our exact methodological debate, although the target is load rather than price.

**Important point:**
The paper says load forecasting models are usually trained using actual past weather readings rather than past weather forecasts. The reason is that using forecasted weather adds forecast noise to the training data, can bias parameter estimation, forecast archives may be unavailable, and meteorological models keep improving. But the paper also warns that when forecast errors not seen during training appear in live use, they can disproportionately affect load models.

**Conclusion:**
Training with actual weather can be justified, but one must account for weather forecast errors when the model is used operationally.

**Interpretation for us:**
This strongly supports the user’s argument. It gives a legitimate reason to train on actual fundamentals: forecast noise can pollute training, and old forecast models may not represent future forecast models. But it also supports the counterargument: if the model never sees forecast error during training, live forecast inputs can hurt performance.

---

## 5. Wang et al. — “Building Thermal Load Prediction through Shallow Machine Learning and Deep Learning”

**Title:**
Building Thermal Load Prediction through Shallow Machine Learning and Deep Learning

**Authors:**
Z. Wang et al.

**Year:**
2020

**Link:**
https://www.sciencedirect.com/science/article/am/pii/S0306261920301951

**Alternative LBNL page:**
https://eta.lbl.gov/publications/building-thermal-load-prediction

**Field:**
Building thermal load prediction

**Why it matters here:**
This is a very relevant non-electricity-price example. It compares the effect of using actual weather versus forecasted weather in training/testing building load models.

**Conclusion:**
The study develops multiple shallow and deep learning models for building thermal load prediction. The useful methodological takeaway for us is that weather forecast uncertainty matters. In the specific comparison we discussed, using actual weather in training but forecasted weather at test time hurt some models, especially XGBoost, while models exposed to forecast uncertainty during training were more robust. LSTM was less sensitive in that example.

**Interpretation for us:**
This is a warning against the pure “train on actuals, deploy on forecasts” approach. Some ML models can be sensitive to train-live mismatch. If forecasted inputs are noisy, training with similar noisy inputs can improve robustness.

---

## 6. Runge & Saloux — “A Comparison of Prediction and Forecasting Artificial Intelligence Models to Estimate the Future Energy Demand in a District Heating System”

**Title:**
A Comparison of Prediction and Forecasting Artificial Intelligence Models to Estimate the Future Energy Demand in a District Heating System

**Authors:**
Jason Runge, Etienne Saloux

**Year:**
2023

**Link:**
https://www.sciencedirect.com/science/article/abs/pii/S0360544223000555

**Alternative link:**
https://ideas.repec.org/a/eee/energy/v269y2023ics0360544223000555.html

**Field:**
District heating demand forecasting

**Why it matters here:**
This study explicitly distinguishes between a **prediction** approach and a **forecasting** approach, and uses actual weather forecasts from Canadian meteorological services. It compares AI/ML methods for future heating demand over 6-hour and 24-hour horizons.

**Conclusion:**
The paper is useful because it shows that in energy-demand problems, researchers explicitly separate same-time prediction from true future forecasting. It also shows that using forecasted weather inputs is a real operational issue, not a theoretical detail.

**Interpretation for us:**
Supports our proposed language: an actual-input model is a “prediction/structural” model, while a forecast-input model is the “forecasting/tradable” model.

---

## 7. Fildes, Randall & Stubbs — “One Day Ahead Demand Forecasting in the Utility Industries: Two Case Studies”

**Title:**
One Day Ahead Demand Forecasting in the Utility Industries: Two Case Studies

**Authors:**
R. Fildes, A. Randall, P. Stubbs

**Year:**
1997

**Link:**
https://www.jstor.org/stable/3009939

**Alternative link:**
https://www.researchgate.net/publication/233719876_One_Day_Ahead_Demand_Forecasting_in_the_Utility_Industries_Two_Case_Studies

**Field:**
Utility demand forecasting

**Why it matters here:**
This is an older but useful forecasting methodology paper for utilities.

**Conclusion:**
The paper studies one-day-ahead demand forecasting for water and gas utilities. It finds that extrapolative methods based only on past demand were outperformed by multivariate methods that include weather effects. It also emphasizes that explanatory-variable forecasts matter when evaluating a model ex ante.

**Interpretation for us:**
This supports the idea that fundamentals/weather matter, but evaluation must distinguish between:

* ex-post performance using known explanatory variables
* ex-ante performance when those explanatory variables themselves must be forecast

This supports reporting both an oracle/actual-input result and a tradable/forecast-input result.

---

## 8. Maciejowska, Nitka & Weron — “Enhancing Load, Wind and Solar Generation for Day-Ahead Forecasting of Electricity Prices”

**Title:**
Enhancing Load, Wind and Solar Generation for Day-Ahead Forecasting of Electricity Prices

**Authors:**
Katarzyna Maciejowska, Weronika Nitka, Tomasz Weron

**Year:**
2021

**Link:**
https://www.sciencedirect.com/science/article/abs/pii/S014098832100178X

**Alternative RePEc link:**
https://ideas.repec.org/a/eee/eneeco/v99y2021ics014098832100178x.html

**Field:**
Electricity price forecasting, German market

**Why it matters here:**
This is one of the most relevant electricity-price papers.

**Conclusion:**
The paper studies Germany and shows that TSO forecasts of load, wind, and solar generation can be biased. It improves these forecasts with ARX-type models and then uses the enhanced fundamental forecasts for day-ahead and intraday electricity price forecasting. It finds that improving fundamental forecasts improves price forecast accuracy and can improve decision value.

**Interpretation for us:**
This confirms that serious electricity-price papers often use **forecasted fundamentals**, not future actuals. It also shows that forecast quality matters a lot. However, we do not need to reproduce their extra work of enhancing load/wind/solar forecasts. We can use public forecasts directly and state that improving fundamentals is out of scope.

---

## 9. Uniejewski & Ziel — “Probabilistic Forecasts of Load, Solar and Wind for Electricity Price Forecasting”

**Title:**
Probabilistic Forecasts of Load, Solar and Wind for Electricity Price Forecasting

**Authors:**
Bartosz Uniejewski, Florian Ziel

**Year:**
2025

**Link:**
https://arxiv.org/abs/2501.06180

**PDF link:**
https://arxiv.org/pdf/2501.06180

**Field:**
Electricity price forecasting, German market

**Why it matters here:**
This paper explicitly discusses that traditional electricity price forecasting often uses point forecasts of exogenous variables such as load, solar, and wind. It then proposes using probabilistic/quantile forecasts of those fundamentals instead.

**Conclusion:**
The paper finds that incorporating probabilistic forecasts of load and renewable generation improves day-ahead electricity price forecast accuracy, with full probabilistic forecast information giving the strongest improvements.

**Interpretation for us:**
This supports the literature-standard approach of using forecasted fundamentals. But it also shows that the state of the art often goes further by modelling uncertainty in the fundamentals. We do not have time for that, so we can keep it simple and use point forecasts.

---

## 10. Kulakov & Ziel — “The Impact of Renewable Energy Forecasts on Intraday Electricity Prices”

**Title:**
The Impact of Renewable Energy Forecasts on Intraday Electricity Prices

**Authors:**
Sergei Kulakov, Florian Ziel

**Year:**
2019 arXiv / later publication

**Link:**
https://arxiv.org/abs/1903.09641

**Alternative link:**
https://ideas.repec.org/p/arx/papers/1903.09641.html

**Field:**
German intraday electricity prices

**Why it matters here:**
This paper is about how forecast errors in wind and solar affect intraday prices.

**Conclusion:**
The authors model the impact of wind and solar forecast errors on German intraday electricity prices. Their model uses day-ahead auction curves and renewable forecast errors, and they find that renewable forecast errors have nonlinear effects on intraday prices.

**Interpretation for us:**
This supports the idea that forecast errors are not harmless. If wind/solar forecasts are wrong, price forecasts and market prices can move materially. This is a reason to care about forecasted fundamentals and possibly compare actual-input vs forecast-input models.

---

## 11. Goodarzi, Perera & Bunn — “The Impact of Renewable Energy Forecast Errors on Imbalance Volumes and Electricity Spot Prices”

**Title:**
The Impact of Renewable Energy Forecast Errors on Imbalance Volumes and Electricity Spot Prices

**Authors:**
Shadi Goodarzi, H. N. Perera, Derek W. Bunn

**Year:**
2019

**Link:**
https://www.sciencedirect.com/science/article/abs/pii/S0301421519304057

**Alternative RePEc link:**
https://ideas.repec.org/a/eee/enepol/v134y2019ics0301421519304057.html

**Field:**
Electricity markets, renewable forecast errors, Germany

**Why it matters here:**
This paper directly studies the consequences of renewable forecast errors.

**Conclusion:**
The paper uses OLS, quantile regression, and autoregressive moving-average methods with quarter-hourly data. It finds that higher wind and solar forecast errors increase imbalance volumes and can pass through into higher spot prices. It also finds that wind forecast errors in Germany have a stronger impact on spot prices than solar forecast errors.

**Interpretation for us:**
This is strong evidence that fundamental forecast error is economically meaningful. Therefore, the difference between actual-input and forecast-input model performance is not just a technicality; it is a real trading issue.

---

## 12. Beran, Vogler & Weber — “Multi-Day-Ahead Electricity Price Forecasting: A Comparison of Fundamental, Econometric and Hybrid Models”

**Title:**
Multi-Day-Ahead Electricity Price Forecasting: A Comparison of Fundamental, Econometric and Hybrid Models

**Authors:**
P. Beran, M. Vogler, C. Weber

**Year:**
2021 working paper / later related work

**Link:**
https://ewl.wiwi.uni-due.de/fileadmin/fileupload/BWL-ENERGIE/Arbeitspapiere/RePEc/pdf/wp2102_MultiDayAheadElectricityPriceForecastingAComparisonOfFundamentalEeconometricAndHybridModels.pdf

**Alternative link:**
https://ideas.repec.org/p/dui/wpaper/2102.html

**Field:**
German day-ahead electricity price forecasting

**Why it matters here:**
This paper is very relevant because it explicitly tries to respect realistic information availability and multi-day-ahead horizons.

**Conclusion:**
The paper forecasts German day-ahead electricity prices using hybrids of fundamental and econometric models. It formulates models based on information available on the previous day, including day-ahead forecasts of exogenous variables. It also uses professional forecast provider data for wind, solar, and temperature forecasts where public historical forecasts are limited. It emphasizes information cut-off times and uses only information available at or before the forecast time.

**Interpretation for us:**
This supports using forecasted fundamentals as the main tradable model. It also shows why professional research often becomes complicated: if you go beyond day-ahead, you need forecast data for multiple horizons. For our case study, we can keep it to day-ahead or short prompt aggregation and avoid building a full multi-day fundamental forecast stack.

---

## 13. Lago, Marcjasz, De Schutter & Weron — “Forecasting Day-Ahead Electricity Prices: A Review of State-of-the-Art Algorithms, Best Practices and an Open-Access Benchmark”

**Title:**
Forecasting Day-Ahead Electricity Prices: A Review of State-of-the-Art Algorithms, Best Practices and an Open-Access Benchmark

**Authors:**
Jesus Lago, Grzegorz Marcjasz, Bart De Schutter, Rafał Weron

**Year:**
2021, Applied Energy; arXiv version 2020

**Link:**
https://arxiv.org/abs/2008.08004

**DOI link:**
https://doi.org/10.1016/j.apenergy.2021.116983

**Field:**
Electricity price forecasting review / benchmarking

**Why it matters here:**
This is a broad best-practices paper for day-ahead electricity price forecasting.

**Conclusion:**
The paper argues that many electricity price forecasting studies have weak benchmarking, short test periods, private datasets, inadequate metrics, and insufficient statistical testing. It provides a benchmark and best practices for evaluating models across markets and time periods.

**Interpretation for us:**
For the case study, this supports doing:

* a simple baseline
* an improved model
* walk-forward or blocked validation
* MAE/RMSE
* clean separation of train/test periods
* no leakage
* transparent dataset and reproducible pipeline

---

## 14. Weron — “Electricity Price Forecasting: A Review of the State-of-the-Art with a Look into the Future”

**Title:**
Electricity Price Forecasting: A Review of the State-of-the-Art with a Look into the Future

**Author:**
Rafał Weron

**Year:**
2014

**Likely link / DOI:**
https://doi.org/10.1016/j.ijforecast.2014.08.008

**Field:**
Electricity price forecasting review

**Why it matters here:**
This is a classic review paper in electricity price forecasting.

**Conclusion:**
The paper reviews electricity price forecasting models and discusses the importance of exogenous variables, market fundamentals, seasonality, and probabilistic forecasting. It is often cited as a foundation for the field. Maciejowska et al. cite it as a comprehensive review of spot price forecasting.

**Interpretation for us:**
This supports the general approach of combining price history, seasonality, and fundamentals.

---

# Practical Methodology for the Case Study

## Minimum viable methodology

Because time is limited, do this:

### Market

Germany.

### Target

Hourly German day-ahead electricity price.

### Main features

Use features that are available before the day-ahead auction:

* load forecast
* wind forecast
* solar forecast
* lagged prices
* hour of day
* day of week
* month
* weekend flag
* maybe holidays if easy

### Baseline model

Use one simple benchmark:

```text
last-week-same-hour price
```

or:

```text
average price for same hour and same weekday
```

### Improved model

Use one stronger model:

* Ridge regression
* Lasso regression
* Random forest
* Gradient boosting
* LightGBM / XGBoost if available and not too time-consuming

### Validation

Use time-series validation:

* blocked train/test split
* or walk-forward validation

Do **not** randomly shuffle rows.

### Metrics

Report:

* MAE
* RMSE
* maybe tail metric: MAE on top 10% highest-price hours or spike-hour recall

---

# Recommended Two-Model Design

## Model 1: Tradable model

This is the main submission model.

```text
y = day-ahead price

X =
    load forecast
    wind forecast
    solar forecast
    lagged prices
    calendar features
```

Use this for:

* official model performance
* submission.csv
* prompt curve translation
* long/short/neutral signal

### Why this is main

It avoids leakage and follows standard electricity-price-forecasting practice.

---

## Model 2: Oracle / structural model

This is a secondary comparison.

```text
y = day-ahead price

X =
    actual load
    actual wind
    actual solar
    lagged prices
    calendar features
```

Use this for:

* upper-bound comparison
* explaining how much error comes from fundamental forecast uncertainty
* defending the actual-input argument

### Important caveat

Do **not** present this as the live trading model unless you feed it forecasted values and evaluate it with forecasted values.

---

## Optional Model 3: Structural-trained, forecast-fed model

This directly tests the user’s argument.

Train:

```text
actual fundamentals → price
```

Test/deploy:

```text
forecasted fundamentals → price
```

Compare it with:

```text
forecasted fundamentals → price
```

That gives the exact comparison we wanted:

| Model | Training inputs         | Test inputs             | Meaning                        |
| ----- | ----------------------- | ----------------------- | ------------------------------ |
| A     | actual fundamentals     | actual fundamentals     | oracle upper bound             |
| B     | forecasted fundamentals | forecasted fundamentals | standard tradable MOS-style    |
| C     | actual fundamentals     | forecasted fundamentals | Perfect Prog / user’s approach |

This is the strongest possible design if data allows.

---

# Comparison of Methodologies

## A. Train on actuals, test on actuals

### Meaning

Oracle / perfect foresight.

### Good for

* understanding physics
* upper-bound benchmark
* feature importance
* estimating how much fundamentals explain price

### Bad for

* live trading performance
* avoiding leakage concerns
* official forecast claims

### Verdict

Use as secondary benchmark only.

---

## B. Train on forecasts, test on forecasts

### Meaning

Tradable day-ahead forecast.

### Good for

* live trading realism
* case study evaluation
* avoiding look-ahead bias
* matching training and deployment input distribution

### Bad for

* dependence on forecast provider
* forecast archive availability
* future forecast model changes

### Verdict

Use as main model.

---

## C. Train on actuals, test on forecasts

### Meaning

Perfect Prog / structural model deployed with best available forecasts.

### Good for

* forecast-provider independence
* benefiting from improved future forecasts
* avoiding noisy training inputs
* matching the user’s argument

### Bad for

* train-test distribution mismatch
* forecast errors may hurt live performance
* might underperform MOS-style model if forecasts have systematic biases

### Verdict

Very interesting comparison. Include if time allows.

---

## D. Train on forecasts but enhance/improve fundamentals first

### Meaning

State-of-the-art electricity price forecasting approach.

### Good for

* best accuracy
* accounts for forecast bias
* captures uncertainty better

### Bad for

* more work
* requires modelling load/wind/solar
* beyond scope for this case study

### Verdict

Mention as future work, do not implement.

---

# What to Do Given Limited Time

The practical choice:

1. Build the **tradable forecast-input model** as the main model.
2. Add the **actual-input oracle model** if data collection allows.
3. Do **not** build separate wind/load/solar forecasting models.
4. Explain that improving fundamentals is out of scope.
5. Use public forecasts directly.
6. State that actual fundamentals are only used as an upper-bound structural benchmark.

---

# Strong Final Position

The answer should be:

> The literature standard for tradable day-ahead electricity price forecasting is to use ex-ante fundamental forecasts, because the model must respect the information set available before the auction. However, training on realised fundamentals is also defensible as a structural Perfect Prog model. The best methodology is to report both: a forecast-input model for realistic trading performance and an actual-input model as an oracle upper bound. If time allows, also test the user’s preferred Perfect Prog deployment by training on actuals but evaluating with forecasted inputs. The performance gap between these variants measures the cost of fundamental forecast error and the train-live mismatch.

---

# How This Applies to the Case Study

For the case study, use this exact framing:

```text
I build a German hourly day-ahead price forecasting prototype.

The primary model is tradable and uses only ex-ante inputs:
load forecast, wind forecast, solar forecast, lagged prices, and calendar features.

As a robustness check, I also estimate a perfect-foresight structural model using realised load, wind, and solar. This is not used as the live trading model; it is an upper-bound benchmark.

This design follows the electricity price forecasting literature, while also testing the Perfect Prog argument from meteorological postprocessing: whether a model trained on clean actual fundamentals can be used with external forecasts at deployment time.

The model does not attempt to improve load, wind, or solar forecasts. Those are treated as external public inputs. Improving fundamental forecasts is left as future work.
```

---

# Very Short Version for Future Session

We debated whether to train the electricity price model on actual realised fundamentals or forecasted fundamentals. The user argued for training on actuals because it learns the true structural price relationship and remains independent of any specific forecast provider; then at prediction time the model can be fed the best available forecasts. This is known in weather forecasting as **Perfect Prog**. The standard tradable approach is to train on forecasted fundamentals and deploy on forecasted fundamentals, called **MOS-style** or ex-ante modelling. The literature does not give a universal winner: MOS often performs better because it learns forecast bias/error, but Perfect Prog is easier to train, less tied to one model provider, and can work well for short horizons. For the case study, the best solution is to use the forecast-input model as the main tradable model, and optionally report an actual-input model as an oracle/structural upper bound. If time allows, test three variants: actual→actual, forecast→forecast, and actual→forecast.

# Handover Summary: Modelling Methodology for European Power Day-Ahead Price Forecasting

## 0. Project Context

We are building a prototype for the European power fair-value case study. The likely market is **Germany**, and the target is **hourly day-ahead electricity prices**. The model should use public fundamentals such as **load forecast, wind forecast, solar forecast, lagged prices, and calendar features**.

The modelling discussion focused on:

1. **How much data to pull**
2. **Which model classes to use**
3. **Whether to use neural networks, kernel models, regularised regression, or boosted trees**
4. **How LEAR works**
5. **Whether LEAR should be 24 separate hourly models**
6. **Whether to transform the target using `asinh(price)`**
7. **Whether K-fold cross-validation is useful**
8. **How to validate without leakage**
9. **How to justify all choices using papers**

The final chosen modelling structure is:

```text
M0: Seasonal naïve baseline
M1: LEAR-style regularised ARX model
M2: Boosted tree model, preferably LightGBM or sklearn HistGradientBoostingRegressor
No M3 probabilistic layer
```

---

# 1. How Much Data Should We Pull?

## Decision

Use around **5 years of hourly data**, probably:

```text
2021-01-01 to 2025-12-31
```

That gives roughly:

```text
5 × 8760 ≈ 43,800 hourly rows
```

## Why not less?

Two years is enough for a prototype but weak for serious validation. It may not cover enough seasons, price regimes, renewable patterns, and volatility episodes.

## Why not much more?

Very old electricity-market data may not be representative. The European power market changed strongly after the 2021–2022 energy crisis, and Germany’s generation mix has also changed. More data is not always better if the regime changed.

## Justification

Lago et al. emphasize that electricity price forecasting papers often suffer from short test periods, private datasets, and weak benchmarking. They argue for rigorous evaluation across longer and properly structured datasets.

Paper:

**Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark**
Authors: Jesus Lago, Grzegorz Marcjasz, Bart De Schutter, Rafał Weron
Applied Energy, 2021
Link: https://arxiv.org/abs/2008.08004
DOI / published link: https://doi.org/10.1016/j.apenergy.2021.116983

Conclusion for our project:

> Use enough data for real time-series validation, but do not blindly pull 10–15 years if the market regime has changed. A 2021–2025 dataset is a good compromise.

---

# 2. Baseline Model M0

## Decision

Use a simple seasonal naïve baseline:

```text
ŷ_t = y_{t-168}
```

Meaning:

> Predict this hour using the same hour from one week ago.

Example:

```text
Forecast Monday 17:00 = actual price from previous Monday 17:00
```

## Why this baseline?

Electricity prices have strong weekly seasonality. A last-week-same-hour baseline is simple, transparent, and hard enough to beat that it is meaningful.

## Justification

The case study requires at least one baseline. Lago et al. also emphasize that new models should be compared against simple but strong benchmarks rather than weak toy baselines.

Paper:

**Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark**
Link: https://arxiv.org/abs/2008.08004

Conclusion for our project:

> M0 gives a fair benchmark. If M1/M2 cannot beat last-week-same-hour, the model is not useful.

---

# 3. Improved Model M1: LEAR-Style Regularised ARX

## Decision

Use a self-coded **LEAR-style regularised ARX model**.

LEAR means roughly:

```text
LASSO-Estimated AutoRegressive model
```

or:

```text
parameter-rich ARX model estimated with LASSO
```

ARX means:

```text
AutoRegressive with eXogenous variables
```

So the model uses:

* lagged prices
* load forecast
* wind forecast
* solar forecast
* residual load
* calendar features
* regularisation

## Main LEAR sources

### Source 1: epftoolbox LEAR documentation

Title:

**LEAR — epftoolbox documentation**

Link: https://epftoolbox.readthedocs.io/en/latest/modules/lear_model.html

Key conclusion:

> The documentation describes LEAR as a parameter-rich ARX model estimated using LASSO as implicit feature selection.

### Source 2: Ziel 2016

Title:

**Forecasting Electricity Spot Prices using Lasso: On Capturing the Autoregressive Intraday Structure**

Author: Florian Ziel
Link: https://arxiv.org/abs/1509.01966

Key conclusion:

> Ziel presents a regression-based day-ahead electricity spot price model estimated by LASSO. LASSO allows many possible regressors while shrinking and sparsifying coefficients to avoid overfitting and capture autoregressive intraday dependence.

### Source 3: Lago et al. 2021 / epftoolbox

Title:

**Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark**

Link: https://arxiv.org/abs/2008.08004
Toolbox: https://github.com/jeslago/epftoolbox

Key conclusion:

> The open benchmark includes LEAR and DNN as state-of-the-art reference models. This makes LEAR a good literature-aligned model for the case study.

---

# 4. Why LEAR Uses 24 Separate Hourly Models

## Question

Why should LEAR have 24 separate hourly models? Why not just one model?

## Answer

Day-ahead electricity prices are naturally a **24-dimensional daily object**:

```text
For each delivery day d:
price_d,00
price_d,01
...
price_d,23
```

Each hour behaves differently.

Examples:

```text
03:00 price: night demand, wind, baseload generation
12:00 price: solar-heavy, midday demand
18:00 price: evening peak, fading solar, high residual load
```

A linear model cannot automatically learn very different hour-specific relationships unless we either:

1. train **24 separate models**, or
2. use one pooled model with many hour interaction terms.

LEAR usually uses the first approach.

## Mathematical structure

For each hour `h`, estimate a separate regression:

```text
price_{d,h} = β_{h,0} + β_h · X_d + ε_{d,h}
```

So:

```text
price_{d,0}  = β_0  · X_d + error
price_{d,1}  = β_1  · X_d + error
...
price_{d,23} = β_23 · X_d + error
```

Each hour has its own coefficient vector.

## Why this helps

The model can learn:

```text
solar matters strongly at 12:00
solar matters almost zero at 03:00
load matters strongly in evening peak
wind may matter differently at night than during day
```

## Implementation idea

```python
models = {}

for h in range(24):
    y_h = Y_train[:, h]
    model_h = Lasso(alpha=alpha_h)
    model_h.fit(X_train, y_h)
    models[h] = model_h

forecast_24h = np.array([
    models[h].predict(X_next_day)[0]
    for h in range(24)
])
```

## Key source

The epftoolbox LEAR implementation uses daily matrices with 24 hourly targets and loops over the 24 hours.

Source:

epftoolbox LEAR docs:
https://epftoolbox.readthedocs.io/en/latest/modules/lear_model.html

epftoolbox GitHub:
https://github.com/jeslago/epftoolbox

---

# 5. Do We Need 24 Models for M2 Too?

## Decision

No.

Use:

```text
M1: 24 separate LEAR-style hourly models
M2: one pooled boosted-tree hourly model
```

## Why?

Boosted trees can learn nonlinear interactions between hour-of-day and fundamentals. A pooled M2 can have one row per hour:

```text
one row = one delivery hour
target = hourly day-ahead price
features = hour, weekday, load forecast, wind forecast, solar forecast, lags, etc.
```

The boosted tree can learn relationships like:

```text
if hour = 12 and solar high → lower price
if hour = 18 and residual load high → higher price
if weekend and wind high → possible negative price
```

A linear model needs explicit structure or interactions. A boosted-tree model can learn many of these interactions automatically.

## Justification

Ziel & Weron studied the question of **univariate vs multivariate modelling frameworks** in day-ahead electricity price forecasting. They found that multivariate frameworks have a small overall edge but do not uniformly dominate univariate models across all datasets, seasons, or hours. Sometimes univariate models are better.

Paper:

**Day-ahead electricity price forecasting with high-dimensional structures: Univariate vs. multivariate modeling frameworks**
Authors: Florian Ziel, Rafał Weron
Link: https://arxiv.org/abs/1805.06649
Published link: https://doi.org/10.1016/j.eneco.2017.12.016

Conclusion for our project:

> It is defensible to use a 24-hour LEAR-style structure for M1 and a pooled nonlinear model for M2. There is no universal rule that one structure dominates.

---

# 6. LEAR Mathematical Grounding

## Objective function

A simple LASSO version solves:

```text
min_β  Σ_t (y_t - x_t β)^2 + λ Σ_j |β_j|
```

The first term fits the data.
The second term penalizes large/unused coefficients.

## Why LASSO?

Because electricity price forecasting can have many possible regressors:

* price lags
* all 24 prices from previous day
* all 24 prices from previous week
* load
* wind
* solar
* residual load
* weekday effects
* holiday effects
* seasonality
* interactions

Without regularisation, this can overfit.

LASSO helps by shrinking many coefficients to exactly zero.

## Why maybe ElasticNet instead of pure LASSO?

Pure LASSO can be unstable when features are correlated. In power markets, many features are correlated:

```text
load, residual load, wind share, solar share, hour, season
```

ElasticNet uses both L1 and L2 penalties:

```text
min_β  Σ_t (y_t - x_t β)^2
      + λ [ α Σ_j |β_j| + (1 - α) Σ_j β_j² ]
```

This is more stable when regressors are correlated.

## Supporting paper

Title:

**Regularization for electricity price forecasting**
Author: Bartosz Uniejewski
Link: https://arxiv.org/abs/2404.03968

Key conclusion:

> This study compares different regularisation penalties for electricity price forecasting and finds that alternatives such as elastic net can be competitive or better than plain LASSO. It also reports that cross-validation can work well for parameter optimization.

Conclusion for our project:

> Use a LEAR-style model but implement it as ElasticNet or compare LASSO vs ElasticNet. This is defensible and not merely arbitrary.

---

# 7. M1 Feature Design

## Daily LEAR-style data format

For M1, make one row per delivery day.

Target:

```text
Y_d = [price_d,00, price_d,01, ..., price_d,23]
```

Features could include:

```text
previous day 24 prices:
price_{d-1,00}, ..., price_{d-1,23}

previous two-day 24 prices:
price_{d-2,00}, ..., price_{d-2,23}

previous week 24 prices:
price_{d-7,00}, ..., price_{d-7,23}

fundamental forecasts:
load_forecast_{d,h}
wind_forecast_{d,h}
solar_forecast_{d,h}
residual_load_{d,h}

calendar:
weekday dummies
weekend flag
month or seasonal Fourier terms
holiday dummy if available
```

Then train 24 models:

```text
model_h: X_d → price_{d,h}
```

## Why include all 24 previous-day prices?

Because electricity prices have strong intraday dependence. The price at 18:00 today may depend on the whole previous-day curve, not only yesterday 18:00.

## Supporting paper

Title:

**Forecasting Electricity Spot Prices using Lasso: On Capturing the Autoregressive Intraday Structure**
Author: Florian Ziel
Link: https://arxiv.org/abs/1509.01966

Conclusion for our project:

> Use a rich lag structure and regularisation. Do not just use `price_lag_24` alone if implementing a LEAR-style model.

---

# 8. Target Transformation: Raw Price vs `asinh(price)`

## Question

If raw price can be modelled linearly, how can `asinh(price)` also be modelled linearly? Both cannot be exactly true.

## Answer

Correct. Both cannot be exactly linearly true under the same features.

Using `asinh(price)` is not a statement that the true relationship is mathematically linear after transformation. It is an empirical modelling choice.

Electricity prices are:

* spiky
* heavy-tailed
* sometimes negative
* volatile

`log(price)` is problematic because power prices can be negative. But `asinh(price)` works for positive and negative values.

Properties:

```text
for small y:  asinh(y) ≈ y
for large y:  asinh(y) ≈ log(2y)
```

So it compresses large price spikes while preserving sign.

## Decision

Do **not** blindly use `asinh`.

Test both:

```text
M1_raw:    LEAR-style model on y
M1_asinh:  LEAR-style model on asinh(y)
```

Choose using blocked/rolling validation.

## Supporting paper

Title:

**Efficient Forecasting of Electricity Spot Prices with Expert and LASSO Models**
Authors: Bartosz Uniejewski, Rafał Weron
Link: https://www.mdpi.com/1996-1073/11/8/2039

Key conclusion:

> The paper finds that a complex LASSO-style model with nearly 400 explanatory variables, a well-chosen variance-stabilising transformation such as `asinh` or N-PIT, and frequent recalibration can improve electricity price forecast accuracy.

Conclusion for our project:

> `asinh` is literature-supported, but should be treated as a hyperparameter / modelling variant, not assumed to be correct.

---

# 9. Improved Model M2: Boosted Trees

## Question

Which boosted model should we use?

Candidate models:

```text
GradientBoostingRegressor
HistGradientBoostingRegressor
XGBoost
LightGBM
CatBoost
```

## Decision

Use:

```text
Preferred: LightGBMRegressor
Fallback: sklearn HistGradientBoostingRegressor
```

## Why not classic GradientBoostingRegressor?

It is older and slower. Histogram-based methods are usually better for medium/large tabular data.

## Why LightGBM?

LightGBM is fast, strong on tabular data, and performs well in recent day-ahead electricity price forecasting studies.

## Why HistGradientBoostingRegressor fallback?

It is included in scikit-learn, has fewer dependency issues, is fast for datasets with more than 10,000 samples, and supports missing values.

## Supporting papers / sources

### Source 1: XGBoost paper

Title:

**Forecasting the clearing price in the day-ahead spot market using eXtreme Gradient Boosting**

Authors: H. Xie, S. Chen, C. Lai, G. Ma, W. Huang
Published in Electrical Engineering, 2022
Link: https://link.springer.com/article/10.1007/s00202-021-01410-6
Alternative summary: https://www.semanticscholar.org/paper/Forecasting-the-clearing-price-in-the-day-ahead-Xie-Chen/606a36a1799e9818e8fcc2513b9410659c276758

Conclusion:

> This paper uses XGBoost for day-ahead market price forecasting and combines it with feature engineering / interpretability. It proves that boosted trees are used in electricity price forecasting literature.

### Source 2: Recent LightGBM/XGBoost/CatBoost comparison

Title:

**Data-driven Day Ahead Market Prices Forecasting: A Focus on Short Training Set Windows**

Link: https://arxiv.org/html/2506.10536v1

Conclusion:

> The study compares LSTM, XGBoost, LightGBM, and CatBoost across European day-ahead markets and uses ENTSO-E forecast-derived features. It reports strong results for LightGBM and supports boosted-tree models for short-term electricity price forecasting.

### Source 3: sklearn documentation

Title:

**HistGradientBoostingRegressor documentation**

Link: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html

Conclusion:

> Good practical fallback with fewer dependencies.

Conclusion for our project:

> M2 should be LightGBM if dependencies are acceptable. Otherwise use HistGradientBoostingRegressor.

---

# 10. Why Not a Neural Network?

## Question

Should we use a fully connected neural network or LSTM to make the project more impressive?

## Decision

Do not use a neural network as the main model.

## Why?

A neural net creates many extra choices:

```text
number of layers
hidden units
activation
learning rate
optimizer
batch size
dropout
weight decay
early stopping
normalisation
random seeds
```

This increases complexity but not necessarily credibility for a short case study.

## Literature

Neural networks are used in electricity price forecasting.

Paper:

**Forecasting spot electricity prices: Deep learning approaches and empirical comparison of traditional algorithms**
Authors: Jesus Lago, Fjo De Ridder, Bart De Schutter
Applied Energy, 2018
Link: https://www.sciencedirect.com/science/article/pii/S030626191830196X

Conclusion:

> The paper proposes DNN, LSTM-DNN, GRU-DNN, and CNN-based approaches and finds that deep learning can improve predictive accuracy in their benchmark setup.

But for our project:

> A neural net is legitimate but not necessary. LEAR-style regularisation plus boosted trees is more explainable and easier to defend.

---

# 11. Why Not Kernel Regression / SVR?

## Question

Could we use kernel mapping or SVR instead of boosted trees?

## Decision

Do not use kernel methods as the main model.

## Why?

With 5 years of hourly data:

```text
≈ 43,800 rows
```

Kernel methods can become computationally heavy and require annoying tuning:

```text
kernel type
gamma
C
epsilon
scaling
```

They are less transparent than LEAR and less practical than boosted trees.

Conclusion:

> Kernel methods are mathematically interesting but not the best use of time here.

---

# 12. K-Fold Cross-Validation

## Question

Is K-fold cross-validation useful here?

## Answer

Yes, but only for:

```text
model selection
hyperparameter tuning
robustness checking
```

It does not magically improve the final model unless used to choose better hyperparameters.

## Important warning

Do **not** use ordinary random K-fold.

Random K-fold leaks future information into the past because it shuffles time. This is not suitable for time series.

## Decision

Use:

```text
rolling-origin blocked cross-validation
```

or:

```text
fixed-length rolling window validation
```

Example:

```text
Fold 1:
Train: 2021-01 to 2022-12
Test:  2023-01

Fold 2:
Train: 2021-02 to 2023-01
Test:  2023-02

Fold 3:
Train: 2021-03 to 2023-02
Test:  2023-03
```

Every fold has the same training length:

```text
24 months train → 1 month test
```

This avoids the problem that early folds are undertrained and late folds are overtrained.

## Supporting paper

Lago et al. emphasize rigorous evaluation, proper benchmarking, and avoiding weak evaluation methods.

Paper:

**Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark**
Link: https://arxiv.org/abs/2008.08004

Supporting recent regularisation paper:

**Regularization for electricity price forecasting**
Link: https://arxiv.org/abs/2404.03968

This paper reports that cross-validation can be effective for parameter optimization in electricity price forecasting.

Conclusion for our project:

> Use time-series cross-validation for hyperparameter selection, not random K-fold.

---

# 13. Blocked Validation Concern

## User concern

Blocked validation can seem weak because:

```text
early folds may undertrain
late folds may overtrain
results can vary a lot by period
```

## Answer

This criticism is correct for badly designed expanding-window validation.

Bad example:

```text
Fold 1: train 1 month, test next month
Fold 2: train 2 months, test next month
Fold 3: train 3 months, test next month
```

Better design:

```text
fixed-length rolling training window
```

Example:

```text
Train always = previous 24 months
Test always = next 1 month
```

This answers a realistic trading question:

> If I retrain every month using the last 24 months of data, how well does the model perform next month?

## Decision

Use:

```text
rolling 24-month training window
1-month test block
```

Then use a final untouched test set, e.g.:

```text
Final test: 2025
```

or:

```text
Final test: last 6 months of 2025
```

Conclusion:

> Rolling fixed-window validation is stronger and more realistic than random K-fold or naive blocked validation.

---

# 14. Actual Inputs vs Forecasted Inputs

This was discussed earlier but remains important.

## Main tradable model

Use forecasted fundamentals available before the day-ahead auction:

```text
load forecast
wind forecast
solar forecast
```

This avoids leakage.

## Optional structural/oracle model

Use actual realised fundamentals as an upper-bound benchmark:

```text
actual load
actual wind
actual solar
```

This tests how much price could be explained if fundamentals were known perfectly.

## Three possible comparisons

```text
A: forecasted inputs → forecasted inputs
   Main tradable model

B: actual inputs → actual inputs
   Oracle / perfect-foresight benchmark

C: actual inputs → forecasted inputs
   Perfect Prog / structural model deployed with forecasts
```

## Supporting methodology papers

### MOS vs Perfect Prog

Title:

**MOS, Perfect Prog, and Reanalysis**

Authors: C. Marzban, S. Sandgathe, E. Kalnay
Link: https://faculty.washington.edu/marzban/mos.pdf
Alternative: https://journals.ametsoc.org/view/journals/mwre/134/2/mwr3088.1.xml

Conclusion:

> Perfect Prog trains on observed/actual predictors and deploys with model forecasts. MOS trains directly on model forecasts. This maps to our actual-vs-forecasted-input debate.

### Forecasted fundamentals matter

Title:

**Enhancing Load, Wind and Solar Generation for Day-Ahead Forecasting of Electricity Prices**

Authors: Katarzyna Maciejowska, Weronika Nitka, Tomasz Weron
Link: https://www.sciencedirect.com/science/article/abs/pii/S014098832100178X
Alternative: https://ideas.repec.org/a/eee/eneeco/v99y2021ics014098832100178x.html

Conclusion:

> TSO forecasts of load, wind, and solar can be biased. Improving these forecasts improves day-ahead and intraday price forecasting. This proves that input forecast quality matters.

Conclusion for our project:

> Main model should use ex-ante forecasted fundamentals. Actuals can be included only as an oracle/structural comparison.

---

# 15. Final Proposed Model Stack

## M0: Seasonal naive

```text
ŷ_t = y_{t-168}
```

Purpose:

```text
baseline
```

## M1: LEAR-style ElasticNet/LASSO ARX

Structure:

```text
24 separate hourly models
one model per delivery hour
```

Features:

```text
previous day 24 prices
previous two-day 24 prices
previous week 24 prices
load forecast
wind forecast
solar forecast
residual load
weekday/month/holiday/seasonality
```

Target:

```text
hourly day-ahead price
```

Variants:

```text
raw y
asinh(y)
```

Choose via rolling validation.

## M2: Boosted tree model

Preferred:

```text
LightGBMRegressor
```

Fallback:

```text
HistGradientBoostingRegressor
```

Structure:

```text
one pooled hourly model
one row per hour
hour feature included
```

Features:

```text
price_lag_24
price_lag_48
price_lag_168
rolling_mean_24
rolling_mean_168
load_forecast
wind_forecast
solar_forecast
residual_load
wind_share
solar_share
hour
weekday
month
weekend
holiday maybe
```

---

# 16. Final Validation Plan

## Data period

```text
2021-01-01 to 2025-12-31
```

## Model selection

Use rolling validation:

```text
Train: previous 24 months
Test: next 1 month
```

Use it to choose:

```text
M1 raw vs asinh
M1 LASSO vs ElasticNet
M1 alpha/l1_ratio
M2 LightGBM parameters
possibly training-window length
```

## Final holdout

Use a final untouched period:

```text
2025 full year
```

or if too large:

```text
2025-H2
```

## Metrics

Report:

```text
MAE
RMSE
tail MAE on top 10% price hours
maybe negative-price-hour MAE
```

The task asks for MAE/RMSE and a tail metric if modelling extremes.

---

# 17. What to Say in the Report

A strong report paragraph:

```text
The forecasting stack contains one simple seasonal-naive benchmark, one literature-aligned LEAR-style regularised ARX model, and one nonlinear boosted-tree model. The LEAR-style model follows the electricity-price-forecasting literature by estimating hour-specific regularised regressions over a parameter-rich set of autoregressive, calendar, and fundamental regressors. The boosted-tree model is estimated as a pooled hourly model with hour-of-day and fundamental features, allowing nonlinear interactions between residual load, renewable generation, and time of day. Hyperparameters and target transformations are selected using rolling-origin blocked validation, avoiding random K-fold leakage. The final model is evaluated on an untouched chronological test set.
```

Another strong paragraph for the actual-vs-forecasted input issue:

```text
The primary tradable model uses ex-ante load, wind, and solar forecasts available before the day-ahead auction. A realised-fundamentals specification can be reported separately as a perfect-foresight benchmark. This separates price-response error from fundamental-forecast error and avoids overstating live trading performance.
```

---

# 18. Final Decisions and Paper Support

## Decision 1: Use LEAR-style M1

Supported by:

* Ziel 2016: https://arxiv.org/abs/1509.01966
* epftoolbox docs: https://epftoolbox.readthedocs.io/en/latest/modules/lear_model.html
* Lago et al. 2021: https://arxiv.org/abs/2008.08004

Reason:

> LEAR/LASSO is a recognised state-of-the-art statistical benchmark in electricity price forecasting.

## Decision 2: Use 24 separate hourly models for M1

Supported by:

* epftoolbox LEAR implementation/docs
* Ziel & Weron 2018: https://arxiv.org/abs/1805.06649

Reason:

> Hourly prices have different dynamics. Univariate/hour-specific structures are defensible and sometimes competitive with multivariate structures.

## Decision 3: Use LightGBM or HistGradientBoostingRegressor for M2

Supported by:

* XGBoost EPF paper: https://link.springer.com/article/10.1007/s00202-021-01410-6
* Recent LightGBM/XGBoost/CatBoost paper: https://arxiv.org/html/2506.10536v1
* sklearn HistGradientBoostingRegressor docs: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html

Reason:

> Boosted trees are strong nonlinear tabular models and are used in electricity price forecasting literature.

## Decision 4: Do not make neural nets the main model

Supported by:

* Lago, De Ridder & De Schutter 2018: https://www.sciencedirect.com/science/article/pii/S030626191830196X
* Lago et al. 2021: https://arxiv.org/abs/2008.08004

Reason:

> Deep learning is valid but adds tuning complexity. For this case study, LEAR + boosted trees gives a stronger trade-off between performance, interpretability, and delivery.

## Decision 5: Test raw y vs asinh(y)

Supported by:

* Uniejewski & Weron 2018: https://www.mdpi.com/1996-1073/11/8/2039

Reason:

> `asinh` is a literature-supported variance-stabilising transformation for electricity prices, but it is an empirical modelling choice and should be selected by validation.

## Decision 6: Use rolling blocked validation, not random K-fold

Supported by:

* Lago et al. 2021: https://arxiv.org/abs/2008.08004
* Uniejewski 2024 regularisation paper: https://arxiv.org/abs/2404.03968

Reason:

> Time-series validation avoids leakage. Cross-validation is useful for hyperparameter selection only if it respects chronology.

## Decision 7: Use forecasted fundamentals as main inputs

Supported by:

* Maciejowska, Nitka & Weron 2021: https://www.sciencedirect.com/science/article/abs/pii/S014098832100178X
* Uniejewski & Ziel 2025: https://arxiv.org/abs/2501.06180
* Marzban et al. MOS vs Perfect Prog: https://faculty.washington.edu/marzban/mos.pdf

Reason:

> Tradable forecasts must use information available before the auction. Actual fundamentals can be used only as an oracle benchmark.

---

# 19. Final Short Version

The modelling plan is:

```text
Use Germany hourly data from 2021–2025.

M0:
    last-week-same-hour baseline.

M1:
    self-coded LEAR-style regularised ARX model.
    24 separate hourly LASSO/ElasticNet models.
    Use rich lag features, fundamentals, and calendar features.
    Test raw price vs asinh(price).

M2:
    pooled hourly LightGBM model.
    Fallback: sklearn HistGradientBoostingRegressor.
    Use lagged prices, forecasted fundamentals, residual load, renewable shares, and calendar features.

Validation:
    no random K-fold.
    use rolling-origin blocked validation:
        train previous 24 months
        test next 1 month.
    final untouched holdout in 2025.

Main tradable inputs:
    forecasted load, wind, solar available before day-ahead auction.

Optional robustness:
    actual-fundamentals oracle model to estimate perfect-foresight upper bound.
```

This is mathematically interesting but still doable within the case-study deadline.
