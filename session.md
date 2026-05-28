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