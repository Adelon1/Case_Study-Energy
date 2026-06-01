"""Shared modelling settings."""

TARGET_COLUMN = "day_ahead_price_eur_per_mwh"
TIMESTAMP_COLUMN = "timestamp_utc"

# Rolling validation window, measured in calendar months.
TRAIN_MONTHS = 24
TEST_MONTHS = 1
STEP_MONTHS = 1

# A fold must predict at least this share of its test hours to count as valid.
MIN_PREDICTION_COVERAGE = 0.99

# Prices at or above this level are reported as scarcity hours when measuring
# stress-period accuracy.
SCARCITY_PRICE_THRESHOLD_EUR_PER_MWH = 150.0

# LEAR regularisation search. Each hourly model picks its own penalty by
# time-series cross-validation, so peak and night hours regularise independently
# instead of sharing one global strength.
LEAR_CV_SPLITS = 5
LEAR_ALPHA_PATH_LENGTH = 50            # alphas tried along the Lasso/ElasticNet path
LEAR_L1_RATIO_GRID = [0.2, 0.5, 0.8]  # ElasticNet mix of L1 vs L2
RIDGE_ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]

# Model selection: MAE first, with stress metrics as tie-breakers.
MODEL_SELECTION_METRICS = [
    "mae",
    "top_decile_mae",
    "bottom_decile_mae",
    "rmse",
]

METRIC_NAMES = [
    "mae",
    "rmse",
    "bias",
    "top_decile_mae",
    "bottom_decile_mae",
    "scarcity_price_mae",
]

# Forecast bands come from out-of-sample residual quantiles per delivery hour.
# P10/P90 gives an 80% empirical prediction interval around each point forecast.
BAND_LOWER_QUANTILE = 0.10
BAND_UPPER_QUANTILE = 0.90
