"""Shared modelling settings."""

TARGET_COLUMN = "day_ahead_price_eur_per_mwh"
TIMESTAMP_COLUMN = "timestamp_utc"

TRAIN_MONTHS = 24
TEST_MONTHS = 1
STEP_MONTHS = 3

SCARCITY_PRICE_THRESHOLD_EUR_PER_MWH = 150.0
LEAR_ALPHA_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 1.5, 1.8, 2.0]
LEAR_L1_RATIO_GRID = [0.2, 0.5, 0.8]

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
    "negative_price_mae",
    "scarcity_price_mae",
]
