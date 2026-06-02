"""Settings for translating hourly forecasts into prompt-curve views.

This file has no public functions. It stores desk-view thresholds, benchmark
settings, default model choices, and German local peakload definitions used by
the curve-translation helpers.
"""

PEAK_START_HOUR = 8
PEAK_END_HOUR = 20
MARKET_TIMEZONE = "Europe/Berlin"
TRAILING_BENCHMARK_DAYS = 30

# When no empirical P10-P90 band is available, the signal falls back to a
# symmetric band of max(MAE, TAIL_RISK_WEIGHT * tail_error) around the forecast.
TAIL_RISK_WEIGHT = 0.5
MIN_PREDICTION_COVERAGE = 0.99

DEFAULT_MODEL = "lear_model"
DEFAULT_REGULARIZATION = "lasso"
DEFAULT_TARGET_TRANSFORM = "raw"
