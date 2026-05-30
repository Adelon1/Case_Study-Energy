"""Combine standardized per-dataset CSV files into one hourly model table."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pipeline_helpers.entsoe_data import constants


RAW_FORECAST_INPUT_COLUMNS = [
    "load_forecast_mw",
    "solar_forecast_mw",
    "wind_onshore_forecast_mw",
    "wind_offshore_forecast_mw",
]


@dataclass(frozen=True)
class ImputationReport:
    """Counts from the leakage-safe missing-value fill step."""

    columns: list[str]
    missing_before: dict[str, int]
    filled_from_previous_day: dict[str, int]
    missing_after_fill: dict[str, int]
    dropped_rows_after_fill: int


@dataclass(frozen=True)
class CombinedDataset:
    """Path and in-memory table produced by the combine stage."""

    path: Path
    table: pd.DataFrame
    imputation_report: ImputationReport


def read_standardized_dataset_csv(path: str | Path) -> pd.DataFrame:
    """Read a parsed dataset CSV and normalize its timestamp column."""

    table = pd.read_csv(path)
    table["timestamp_utc"] = pd.to_datetime(table["timestamp_utc"], utc=True)
    return table


def combine_interim_csvs(interim_folder: str | Path, datasets: list[str]) -> pd.DataFrame:
    """Outer-join parsed CSV files by ``timestamp_utc``."""

    interim_folder = Path(interim_folder)
    tables = []
    for dataset in datasets:
        path = interim_folder / f"{dataset}.csv"
        table = read_standardized_dataset_csv(path)
        expected_column = constants.get_entsoe_dataset_request(dataset).output_column
        if expected_column not in table.columns:
            raise ValueError(f"Expected column '{expected_column}' missing from {path}")
        tables.append(table)

    combined = tables[0]
    for table in tables[1:]:
        combined = pd.merge(combined, table, on="timestamp_utc", how="outer")

    combined = combined.sort_values("timestamp_utc").reset_index(drop=True)
    combined.insert(
        1,
        "timestamp_local",
        combined["timestamp_utc"].dt.tz_convert(constants.GERMANY_MARKET_TIMEZONE),
    )
    return combined


def filter_by_utc_window(
    table: pd.DataFrame,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> pd.DataFrame:
    """Keep only rows inside the requested half-open UTC window."""

    mask = (table["timestamp_utc"] >= start_utc) & (table["timestamp_utc"] < end_utc)
    return table.loc[mask].reset_index(drop=True)


def aggregate_to_hourly(table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate any sub-hourly input table to hourly means.

    The ENTSO-E API can return different resolutions such as PT15M, PT30M, or
    PT60M. The modelling dataset should be hourly, so all numeric dataset
    columns are grouped by the UTC hour and averaged. If input data is already
    hourly, this keeps one value per hour unchanged.
    """

    value_columns = [
        column for column in table.columns if column not in {"timestamp_utc", "timestamp_local"}
    ]
    hourly = table.copy()
    hourly["timestamp_utc"] = hourly["timestamp_utc"].dt.floor("h")
    hourly = (
        hourly.groupby("timestamp_utc", as_index=False)[value_columns]
        .mean()
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )
    hourly.insert(
        1,
        "timestamp_local",
        hourly["timestamp_utc"].dt.tz_convert(constants.GERMANY_MARKET_TIMEZONE),
    )
    return hourly


def fill_missing_forecasts_from_previous_day(
    table: pd.DataFrame,
) -> tuple[pd.DataFrame, ImputationReport]:
    """Fill missing forecast driver values using only the same hour from the previous day.

    ENTSO-E can occasionally miss a full day for one forecast series while other
    series are available. This fill rule is intentionally simple and
    leakage-safe: a missing value at timestamp ``t`` may only use the same
    column at ``t - 24h``. Remaining rows with missing forecast drivers are
    dropped before feature engineering.
    """

    filled = table.sort_values("timestamp_utc").reset_index(drop=True).copy()
    columns = [column for column in RAW_FORECAST_INPUT_COLUMNS if column in filled.columns]

    if not columns:
        return filled, ImputationReport(
            columns=[],
            missing_before={},
            filled_from_previous_day={},
            missing_after_fill={},
            dropped_rows_after_fill=0,
        )

    missing_before = filled[columns].isna().sum().astype(int).to_dict()
    filled_from_previous_day: dict[str, int] = {}

    for column in columns:
        missing_mask = filled[column].isna()
        filled[column] = filled[column].fillna(filled[column].shift(24))
        filled_from_previous_day[column] = int((missing_mask & filled[column].notna()).sum())

    missing_after_fill = filled[columns].isna().sum().astype(int).to_dict()
    rows_with_remaining_missing = filled[columns].isna().any(axis=1)
    dropped_rows_after_fill = int(rows_with_remaining_missing.sum())

    if dropped_rows_after_fill:
        filled = filled.loc[~rows_with_remaining_missing].reset_index(drop=True)

    report = ImputationReport(
        columns=columns,
        missing_before=missing_before,
        filled_from_previous_day=filled_from_previous_day,
        missing_after_fill=missing_after_fill,
        dropped_rows_after_fill=dropped_rows_after_fill,
    )
    return filled, report


def write_combined_dataset(
    interim_folder: str | Path,
    processed_folder: str | Path,
    datasets: list[str],
    start_utc: pd.Timestamp | None = None,
    end_utc: pd.Timestamp | None = None,
    output_filename: str = "germany_model_dataset.csv",
) -> CombinedDataset:
    """Combine parsed dataset CSVs and write one processed model dataset."""

    processed_folder = Path(processed_folder)
    processed_folder.mkdir(parents=True, exist_ok=True)
    combined = combine_interim_csvs(interim_folder, datasets)
    if start_utc is not None and end_utc is not None:
        combined = filter_by_utc_window(combined, start_utc, end_utc)
    combined = aggregate_to_hourly(combined)
    combined, imputation_report = fill_missing_forecasts_from_previous_day(combined)
    output_path = processed_folder / output_filename
    combined.to_csv(output_path, index=False)
    return CombinedDataset(path=output_path, table=combined, imputation_report=imputation_report)
