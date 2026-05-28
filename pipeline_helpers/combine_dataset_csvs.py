"""Combine standardized per-dataset CSV files into one hourly model table."""

from __future__ import annotations

from functools import reduce
from pathlib import Path

import pandas as pd

from pipeline_helpers.constants import GERMANY_MARKET_TIMEZONE, get_entsoe_dataset_request


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
        expected_column = get_entsoe_dataset_request(dataset).output_column
        if expected_column not in table.columns:
            raise ValueError(f"Expected column '{expected_column}' missing from {path}")
        tables.append(table)

    combined = reduce(
        lambda left, right: pd.merge(left, right, on="timestamp_utc", how="outer"),
        tables,
    )
    combined = combined.sort_values("timestamp_utc").reset_index(drop=True)
    combined.insert(
        1,
        "timestamp_local",
        combined["timestamp_utc"].dt.tz_convert(GERMANY_MARKET_TIMEZONE),
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
        hourly["timestamp_utc"].dt.tz_convert(GERMANY_MARKET_TIMEZONE),
    )
    return hourly


def write_combined_dataset(
    interim_folder: str | Path,
    processed_folder: str | Path,
    datasets: list[str],
    start_utc: pd.Timestamp | None = None,
    end_utc: pd.Timestamp | None = None,
    output_filename: str = "germany_model_dataset.csv",
) -> Path:
    """Combine parsed dataset CSVs and write one processed model dataset."""

    processed_folder = Path(processed_folder)
    processed_folder.mkdir(parents=True, exist_ok=True)
    combined = combine_interim_csvs(interim_folder, datasets)
    if start_utc is not None and end_utc is not None:
        combined = filter_by_utc_window(combined, start_utc, end_utc)
    combined = aggregate_to_hourly(combined)
    output_path = processed_folder / output_filename
    combined.to_csv(output_path, index=False)
    return output_path
