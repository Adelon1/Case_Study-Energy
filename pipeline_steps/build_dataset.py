"""Pipeline step: download XMLs, parse CSVs, and combine them into one dataset.

Example:
    .venv/bin/python pipeline_steps/build_dataset.py \
      --datasets day_ahead_prices load_forecast solar_forecast wind_onshore_forecast wind_offshore_forecast \
      --start 01-05-2026 \
      --end 02-05-2026

Each run creates matching folders:

``data/raw/DataSet<i>/``
``data/interim/DataSet<i>/``
``data/processed/DataSet<i>/``
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_helpers.entsoe_data import constants  # noqa: E402
from pipeline_helpers.entsoe_data.combine_dataset_csvs import write_combined_dataset  # noqa: E402
from pipeline_helpers.entsoe_data.build_features import write_feature_dataset  # noqa: E402
from pipeline_helpers.entsoe_data.date_windows import (  # noqa: E402
    parse_local_date_window,
    split_local_date_window_into_months,
)
from pipeline_helpers.entsoe_data.dataset_folders import create_folders_for_mode  # noqa: E402
from pipeline_helpers.entsoe_data.entsoe_api import save_raw_xml_response, send_entsoe_get_request  # noqa: E402
from pipeline_helpers.entsoe_data.entsoe_xml_to_csv import write_dataset_csv_from_xml_files  # noqa: E402


def parse_command_line_arguments() -> argparse.Namespace:
    """Read requested datasets and time range from the command line."""

    parser = argparse.ArgumentParser(description="Build a joined ENTSO-E model dataset.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        choices=sorted(constants.ENTSOE_DATASETS),
        help="Dataset names to download, parse, and combine.",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start delivery date in German local time, format DD-MM-YYYY.",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="Exclusive end delivery date in German local time, format DD-MM-YYYY.",
    )
    parser.add_argument(
        "--mode",
        choices=["test", "modelling"],
        default="test",
        help="Folder naming mode. 'test' uses DataSet<i>; 'modelling' uses a standard modelling name.",
    )
    parser.add_argument("--env", default=".env", help="Path to the local .env file.")
    return parser.parse_args()


def expected_hourly_rows(start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> int:
    """Return the expected number of hourly rows in a half-open UTC window."""

    return int((end_utc - start_utc) / pd.Timedelta(hours=1))


def count_obvious_outliers(table: pd.DataFrame) -> dict[str, int]:
    """Count simple sanity-check outliers by data column."""

    rules = {
        "day_ahead_price_eur_per_mwh": lambda series: (series < -500) | (series > 1000),
        "load_forecast_mw": lambda series: (series <= 0) | (series > 120000),
        "load_actual_mw": lambda series: (series <= 0) | (series > 120000),
        "solar_forecast_mw": lambda series: (series < 0) | (series > 90000),
        "wind_onshore_forecast_mw": lambda series: (series < 0) | (series > 90000),
        "wind_offshore_forecast_mw": lambda series: (series < 0) | (series > 30000),
    }
    counts: dict[str, int] = {}
    for column, rule in rules.items():
        if column not in table.columns:
            continue
        counts[column] = int(rule(table[column]).fillna(False).sum())
    return counts


def infer_input_frequencies(interim_folder: Path, datasets: list[str]) -> dict[str, str]:
    """Infer raw parsed CSV frequency per dataset before hourly aggregation."""

    frequencies: dict[str, str] = {}
    for dataset in datasets:
        path = interim_folder / f"{dataset}.csv"
        table = pd.read_csv(path)
        timestamps = pd.to_datetime(table["timestamp_utc"], utc=True).sort_values()
        if len(timestamps) < 2:
            frequencies[dataset] = "not enough rows to infer"
            continue
        most_common_step = timestamps.diff().dropna().mode()
        frequencies[dataset] = str(most_common_step.iloc[0]) if not most_common_step.empty else "unknown"
    return frequencies


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    """Create a compact Markdown table."""

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def format_elapsed_time(seconds: float) -> str:
    """Format a runtime duration for command-line output."""

    minutes, remaining_seconds = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)

    if hours:
        return f"{hours} h {minutes} min {remaining_seconds:.1f} sec"
    if minutes:
        return f"{minutes} min {remaining_seconds:.1f} sec"
    return f"{remaining_seconds:.1f} sec"


def write_qa_report(
    report_path: Path,
    dataset_name: str,
    datasets: list[str],
    combined_table: pd.DataFrame,
    date_window,
    interim_folder: Path,
    processed_csv_path: Path,
) -> Path:
    """Write the required public data ingestion and QA report."""

    data_columns = [column for column in combined_table.columns if column not in {"timestamp_utc", "timestamp_local"}]
    missing_counts = combined_table[data_columns].isna().sum().astype(int).to_dict()
    outlier_counts = count_obvious_outliers(combined_table)
    duplicate_timestamps = int(combined_table["timestamp_utc"].duplicated().sum())
    expected_rows = expected_hourly_rows(date_window.start_utc, date_window.end_utc)
    actual_rows = len(combined_table)
    coverage_pct = round(actual_rows / expected_rows * 100, 2) if expected_rows else 0
    input_frequencies = infer_input_frequencies(interim_folder, datasets)
    monotonic_utc = bool(pd.to_datetime(combined_table["timestamp_utc"], utc=True).is_monotonic_increasing)
    timestamp_range = (
        f"{combined_table['timestamp_utc'].iloc[0]} to {combined_table['timestamp_utc'].iloc[-1]}"
        if not combined_table.empty
        else "empty"
    )
    local_range = (
        f"{combined_table['timestamp_local'].iloc[0]} to {combined_table['timestamp_local'].iloc[-1]}"
        if not combined_table.empty
        else "empty"
    )

    source_rows = []
    for dataset in datasets:
        request = constants.ENTSOE_DATASETS[dataset]
        source_rows.append(
            [
                dataset,
                request.description,
                request.output_column,
                request.params,
            ]
        )

    missing_rows = [[column, missing_counts.get(column, 0)] for column in data_columns]
    outlier_rows = [[column, outlier_counts.get(column, 0)] for column in data_columns]
    frequency_rows = [[dataset, input_frequencies[dataset]] for dataset in datasets]

    report = f"""# Data Ingestion and QA Report

## Dataset Run

- Dataset folder: `{dataset_name}`
- Processed CSV: `{processed_csv_path}`
- Data source: ENTSO-E Transparency Platform REST API
- API endpoint: `{constants.ENTSOE_BASE_URL}`
- Market: Germany/Luxembourg bidding zone (`DE-LU`)
- Timezone convention: ENTSO-E XML timestamps are parsed as UTC. `timestamp_utc` is the canonical join key. `timestamp_local` is derived from UTC using `Europe/Berlin` for reporting and delivery-period interpretation.
- Requested local delivery window: `{date_window.start_local}` to `{date_window.end_local}`; end is exclusive.
- API UTC window: `{date_window.entsoe_start}` to `{date_window.entsoe_end}`
- Final timestamp UTC range: `{timestamp_range}`
- Final timestamp local range: `{local_range}`

## Included Data

{markdown_table(["Dataset name", "Dataset description", "Final column", "ENTSO-E request params"], source_rows)}

## Frequency and Coverage

{markdown_table(["Dataset", "Parsed input frequency before assembly"], frequency_rows)}

- Final assembled frequency: hourly mean.
- Expected hourly rows: `{expected_rows}`
- Actual hourly rows: `{actual_rows}`
- Coverage: `{coverage_pct}%`
- Duplicate `timestamp_utc` rows: `{duplicate_timestamps}`
- UTC timestamps monotonic increasing: `{monotonic_utc}`

## Missing Data

{markdown_table(["Column", "Missing values"], missing_rows)}

## Obvious Outlier Checks

Outlier checks are QA flags only; the pipeline does not remove values automatically.

Rules:

- `day_ahead_price_eur_per_mwh`: below -500 or above 1000
- `load_forecast_mw`, `load_actual_mw`: <= 0 or above 120000
- `solar_forecast_mw`, `wind_onshore_forecast_mw`: below 0 or above 90000
- `wind_offshore_forecast_mw`: below 0 or above 30000

{markdown_table(["Column", "Obvious outlier count"], outlier_rows)}

## Timestamp Alignment

- Datasets are joined by `timestamp_utc`.
- Any timestamp alignment problem appears as missing values after the join.
- Local timestamps are derived after joining and hourly aggregation, not used as the primary key.

## DST Handling

The pipeline does not join on local clock time. ENTSO-E XML timestamps are UTC, which is unique across daylight-saving-time transitions. The requested delivery dates are provided as German local dates (`DD-MM-YYYY`) and converted once to UTC using pandas/zoneinfo timezone rules. This means normal days, 23-hour spring DST days, and 25-hour autumn DST days produce the correct UTC window length.

## Known Limitations

- The report uses simple rule-based outlier checks; it does not classify market-valid scarcity or negative-price events as errors unless they exceed the stated thresholds.
- ENTSO-E may return full market documents that overlap the request window. The final combined dataset is filtered to the requested local delivery window after parsing.
- Final hourly values are arithmetic means of the parsed ENTSO-E resolution. This is appropriate for the current price and MW forecast series, but should be reviewed if future datasets represent totals rather than average levels.
- API availability, revisions, and publication timing are controlled by ENTSO-E and TSOs.
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    """Run the full XML -> interim CSV -> combined CSV workflow."""

    started_at = time.perf_counter()
    args = parse_command_line_arguments()
    date_window = parse_local_date_window(args.start, args.end)
    date_chunks = split_local_date_window_into_months(args.start, args.end)
    folders = create_folders_for_mode(args.mode, args.start, args.end)

    print(f"Created run folders: {folders.name}")
    print(f"  raw: {folders.raw}")
    print(f"  interim: {folders.interim}")
    print(f"  processed: {folders.processed}")
    print("Requested German local delivery window:")
    print(f"  local: {date_window.start_local} to {date_window.end_local}")
    print(f"  UTC/API: {date_window.entsoe_start} to {date_window.entsoe_end}")
    print(f"  API chunks: {len(date_chunks)} monthly chunk(s)")

    for dataset in args.datasets:
        csv_path = folders.interim / f"{dataset}.csv"
        xml_paths: list[Path] = []

        print(f"\nDownloading {dataset}...")
        for index, chunk in enumerate(date_chunks, start=1):
            xml_path = (
                folders.raw
                / f"{dataset}_{chunk.entsoe_start}_{chunk.entsoe_end}.xml"
            )
            xml_text = send_entsoe_get_request(
                dataset,
                chunk.entsoe_start,
                chunk.entsoe_end,
                env_path=args.env,
            )
            save_raw_xml_response(xml_text, xml_path)
            xml_paths.append(xml_path)
            print(f"  chunk {index}/{len(date_chunks)} saved XML: {xml_path}")

        print(f"Parsing {dataset}...")
        write_dataset_csv_from_xml_files(xml_paths, dataset, csv_path)
        print(f"  saved CSV: {csv_path}")

    combined = write_combined_dataset(
        folders.interim,
        folders.processed,
        args.datasets,
        start_utc=date_window.start_utc,
        end_utc=date_window.end_utc,
    )
    report_path = write_qa_report(
        folders.processed / "data_qa_report.md",
        folders.name,
        args.datasets,
        combined.table,
        date_window,
        folders.interim,
        combined.path,
    )
    feature_dataset = write_feature_dataset(
        combined.path,
        folders.processed / "germany_model_features.csv",
    )
    print(f"\nCombined dataset saved: {combined.path}")
    print(f"Feature dataset saved: {feature_dataset.path}")
    print(f"QA report saved: {report_path}")
    print(f"Total runtime: {format_elapsed_time(time.perf_counter() - started_at)}")


if __name__ == "__main__":
    main()
