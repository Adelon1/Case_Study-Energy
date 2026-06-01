"""Pipeline step: download XMLs, parse CSVs, and combine them into one dataset.

Example:
    .venv/bin/python pipeline_steps/01_build_dataset.py --interactive

    .venv/bin/python pipeline_steps/01_build_dataset.py --mode modelling
    --datasets day_ahead_prices load_forecast solar_forecast
    wind_onshore_forecast wind_offshore_forecast --start 01-01-2021
    --end 01-01-2026

Each run creates matching folders:

``data/01_raw/DataSet<i>/``
``data/02_interim/DataSet<i>/``
``data/03_processed/DataSet<i>/``
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

constants = importlib.import_module("pipeline_helpers.01_entsoe_data.00_constants")
combine_dataset_csvs = importlib.import_module("pipeline_helpers.01_entsoe_data.05_combine_dataset_csvs")
build_features = importlib.import_module("pipeline_helpers.01_entsoe_data.06_build_features")
date_windows = importlib.import_module("pipeline_helpers.01_entsoe_data.01_date_windows")
dataset_folders = importlib.import_module("pipeline_helpers.01_entsoe_data.02_dataset_folders")
entsoe_api = importlib.import_module("pipeline_helpers.01_entsoe_data.03_entsoe_api")
entsoe_xml_to_csv = importlib.import_module("pipeline_helpers.01_entsoe_data.04_entsoe_xml_to_csv")

ImputationReport = combine_dataset_csvs.ImputationReport
write_combined_dataset = combine_dataset_csvs.write_combined_dataset
write_feature_dataset = build_features.write_feature_dataset
parse_local_date_window = date_windows.parse_local_date_window
split_local_date_window_into_months = date_windows.split_local_date_window_into_months
create_folders_for_mode = dataset_folders.create_folders_for_mode
save_raw_xml_response = entsoe_api.save_raw_xml_response
send_entsoe_get_request = entsoe_api.send_entsoe_get_request
write_dataset_csv_from_xml_files = entsoe_xml_to_csv.write_dataset_csv_from_xml_files


def parse_command_line_arguments() -> argparse.Namespace:
    """Read requested datasets and time range from the command line."""

    parser = argparse.ArgumentParser(description="Build a joined ENTSO-E model dataset.")
    parser.add_argument("--interactive", action="store_true", help="Ask for settings interactively.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=sorted(constants.ENTSOE_DATASETS),
        help="Dataset names to download, parse, and combine.",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start delivery date in German local time, format DD-MM-YYYY.",
    )
    parser.add_argument(
        "--end",
        default=None,
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


def ask(prompt: str, default: str | None = None) -> str:
    """Ask for one interactive value."""

    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or (default or "")


def ask_choice(prompt: str, choices: list[str], default: str) -> str:
    """Ask until the response is one of the allowed choices."""

    choices_text = ", ".join(choices)
    while True:
        answer = ask(f"{prompt} ({choices_text})", default)
        if answer in choices:
            return answer
        print(f"Please choose one of: {choices_text}")


def ask_dataset_list(default: list[str]) -> list[str]:
    """Ask for a whitespace-separated dataset list and validate the names."""

    choices = sorted(constants.ENTSOE_DATASETS)
    default_text = " ".join(default)
    print("Available datasets:")
    for dataset in choices:
        print(f"  - {dataset}")

    while True:
        answer = ask("Datasets separated by spaces", default_text)
        datasets = answer.split()
        invalid = [dataset for dataset in datasets if dataset not in constants.ENTSOE_DATASETS]
        if datasets and not invalid:
            return datasets
        if invalid:
            print(f"Unknown dataset(s): {', '.join(invalid)}")
        else:
            print("Please enter at least one dataset.")


def apply_interactive_settings(args: argparse.Namespace) -> argparse.Namespace:
    """Fill dataset-building settings interactively."""

    default_datasets = args.datasets or [
        "day_ahead_prices",
        "load_forecast",
        "solar_forecast",
        "wind_onshore_forecast",
        "wind_offshore_forecast",
    ]
    args.datasets = ask_dataset_list(default_datasets)
    args.start = ask("Start delivery date DD-MM-YYYY", args.start or "01-01-2021")
    args.end = ask("Exclusive end delivery date DD-MM-YYYY", args.end or "01-01-2026")
    args.mode = ask_choice("Folder mode", ["test", "modelling"], args.mode)
    args.env = ask("Env file", args.env)
    return args


def validate_required_arguments(args: argparse.Namespace) -> None:
    """Make non-interactive runs fail with a clear message when inputs are missing."""

    missing = []
    if not args.datasets:
        missing.append("--datasets")
    if not args.start:
        missing.append("--start")
    if not args.end:
        missing.append("--end")
    if missing:
        raise ValueError(
            "Missing required argument(s): "
            f"{', '.join(missing)}. Use --interactive to answer prompts instead."
        )


def expected_hourly_rows(start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> int:
    """Return the expected number of hourly rows in a half-open UTC window."""

    return int((end_utc - start_utc) / pd.Timedelta(hours=1))


def count_obvious_outliers(table: pd.DataFrame) -> dict[str, int]:
    """Count simple sanity-check outliers by data column."""

    rules = {
        "day_ahead_price_eur_per_mwh": lambda series: (series < -500) | (series > 1000),
        "load_forecast_mw": lambda series: (series <= 0) | (series > 120000),
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
    """Infer parsed CSV frequency per dataset before hourly aggregation."""

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


def summarize_feature_columns(feature_table: pd.DataFrame) -> str:
    """Return a compact Markdown inventory of the generated feature table."""

    timestamp_columns = ["timestamp_utc", "timestamp_local"]
    target_columns = ["day_ahead_price_eur_per_mwh"]
    non_feature_columns = set(timestamp_columns + target_columns)
    candidate_feature_columns = [
        column for column in feature_table.columns if column not in non_feature_columns
    ]

    price_curve_groups: dict[str, list[int]] = {}
    for column in feature_table.columns:
        match = re.fullmatch(r"price_d(?P<days>\d+)_h(?P<hour>\d{2})", column)
        if not match:
            continue
        price_curve_groups.setdefault(match.group("days"), []).append(int(match.group("hour")))

    price_curve_rows = []
    for days, hours in sorted(price_curve_groups.items(), key=lambda item: int(item[0])):
        sorted_hours = sorted(hours)
        hour_text = (
            f"h{sorted_hours[0]:02d}...h{sorted_hours[-1]:02d}"
            if len(sorted_hours) > 1
            else f"h{sorted_hours[0]:02d}"
        )
        price_curve_rows.append([f"previous {days} day(s)", len(sorted_hours), hour_text])

    daily_summary_columns = sorted(
        column for column in feature_table.columns if re.fullmatch(r"price_d\d+_(min|max|mean)", column)
    )
    fundamental_columns = sorted(
        column
        for column in feature_table.columns
        if any(token in column for token in ["load", "solar", "wind", "renewable", "residual"])
    )
    calendar_columns = sorted(
        column
        for column in feature_table.columns
        if column.startswith("local_") or column == "is_holiday"
    )
    passthrough_source_columns = sorted(
        column
        for column in feature_table.columns
        if column in {request.output_column for request in constants.ENTSOE_DATASETS.values()}
    )

    rows = [
        ["Total feature table columns", len(feature_table.columns)],
        ["Timestamp columns", len([column for column in timestamp_columns if column in feature_table.columns])],
        ["Target columns", len([column for column in target_columns if column in feature_table.columns])],
        ["Candidate feature columns", len(candidate_feature_columns)],
        ["Fundamental/source/derived columns", len(fundamental_columns)],
        ["Calendar columns", len(calendar_columns)],
        ["Daily price curve columns", sum(len(hours) for hours in price_curve_groups.values())],
        ["Daily price summary columns", len(daily_summary_columns)],
    ]

    sections = [
        markdown_table(["Feature inventory item", "Count"], rows),
        "",
        "Feature groups:",
        "",
        f"- Passthrough ENTSO-E columns: `{passthrough_source_columns}`",
        f"- Fundamental derived columns: `{fundamental_columns}`",
        f"- Local calendar columns: `{calendar_columns}`",
        f"- Daily price summary columns: `{daily_summary_columns}`",
    ]

    if price_curve_rows:
        sections.extend(
            [
                "",
                "Daily price curve lag columns:",
                "",
                markdown_table(["Lag group", "Column count", "Hour columns"], price_curve_rows),
            ]
        )
    else:
        sections.append("- Daily price curve lag columns: none")

    return "\n".join(sections)


def write_qa_report(
    report_path: Path,
    dataset_name: str,
    datasets: list[str],
    combined_table: pd.DataFrame,
    feature_table: pd.DataFrame,
    date_window,
    interim_folder: Path,
    processed_csv_path: Path,
    feature_csv_path: Path,
    imputation_report: ImputationReport,
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
    imputation_rows = [
        [
            column,
            imputation_report.missing_before.get(column, 0),
            imputation_report.filled_from_previous_day.get(column, 0),
            imputation_report.missing_after_fill.get(column, 0),
        ]
        for column in imputation_report.columns
    ]
    imputation_table = (
        markdown_table(
            [
                "Column",
                "Missing before fill",
                "Filled from t-24h",
                "Missing after fill",
            ],
            imputation_rows,
        )
        if imputation_rows
        else "No forecast driver columns requiring imputation were included in this run."
    )

    report = f"""# Data Ingestion and QA Report

## Dataset Run

- Dataset folder: `{dataset_name}`
- Stage-3 CSV: `{processed_csv_path}`
- Feature CSV: `{feature_csv_path}`
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

## Leakage-Safe Imputation

- Applied after hourly assembly and before feature generation.
- Columns considered: `{imputation_report.columns}`
- Fill rule: a missing value at time `t` is filled from the same column at `t - 24h`.
- The pipeline does not use future values, interpolation across future points, or backfill.
- Rows dropped after the fill because forecast driver values were still missing: `{imputation_report.dropped_rows_after_fill}`

{imputation_table}

## Obvious Outlier Checks

Outlier checks are QA flags only; the pipeline does not remove values automatically.

Rules:

- `day_ahead_price_eur_per_mwh`: below -500 or above 1000
- `load_forecast_mw`: <= 0 or above 120000
- `solar_forecast_mw`, `wind_onshore_forecast_mw`: below 0 or above 90000
- `wind_offshore_forecast_mw`: below 0 or above 30000

{markdown_table(["Column", "Obvious outlier count"], outlier_rows)}

## Feature Table Inventory

{summarize_feature_columns(feature_table)}

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
    """Run the full XML -> stage-2 CSV -> stage-3 CSV workflow."""

    started_at = time.perf_counter()
    args = parse_command_line_arguments()
    if args.interactive:
        args = apply_interactive_settings(args)
    validate_required_arguments(args)
    date_window = parse_local_date_window(args.start, args.end)
    date_chunks = split_local_date_window_into_months(args.start, args.end)
    folders = create_folders_for_mode(args.mode, args.start, args.end)

    print(f"Run folder: {folders.name}  ({args.mode} mode)")
    print(f"  local window : {date_window.start_local} -> {date_window.end_local}")
    print(f"  UTC window   : {date_window.entsoe_start} -> {date_window.entsoe_end}")
    print(f"  datasets     : {len(args.datasets)} ({', '.join(args.datasets)})")
    print(f"  API chunks   : {len(date_chunks)} monthly")

    print("\nDownload and parse")
    for dataset in args.datasets:
        csv_path = folders.interim / f"{dataset}.csv"
        xml_paths: list[Path] = []
        for chunk in date_chunks:
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
        write_dataset_csv_from_xml_files(xml_paths, dataset, csv_path)
        print(f"  {dataset:<24} {len(date_chunks):>3} chunks -> {csv_path.name}")

    combined = write_combined_dataset(
        folders.interim,
        folders.processed,
        args.datasets,
        start_utc=date_window.start_utc,
        end_utc=date_window.end_utc,
    )
    feature_dataset = write_feature_dataset(
        combined.path,
        folders.processed / "germany_model_features.csv",
    )
    report_path = write_qa_report(
        folders.processed / "data_qa_report.md",
        folders.name,
        args.datasets,
        combined.table,
        feature_dataset.table,
        date_window,
        folders.interim,
        combined.path,
        feature_dataset.path,
        combined.imputation_report,
    )
    expected_rows = expected_hourly_rows(date_window.start_utc, date_window.end_utc)
    imputation = combined.imputation_report
    print("\nAssemble")
    print(f"  combined rows : {len(combined.table):,} / {expected_rows:,} expected")
    print(
        f"  imputed t-24h : {sum(imputation.filled_from_previous_day.values())}"
        f"   dropped: {imputation.dropped_rows_after_fill}"
    )
    print(f"  feature rows  : {len(feature_dataset.table):,}")

    print("\nOutputs")
    print(f"  dataset  : {combined.path}")
    print(f"  features : {feature_dataset.path}")
    print(f"  QA report: {report_path}")
    print(f"\nDone in {format_elapsed_time(time.perf_counter() - started_at)}")


if __name__ == "__main__":
    main()
