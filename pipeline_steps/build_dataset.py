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
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_helpers.combine_dataset_csvs import write_combined_dataset  # noqa: E402
from pipeline_helpers.constants import ENTSOE_DATASETS  # noqa: E402
from pipeline_helpers.date_windows import parse_local_date_window  # noqa: E402
from pipeline_helpers.dataset_folders import create_dataset_folders  # noqa: E402
from pipeline_helpers.entsoe_api import save_raw_xml_response, send_entsoe_get_request  # noqa: E402
from pipeline_helpers.entsoe_xml_to_csv import write_dataset_csv  # noqa: E402


def parse_command_line_arguments() -> argparse.Namespace:
    """Read requested datasets and time range from the command line."""

    parser = argparse.ArgumentParser(description="Build a joined ENTSO-E model dataset.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        choices=sorted(ENTSOE_DATASETS),
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
    parser.add_argument("--env", default=".env", help="Path to the local .env file.")
    return parser.parse_args()


def main() -> None:
    """Run the full XML -> interim CSV -> combined CSV workflow."""

    args = parse_command_line_arguments()
    date_window = parse_local_date_window(args.start, args.end)
    folders = create_dataset_folders()

    print(f"Created run folders: {folders.name}")
    print(f"  raw: {folders.raw}")
    print(f"  interim: {folders.interim}")
    print(f"  processed: {folders.processed}")
    print("Requested German local delivery window:")
    print(f"  local: {date_window.start_local} to {date_window.end_local}")
    print(f"  UTC/API: {date_window.entsoe_start} to {date_window.entsoe_end}")

    for dataset in args.datasets:
        xml_path = folders.raw / f"{dataset}.xml"
        csv_path = folders.interim / f"{dataset}.csv"

        print(f"\nDownloading {dataset}...")
        response = send_entsoe_get_request(
            dataset,
            date_window.entsoe_start,
            date_window.entsoe_end,
            env_path=args.env,
        )
        save_raw_xml_response(response, xml_path)
        print(f"  saved XML: {xml_path}")

        print(f"Parsing {dataset}...")
        write_dataset_csv(xml_path, dataset, csv_path)
        print(f"  saved CSV: {csv_path}")

    combined_path = write_combined_dataset(
        folders.interim,
        folders.processed,
        args.datasets,
        start_utc=date_window.start_utc,
        end_utc=date_window.end_utc,
    )
    print(f"\nCombined dataset saved: {combined_path}")


if __name__ == "__main__":
    main()
