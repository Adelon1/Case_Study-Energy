"""Noninteractive reproducibility runner for the case study.

The numbered scripts in ``pipeline_steps/`` stay interactive and pleasant for
manual work. This module is the reviewer path: it reads one YAML config and
calls the same step functions with explicit settings.

Example:
    .venv/bin/python -m pipeline --config configs/case_study.yaml --noninteractive
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from types import SimpleNamespace

import yaml


build_step = importlib.import_module("pipeline_steps.01_build_dataset")
validation_step = importlib.import_module("pipeline_steps.02_validate_model")
forecast_step = importlib.import_module("pipeline_steps.03_run_forecast_view")
entsoe_constants = importlib.import_module("pipeline_helpers.01_entsoe_data.00_constants")


def parse_args() -> argparse.Namespace:
    """Parse the tiny wrapper CLI."""

    parser = argparse.ArgumentParser(description="Reproduce the case-study pipeline from YAML.")
    parser.add_argument(
        "--config",
        default="configs/case_study.yaml",
        help="YAML config describing build, validation, and forecast-view runs.",
    )
    parser.add_argument(
        "--noninteractive",
        action="store_true",
        help="Required as an explicit acknowledgement that no prompts will be shown.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, object]:
    """Read the reproducibility YAML file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Pipeline config must be a YAML mapping.")
    return config


def namespace_from_defaults(defaults: SimpleNamespace, overrides: dict[str, object]) -> SimpleNamespace:
    """Merge YAML values into a step's existing default namespace."""

    values = vars(defaults).copy()
    values.update(overrides)
    return SimpleNamespace(**values)


def expand_datasets(value: object) -> list[str]:
    """Expand the same ``all`` shorthand used by the interactive build step."""

    if value in (None, "all"):
        return sorted(entsoe_constants.ENTSOE_DATASETS)
    if isinstance(value, str):
        return value.split()
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        if value == ["all"]:
            return sorted(entsoe_constants.ENTSOE_DATASETS)
        return value
    raise ValueError("build.datasets must be 'all', a whitespace string, or a list of names.")


def run_build(config: dict[str, object]) -> None:
    """Run the configured dataset build when enabled."""

    build_config = config.get("build", {})
    if not isinstance(build_config, dict) or not build_config.get("enabled", True):
        print("Skipping dataset build.")
        return

    args = namespace_from_defaults(
        build_step.parse_command_line_arguments(),
        {
            **build_config,
            "datasets": expand_datasets(build_config.get("datasets", "all")),
        },
    )
    print("\n=== 1/3 Build dataset ===")
    build_step.run_dataset_build(args)


def run_validations(config: dict[str, object]) -> None:
    """Run every configured validation job."""

    feature_csv = str(config.get("feature_csv", validation_step.parse_command_line_arguments().features))
    validations = config.get("validations", [])
    if not isinstance(validations, list):
        raise ValueError("validations must be a list.")

    print(f"\n=== 2/3 Validate models ({len(validations)} run(s)) ===")
    for index, validation_config in enumerate(validations, start=1):
        if not isinstance(validation_config, dict):
            raise ValueError("Each validation entry must be a mapping.")
        args = namespace_from_defaults(
            validation_step.parse_command_line_arguments(),
            {
                "features": feature_csv,
                "target_transform": "raw",
                **validation_config,
            },
        )
        print(f"\n--- Validation {index}/{len(validations)}: {args.model} / {args.forecast_setup} ---")
        validation_step.run_validation_step(args)


def run_forecast_views(config: dict[str, object]) -> None:
    """Run every configured forecast-to-curve view."""

    feature_csv = str(config.get("feature_csv", forecast_step.DEFAULT_FEATURES))
    forecast_views = config.get("forecast_views", [])
    if not isinstance(forecast_views, list):
        raise ValueError("forecast_views must be a list.")

    print(f"\n=== 3/3 Forecast views ({len(forecast_views)} run(s)) ===")
    for index, view_config in enumerate(forecast_views, start=1):
        if not isinstance(view_config, dict):
            raise ValueError("Each forecast_views entry must be a mapping.")
        view_values = {
            "features": feature_csv,
            "target_transform": "raw",
            "benchmark": "trailing_average",
            "ai_commentary": False,
            **view_config,
        }
        if (
            view_values.get("forecast_setup") == "hourly_day_ahead"
            and view_values.get("delivery_date")
            and not view_values.get("test_begin")
        ):
            view_values["test_begin"] = view_values["delivery_date"]
            view_values["test_end"] = forecast_step.next_delivery_date(str(view_values["delivery_date"]))
        args = namespace_from_defaults(
            forecast_step.parse_command_line_arguments(),
            view_values,
        )
        print(f"\n--- Forecast view {index}/{len(forecast_views)}: {args.model} / {args.forecast_setup} ---")
        forecast_step.run_forecast_view(args)


def main() -> None:
    """Run the configured full pipeline."""

    args = parse_args()
    if not args.noninteractive:
        raise SystemExit("Pass --noninteractive to confirm this long-running run should not prompt.")

    config = load_config(args.config)
    run_build(config)
    run_validations(config)
    run_forecast_views(config)
    print("\nReproduction run complete.")


if __name__ == "__main__":
    main()
