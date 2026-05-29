"""Build ENTSO-E GET requests, send them, and save raw XML responses.

This module handles only communication with ENTSO-E:

1. load the API token from ``.env`` or the shell environment,
2. build query parameters for a named dataset and date range,
3. send the request with ``requests.get`` and the token in a header,
4. return or save the raw XML response.

Parsing XML into CSV is intentionally separate from this file.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

from pipeline_helpers.entsoe_data import constants


class EntsoeApiConfigurationError(RuntimeError):
    """Raised when the ENTSO-E API token or local API setup is missing."""


def load_entsoe_api_key(env_path: str | Path = ".env") -> str:
    """Load ``ENTSOE_API_KEY`` from a local ``.env`` file or environment."""

    load_dotenv(env_path)
    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key or api_key == "replace_with_your_entsoe_security_token":
        raise EntsoeApiConfigurationError(
            "ENTSOE_API_KEY is missing. Add your token to .env after ENTSO-E grants API access."
        )
    return api_key


def format_entsoe_datetime(value: datetime | str) -> str:
    """Return an ENTSO-E timestamp in ``yyyyMMddHHmm`` format."""

    if isinstance(value, datetime):
        return value.strftime(constants.ENTSOE_DATETIME_FORMAT)

    try:
        datetime.strptime(value, constants.ENTSOE_DATETIME_FORMAT)
    except ValueError as exc:
        raise ValueError(
            f"Invalid ENTSO-E timestamp '{value}'. Expected yyyyMMddHHmm, e.g. 202501010000."
        ) from exc
    return value


def build_get_query_parameters(
    dataset: str,
    start: datetime | str,
    end: datetime | str,
) -> dict[str, str]:
    """Build ENTSO-E GET query parameters without secrets."""

    dataset_request = constants.get_entsoe_dataset_request(dataset)
    return {
        **dataset_request.params,
        "periodStart": format_entsoe_datetime(start),
        "periodEnd": format_entsoe_datetime(end),
    }


def send_entsoe_get_request(
    dataset: str,
    start: datetime | str,
    end: datetime | str,
    env_path: str | Path = ".env",
    timeout_seconds: int = 60,
) -> str:
    """Send one ENTSO-E GET request and return the raw XML body."""

    api_key = load_entsoe_api_key(env_path)
    params = build_get_query_parameters(dataset, start, end)
    headers = {"SECURITY_TOKEN": api_key}

    safe_request = requests.Request("GET", constants.ENTSOE_BASE_URL, params=params).prepare()
    try:
        response = requests.get(
            constants.ENTSOE_BASE_URL,
            params=params,
            headers=headers,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"ENTSO-E GET request failed for dataset '{dataset}'. "
            f"Safe request URL: {safe_request.url}"
        ) from exc

    return response.text


def save_raw_xml_response(xml_text: str, output_path: str | Path) -> Path:
    """Write raw ENTSO-E XML text to disk and return the saved path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml_text, encoding="utf-8")
    return path
