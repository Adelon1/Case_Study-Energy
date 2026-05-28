"""Convert user-friendly local delivery dates into ENTSO-E UTC windows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from pipeline_helpers.constants import ENTSOE_DATETIME_FORMAT, GERMANY_MARKET_TIMEZONE


@dataclass(frozen=True)
class LocalDateWindow:
    """One requested German local delivery window and its UTC equivalent."""

    start_local: pd.Timestamp
    end_local: pd.Timestamp
    start_utc: pd.Timestamp
    end_utc: pd.Timestamp
    entsoe_start: str
    entsoe_end: str


def parse_local_date_window(start: str, end: str) -> LocalDateWindow:
    """Parse ``DD-MM-YYYY`` start/end dates as German local midnights.

    The end date is exclusive. For example, ``01-05-2026`` to ``02-05-2026``
    means the German local delivery day 1 May 2026, from 00:00 to 24:00.
    """

    timezone = ZoneInfo(GERMANY_MARKET_TIMEZONE)
    try:
        start_date = datetime.strptime(start, "%d-%m-%Y")
        end_date = datetime.strptime(end, "%d-%m-%Y")
    except ValueError as exc:
        raise ValueError("Dates must use DD-MM-YYYY format, for example 01-05-2026.") from exc

    if end_date <= start_date:
        raise ValueError("End date must be after start date.")

    start_local = pd.Timestamp(start_date, tz=timezone)
    end_local = pd.Timestamp(end_date, tz=timezone)
    start_utc = start_local.tz_convert("UTC")
    end_utc = end_local.tz_convert("UTC")

    return LocalDateWindow(
        start_local=start_local,
        end_local=end_local,
        start_utc=start_utc,
        end_utc=end_utc,
        entsoe_start=start_utc.strftime(ENTSOE_DATETIME_FORMAT),
        entsoe_end=end_utc.strftime(ENTSOE_DATETIME_FORMAT),
    )
