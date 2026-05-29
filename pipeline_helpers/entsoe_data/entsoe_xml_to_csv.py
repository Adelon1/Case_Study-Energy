"""Parse ENTSO-E XML responses into simple timestamp/value CSV tables."""

from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

from pipeline_helpers.entsoe_data.constants import get_entsoe_dataset_request

RESOLUTION_PATTERN = re.compile(r"^PT(?P<number>\d+)(?P<unit>M|H)$")


def xml_namespace(root: ET.Element) -> dict[str, str]:
    """Return the XML namespace mapping used by an ENTSO-E document."""

    if root.tag.startswith("{"):
        return {"ns": root.tag.split("}", 1)[0].strip("{")}
    return {"ns": ""}


def child_text(parent: ET.Element, tag: str, namespace: dict[str, str]) -> str | None:
    """Read direct child text while hiding ENTSO-E namespace syntax."""

    prefix = "ns:" if namespace.get("ns") else ""
    return parent.findtext(f"{prefix}{tag}", namespaces=namespace)


def parse_resolution_to_timedelta(resolution: str) -> timedelta:
    """Convert ENTSO-E ISO-like resolutions such as ``PT15M`` or ``PT60M``."""

    match = RESOLUTION_PATTERN.match(resolution)
    if not match:
        raise ValueError(f"Unsupported ENTSO-E resolution: {resolution}")

    amount = int(match.group("number"))
    unit = match.group("unit")
    if unit == "M":
        return timedelta(minutes=amount)
    if unit == "H":
        return timedelta(hours=amount)
    raise ValueError(f"Unsupported ENTSO-E resolution unit: {unit}")


def number_of_positions_in_period(
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    step: timedelta,
) -> int:
    """Return how many positions fit inside a Period time interval."""

    period_length_seconds = (period_end - period_start).total_seconds()
    step_seconds = step.total_seconds()
    if period_length_seconds % step_seconds != 0:
        raise ValueError(
            f"Period length {period_end - period_start} is not divisible by resolution {step}."
        )
    return int(period_length_seconds / step_seconds)


def time_series_matches_filters(
    time_series: ET.Element,
    required_tags: dict[str, str],
    namespace: dict[str, str],
) -> bool:
    """Check whether a TimeSeries contains required metadata tag values."""

    for tag, expected_value in required_tags.items():
        actual_value = child_text(time_series, tag, namespace)
        if actual_value != expected_value:
            return False
    return True


def read_period_points(
    period: ET.Element,
    value_tag: str,
    namespace: dict[str, str],
) -> list[tuple[int, float]]:
    """Read ``(position, value)`` pairs from one ENTSO-E Period."""

    prefix = "ns:" if namespace.get("ns") else ""
    points: list[tuple[int, float]] = []
    for point in period.findall(f"{prefix}Point", namespace):
        position_text = child_text(point, "position", namespace)
        value_text = child_text(point, value_tag, namespace)
        if position_text is None or value_text is None:
            continue
        points.append((int(position_text), float(value_text)))
    return sorted(points)


def expand_points_to_all_positions(
    points: list[tuple[int, float]],
    total_positions: int,
    curve_type: str | None,
) -> list[tuple[int, float]]:
    """Expand ENTSO-E Point entries to one value per time position.

    ``curveType=A03`` means the XML may contain variable-sized blocks: a value
    at position ``p`` is valid from ``p`` until the next explicitly listed
    position. This is common for solar night values and some prices. For other
    curve types, listed points are used directly.
    """

    if curve_type != "A03" or not points:
        return points

    expanded: list[tuple[int, float]] = []
    for index, (position, value) in enumerate(points):
        next_position = points[index + 1][0] if index + 1 < len(points) else total_positions + 1
        for expanded_position in range(position, next_position):
            if 1 <= expanded_position <= total_positions:
                expanded.append((expanded_position, value))
    return expanded


def parse_entsoe_xml_to_table(xml_path: str | Path, dataset: str) -> pd.DataFrame:
    """Parse one ENTSO-E XML file into ``timestamp_utc`` plus one value column.

    The dataset metadata in ``constants.py`` defines which value tag should be
    read and which TimeSeries blocks should be kept. For example, day-ahead
    prices keep only Sequence 1 and read ``price.amount``.
    """

    dataset_request = get_entsoe_dataset_request(dataset)
    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()
    namespace = xml_namespace(root)
    prefix = "ns:" if namespace.get("ns") else ""
    rows: list[dict[str, object]] = []

    for time_series in root.findall(f"{prefix}TimeSeries", namespace):
        if not time_series_matches_filters(
            time_series,
            dataset_request.required_time_series_tags,
            namespace,
        ):
            continue

        for period in time_series.findall(f"{prefix}Period", namespace):
            period_start_text = period.findtext(f"{prefix}timeInterval/{prefix}start", namespaces=namespace)
            period_end_text = period.findtext(f"{prefix}timeInterval/{prefix}end", namespaces=namespace)
            resolution_text = child_text(period, "resolution", namespace)
            curve_type = child_text(time_series, "curveType", namespace)
            if period_start_text is None or period_end_text is None or resolution_text is None:
                continue

            period_start = pd.Timestamp(period_start_text)
            period_end = pd.Timestamp(period_end_text)
            step = parse_resolution_to_timedelta(resolution_text)
            total_positions = number_of_positions_in_period(period_start, period_end, step)
            points = read_period_points(period, dataset_request.value_tag, namespace)
            expanded_points = expand_points_to_all_positions(points, total_positions, curve_type)

            for position, value in expanded_points:
                timestamp_utc = period_start + (position - 1) * step
                rows.append(
                    {
                        "timestamp_utc": timestamp_utc,
                        dataset_request.output_column: value,
                    }
                )

    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(columns=["timestamp_utc", dataset_request.output_column])

    table = table.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")
    return table.reset_index(drop=True)


def write_dataset_csv(xml_path: str | Path, dataset: str, output_path: str | Path) -> Path:
    """Parse one ENTSO-E XML file and write the standardized CSV output."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = parse_entsoe_xml_to_table(xml_path, dataset)
    table.to_csv(output_path, index=False)
    return output_path
