"""Small helpers for model artifact paths and JSON IO.

Public entry points:
    ``dataset_name_from_feature_path(...)``
    ``default_models_base_folder(...)``
    ``write_json(...)``
    ``read_json(...)``

``clean_json_value`` is an internal helper that prevents invalid JSON values
such as ``NaN`` from leaking into metadata artifacts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def clean_json_value(value):
    """Convert Python/pandas non-finite values into valid JSON values."""

    if isinstance(value, dict):
        return {key: clean_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [clean_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dataset_name_from_feature_path(feature_path: str | Path) -> str:
    """Use the stage-3 dataset folder name as the model namespace."""

    return Path(feature_path).resolve().parent.name


def default_models_base_folder(feature_path: str | Path) -> Path:
    """Return the default external model artifact folder."""

    return Path("models") / dataset_name_from_feature_path(feature_path)


def write_json(data: dict[str, object], path: str | Path) -> None:
    """Write a JSON artifact with stable formatting."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_data = clean_json_value(data)
    path.write_text(
        json.dumps(clean_data, indent=2, sort_keys=True, default=str, allow_nan=False),
        encoding="utf-8",
    )


def read_json(path: str | Path) -> dict[str, object]:
    """Read a JSON object from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))
