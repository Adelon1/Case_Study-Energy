"""Small helpers for model artifact paths and JSON IO."""

from __future__ import annotations

import json
from pathlib import Path


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
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, object]:
    """Read a JSON object from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))
