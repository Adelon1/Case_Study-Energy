"""Create matching raw, interim, and processed folders for one pipeline run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetFolders:
    """Folder paths allocated to one data-building run."""

    name: str
    raw: Path
    interim: Path
    processed: Path


def next_dataset_name(base_dir: str | Path = "data/raw") -> str:
    """Return the next available ``DataSet<i>`` name based on raw folders."""

    base_path = Path(base_dir)
    index = 0
    while (base_path / f"DataSet{index}").exists():
        index += 1
    return f"DataSet{index}"


def create_dataset_folders(dataset_name: str | None = None) -> DatasetFolders:
    """Create matching folders under ``data/raw``, ``data/interim``, and ``data/processed``."""

    name = dataset_name or next_dataset_name()
    folders = DatasetFolders(
        name=name,
        raw=Path("data/raw") / name,
        interim=Path("data/interim") / name,
        processed=Path("data/processed") / name,
    )
    folders.raw.mkdir(parents=True, exist_ok=False)
    folders.interim.mkdir(parents=True, exist_ok=False)
    folders.processed.mkdir(parents=True, exist_ok=False)
    return folders
