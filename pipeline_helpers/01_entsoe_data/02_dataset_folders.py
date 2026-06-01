"""Create matching numbered data-stage folders for one pipeline run."""

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


def next_dataset_name(base_dir: str | Path = "data/01_raw") -> str:
    """Return the next available ``DataSet<i>`` name based on stage-1 folders."""

    base_path = Path(base_dir)
    index = 0
    while (base_path / f"DataSet{index}").exists():
        index += 1
    return f"DataSet{index}"


def available_named_folder(preferred_name: str, base_dir: str | Path = "data/01_raw") -> str:
    """Return ``preferred_name`` or add a numeric suffix if it already exists."""

    base_path = Path(base_dir)
    if not (base_path / preferred_name).exists():
        return preferred_name

    index = 1
    while (base_path / f"{preferred_name}_{index}").exists():
        index += 1
    return f"{preferred_name}_{index}"


def create_dataset_folders(dataset_name: str | None = None) -> DatasetFolders:
    """Create matching folders under ``data/01_raw``, ``data/02_interim``, and ``data/03_processed``."""

    name = available_named_folder(dataset_name) if dataset_name else next_dataset_name()
    folders = DatasetFolders(
        name=name,
        raw=Path("data/01_raw") / name,
        interim=Path("data/02_interim") / name,
        processed=Path("data/03_processed") / name,
    )
    folders.raw.mkdir(parents=True, exist_ok=False)
    folders.interim.mkdir(parents=True, exist_ok=False)
    folders.processed.mkdir(parents=True, exist_ok=False)
    return folders


def create_folders_for_mode(mode: str, start: str, end: str) -> DatasetFolders:
    """Create dataset folders using a standard name for modelling runs."""

    if mode == "test":
        return create_dataset_folders()
    if mode == "modelling":
        start_year = start[-4:]
        end_year = end[-4:]
        return create_dataset_folders(f"germany_modelling_{start_year}_{end_year}")
    raise ValueError(f"Unsupported mode: {mode}")
