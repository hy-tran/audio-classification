#!/usr/bin/env python3
"""Utility for downloading and extracting the Audio Scene Classification dataset."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen


DATA_URL = (
    "https://tcddeeplearning.blob.core.windows.net/"
    "deeplearning202324/audiosceneclassification2025.zip"
)
ARCHIVE_NAME = "audiosceneclassification2025.zip"
DATASET_DIR_NAME = "audiosceneclassification2025"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ARCHIVE_PATH = RAW_DIR / ARCHIVE_NAME
EXTRACT_DIR = RAW_DIR / DATASET_DIR_NAME


def download_archive(url: str, destination: Path, force: bool = False) -> None:
    """Download the dataset archive if it is not already present."""
    if destination.exists() and not force:
        print(f"Archive already present at {destination}")
        return

    print(f"Downloading dataset from {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        with urlopen(url) as response, open(destination, "wb") as target_file:
            shutil.copyfileobj(response, target_file)
    except Exception as exc:  # noqa: BLE001 - want to surface download errors
        if destination.exists():
            destination.unlink()
        raise RuntimeError(f"Failed to download archive: {exc}") from exc

    print(f"Saved archive to {destination}")


def _select_source_root(temp_dir: Path) -> Path:
    entries = [
        entry for entry in temp_dir.iterdir() if entry.name and entry.name != "__MACOSX"
    ]

    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]

    return temp_dir


def extract_archive(archive_path: Path, destination: Path, force: bool = False) -> None:
    """Extract the dataset archive into the destination directory."""
    if destination.exists() and not force:
        print(f"Dataset already extracted at {destination}")
        return

    print(f"Extracting archive to {destination}")

    if destination.exists():
        shutil.rmtree(destination)

    with zipfile.ZipFile(archive_path, "r") as archive_file:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            archive_file.extractall(tmp_dir)
            source_root = _select_source_root(tmp_dir)

            destination.mkdir(parents=True, exist_ok=True)
            for item in source_root.iterdir():
                target_path = destination / item.name
                shutil.move(str(item), target_path)

    print("Extraction complete")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract the Audio Scene Classification dataset."
    )
    parser.add_argument(
        "--url",
        default=DATA_URL,
        help="Source URL for the dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the archive even if it already exists.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract the dataset even if the folder already exists.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    download_archive(args.url, ARCHIVE_PATH, force=args.force_download)
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(
            f"Expected archive at {ARCHIVE_PATH}, but it does not exist."
        )

    extract_archive(ARCHIVE_PATH, EXTRACT_DIR, force=args.force_extract)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(1)

