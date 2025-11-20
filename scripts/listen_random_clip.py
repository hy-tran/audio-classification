#!/usr/bin/env python3
"""Play a random clip from the dataset and display its label."""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
import os
import subprocess
import platform

try:
    from IPython.display import Audio, display
except ImportError:  # pragma: no cover - runtime dependency
    Audio = None
    display = None

DEFAULT_DATA_ROOT = "data/raw/audiosceneclassification2025"
META_FILENAME = "meta.txt"
AUDIO_SUBDIR = "audio"


@dataclass
class ClipInfo:
    path: Path
    label: str | None = None
    clip_id: str | None = None


def read_metadata(meta_path: Path) -> dict[str, dict[str, str]]:
    """Load metadata TSV into a dictionary keyed by relative path."""
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    meta: dict[str, dict[str, str]] = {}
    with meta_path.open(encoding="utf-8") as fh:
        for line_num, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rel_path, label, clip_id = line.split("\t")
            except ValueError:
                print(
                    f"Skipping malformed metadata line {line_num}: {raw_line!r}",
                    file=sys.stderr,
                )
                continue
            meta[rel_path] = {"label": label, "clip_id": clip_id}
    return meta


def pick_clip(
    audio_dir: Path, meta: dict[str, dict[str, str]], rng: random.Random, rel_path: str | None
) -> ClipInfo:
    """Pick either a requested clip or random clip and attach metadata."""
    if rel_path:
        candidate = audio_dir / rel_path
        if not candidate.is_file():
            raise FileNotFoundError(f"Specified clip does not exist: {candidate}")
        chosen = candidate
    else:
        choices = sorted(audio_dir.glob("*.wav"))
        if not choices:
            raise FileNotFoundError(f"No .wav files found under {audio_dir}")
        chosen = rng.choice(choices)

    rel_from_audio_dir = chosen.relative_to(audio_dir)
    key_primary = (Path(AUDIO_SUBDIR) / rel_from_audio_dir).as_posix()
    key_fallback = chosen.relative_to(audio_dir.parents[0]).as_posix()
    info = meta.get(key_primary) or meta.get(key_fallback)

    return ClipInfo(
        path=chosen,
        label=info["label"] if info else None,
        clip_id=info["clip_id"] if info else None,
    )


def play_clip(clip: ClipInfo) -> None:
    """Play clip via IPython display if available, else print instructions."""
    print(f"Playing: {clip.path}")
    if clip.label or clip.clip_id:
        print(f"Label: {clip.label or 'unknown'} | Clip: {clip.clip_id or 'unknown'}")
    else:
        print("No metadata found for this file.")

    # Prefer platform-native playback on Windows (no extra deps).
    try:
        if platform.system().lower().startswith("win"):
            # On Windows prefer launching the default application via cmd start
            # so playback continues independently and the script exits.
            try:
                subprocess.Popen(["cmd", "/c", "start", "", str(clip.path)])
                print("Playback method: cmd start (detached)")
                return
            except Exception as exc:  # pragma: no cover - runtime fallback
                print(f"cmd start failed: {exc}")

            # Fall back to winsound if start fails.
            try:
                import winsound

                # Play asynchronously so the script doesn't block the shell.
                winsound.PlaySound(str(clip.path), winsound.SND_FILENAME | winsound.SND_ASYNC)
                print("Playback method: winsound (async)")
                return
            except Exception as exc:  # pragma: no cover - playback error
                print(f"winsound playback failed: {exc}")

        # If running under IPython, use its Audio display for inline playback.
        if Audio and display:
            display(Audio(filename=str(clip.path)))
            return

        # Cross-platform fallback: try to open with the system default application.
        if hasattr(os, "startfile"):
            try:
                os.startfile(str(clip.path))
                return
            except Exception as exc:  # pragma: no cover - runtime fallback
                print(f"os.startfile failed: {exc}")

        # macOS / Linux open commands
        opener = None
        if platform.system().lower().startswith("darwin"):
            opener = "open"
        else:
            opener = "xdg-open"

        try:
            # Use Popen so we don't block waiting for the application to exit.
            subprocess.Popen([opener, str(clip.path)])
            return
        except Exception as exc:  # pragma: no cover - final fallback
            print(f"Failed to open file with {opener}: {exc}")

        # If everything else fails, instruct the user how to play manually.
        print(
            "\nCould not automatically play the file. "
            "Open it manually in your audio player."
        )
    except Exception as exc:  # pragma: no cover - unexpected
        print(f"Unexpected error while attempting playback: {exc}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play a random (or specified) audio clip with its label."
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help=f"Path to the dataset root (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--clip",
        default=None,
        help="Relative path inside the audio/ folder to play (e.g., a001_50_60.wav).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for random selection to reproduce a chosen clip.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rng = random.Random(args.seed)

    root = Path(args.data_root).resolve()
    audio_dir = root / AUDIO_SUBDIR
    meta_path = root / META_FILENAME

    meta = read_metadata(meta_path)
    clip = pick_clip(audio_dir, meta, rng, args.clip)
    play_clip(clip)


if __name__ == "__main__":
    main(sys.argv[1:])

