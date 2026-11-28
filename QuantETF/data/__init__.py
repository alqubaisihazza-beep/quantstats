"""Data layer: market data, dividends, validation and persistence."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load config.yaml (or $QUANTETF_CONFIG)."""
    path = Path(path or os.getenv("QUANTETF_CONFIG") or ROOT / "config.yaml")
    if not path.is_absolute():
        path = ROOT / path
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def resolve_path(path: str | os.PathLike) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
