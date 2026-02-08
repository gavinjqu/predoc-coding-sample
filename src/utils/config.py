from __future__ import annotations

import pathlib
import yaml


def load_config(path: str | pathlib.Path) -> dict:
    """Load YAML config into a dict."""
    path = pathlib.Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | pathlib.Path) -> pathlib.Path:
    """Resolve a path relative to the repo root (cwd)."""
    return pathlib.Path(path).expanduser().resolve()
