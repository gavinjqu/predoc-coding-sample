from __future__ import annotations

import pathlib
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader


def run(config: dict) -> None:
    """Compatibility wrapper that loads 05_report.py."""
    path = pathlib.Path(__file__).with_name("05_report.py")
    loader = SourceFileLoader("report05", str(path))
    spec = spec_from_loader(loader.name, loader)
    module = module_from_spec(spec)
    loader.exec_module(module)  # type: ignore
    module.run(config)  # type: ignore
