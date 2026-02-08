from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _clean_outputs() -> None:
    for rel in ["output/tables", "output/figures", "output/metrics", "output/logs"]:
        path = ROOT / rel
        if path.exists():
            shutil.rmtree(path)


def _run_pipeline_fast() -> None:
    env = os.environ.copy()
    env["FAST"] = "1"
    subprocess.run(
        [sys.executable, "-m", "src.cli", "--config", "configs/config.yaml", "--fast"],
        cwd=ROOT,
        check=True,
        env=env,
    )


@pytest.fixture(scope="session")
def run_pipeline() -> None:
    _clean_outputs()
    _run_pipeline_fast()
