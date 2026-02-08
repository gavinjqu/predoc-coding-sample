from __future__ import annotations

import logging

from src.analysis.figures import FIGURE_REGISTRY
from src.analysis.tables import TABLE_REGISTRY


def run(config: dict) -> None:
    report_cfg = config["params"]["report"]

    for table_name in report_cfg["tables"]:
        func = TABLE_REGISTRY.get(table_name)
        if func is None:
            logging.warning("Unknown table '%s'; skipping", table_name)
            continue
        logging.info("Generating table: %s", table_name)
        func(config)

    for fig_name in report_cfg["figures"]:
        func = FIGURE_REGISTRY.get(fig_name)
        if func is None:
            logging.warning("Unknown figure '%s'; skipping", fig_name)
            continue
        logging.info("Generating figure: %s", fig_name)
        func(config)
