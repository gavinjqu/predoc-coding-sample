#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Running real-data pipeline ==="
.venv/bin/python3 -m src.cli_real --config configs/config_real.yaml

echo ""
echo "=== Done ==="
echo "Tables:  output/tables/"
echo "Figures: output/figures/"
echo "Metrics: output/metrics/metrics.json"
echo "Log:     output/logs/pipeline_real.log"
