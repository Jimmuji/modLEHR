#!/usr/bin/env bash
# Train and evaluate the Synthea baseline and its variants, then compare them.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"  # does not rely on the editable-install .pth
M=".venv/bin/python -m modlehr"
$M prepare --config configs/experiment/synthea_gpt.yaml
$M prepare --config configs/experiment/synthea_gpt_values.yaml
for exp in synthea_gpt synthea_gpt_rope_age synthea_gpt_gqa synthea_gpt_values; do
  $M train --config configs/experiment/$exp.yaml
  $M evaluate --config configs/experiment/$exp.yaml
done
.venv/bin/python scripts/compare_runs.py runs/synthea_gpt runs/synthea_gpt_rope_age runs/synthea_gpt_gqa runs/synthea_gpt_values
