#!/usr/bin/env bash
# Train and evaluate the plain GPT on Synthea and its variants over several seeds, fit the
# classical baselines on the same task, then compare everything against the plain GPT.
#
#   scripts/compare_synthea.sh                 # seeds 1 2 3
#   SEEDS="1" scripts/compare_synthea.sh       # one seed, about a quarter of the time
#
# Each run writes runs/<experiment>_s<seed>/. Prepare runs once per data config.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"  # does not rely on the editable-install .pth
M=".venv/bin/python -m modlehr"
SEEDS="${SEEDS:-1 2 3}"
EXPERIMENTS="synthea_gpt synthea_gpt_rope_age synthea_gpt_alibi_age synthea_gpt_gqa synthea_gpt_values"

runs=()
for exp in $EXPERIMENTS; do
  for seed in $SEEDS; do
    $M run --config "configs/experiment/$exp.yaml" --seed "$seed"
    runs+=("runs/${exp}_s$seed")
  done
done
$M baseline --config configs/experiment/synthea_gpt.yaml
$M baseline --config configs/experiment/synthea_gpt_values.yaml
$M compare "${runs[@]}" runs/synthea_gpt_logistic runs/synthea_gpt_gbm \
  runs/synthea_gpt_values_logistic runs/synthea_gpt_values_gbm
