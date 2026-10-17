#!/usr/bin/env bash
# Train and evaluate the plain GPT on MIMIC-IV diagnoses and its variants over several seeds, fit
# logistic regression on the same task, then compare everything against the plain GPT.
#
#   scripts/compare_mimic_iv_core.sh                    # seeds 1 2 3
#   SEEDS="1" scripts/compare_mimic_iv_core.sh          # one seed
#   VARIANTS="rope_age moe" scripts/compare_mimic_iv_core.sh
#
# Needs data/mimic_iv_core_meds (scripts/run_mimic_iv_core.sh makes it). Every variant runs at one
# seed before the next seed starts, so a full comparison exists after the first seed. A run that
# already has evaluate.json is skipped and one with only last.pt resumes, so the script can be
# stopped and started again. Extra arguments go to every `modlehr run`, e.g. --set train.device=cuda.
# Gradient boosting is left out: one fit per target and landmark is too slow at this size.
#
# Credentialed data: the output is summary numbers only (counts, losses, AUCs).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"  # does not rely on the editable-install .pth
M=".venv/bin/python -m modlehr"
SEEDS="${SEEDS:-1 2 3}"
VARIANTS="${VARIANTS:-rope_age alibi_age local_age gqa mla qknorm rmsnorm swiglu moe}"
BASE=mimic_iv_core_gpt
CONFIG="configs/experiment/$BASE.yaml"
EXTRA=("$@")

run_one() {  # run_one <experiment> <seed>
  local dir="runs/${1}_s$2"
  if [ -f "$dir/evaluate.json" ]; then
    echo "[compare] $dir is already evaluated; skipping"
  elif [ -f "$dir/last.pt" ]; then
    $M run --config "configs/experiment/$1.yaml" --seed "$2" --resume ${EXTRA[@]+"${EXTRA[@]}"}
  else
    $M run --config "configs/experiment/$1.yaml" --seed "$2" ${EXTRA[@]+"${EXTRA[@]}"}
  fi
}

compare_all() {
  local runs=() exp seed
  for exp in "$BASE" $(for v in $VARIANTS; do echo "${BASE}_$v"; done); do
    for seed in $SEEDS; do
      [ -f "runs/${exp}_s$seed/evaluate.json" ] && runs+=("runs/${exp}_s$seed")
    done
  done
  [ -f "runs/${BASE}_logistic/evaluate.json" ] && runs+=("runs/${BASE}_logistic")
  $M compare "${runs[@]}" | tee "runs/compare_${BASE}.txt"
}

for seed in $SEEDS; do
  run_one "$BASE" "$seed"  # the first run prepares the data if needed
  if [ ! -f "runs/${BASE}_logistic/evaluate.json" ]; then
    $M baseline --config "$CONFIG" --models logistic
  fi
  # the plain GPT and logistic regression also under Delphi-2M's control definition
  if [ ! -f "runs/${BASE}_s$seed/evaluate_never.json" ]; then
    $M evaluate --config "$CONFIG" --seed "$seed" --set evaluate.controls=never --out "runs/${BASE}_s$seed/evaluate_never.csv"
  fi
  if [ ! -f "runs/${BASE}_never_logistic/evaluate.json" ]; then
    $M baseline --config "$CONFIG" --models logistic --set evaluate.controls=never --set "train.out_dir=runs/${BASE}_never"
  fi
  for v in $VARIANTS; do
    run_one "${BASE}_$v" "$seed"
  done
  compare_all
done
