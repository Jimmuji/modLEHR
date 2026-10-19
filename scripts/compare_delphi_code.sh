#!/usr/bin/env bash
# Delphi-2M's own code and modLEHR's plain GPT on the same prepared data, scored by the same evaluation.
#
#   scripts/compare_delphi_code.sh                                           # toy data, CPU, one minute
#   scripts/compare_delphi_code.sh --set model.text_encoding=none            # without the notes Delphi cannot read
#   CONFIG=configs/experiment/synthea_gpt.yaml scripts/compare_delphi_code.sh --seed 1
#   CONFIG=configs/experiment/mimic_iv_core_gpt.yaml DEVICE=cuda scripts/compare_delphi_code.sh --seed 1
#
# Clones Delphi-2M's code (github.com/gerstung-lab/Delphi; MIT licence for the code) at a fixed commit into
# external/delphi, unless DELPHI points at a clone. Then: `modlehr run` trains and scores modLEHR;
# `modlehr export-delphi` writes the same prepared data in Delphi's format with a config of the same size
# and schedule; Delphi's train.py trains on it; `modlehr evaluate` and `modlehr delphi` score its
# checkpoint; `modlehr compare` sets the two side by side. Arguments go to every modlehr command.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PY="${PY:-$ROOT/.venv/bin/python}"
M="$PY -m modlehr"
CONFIG="${CONFIG:-configs/experiment/toy_gpt.yaml}"
DEVICE="${DEVICE:-cpu}"
DELPHI="${DELPHI:-external/delphi}"
DELPHI_COMMIT=fb72166be6b29d8db819227a59487e51c1df1454  # gerstung-lab/Delphi, August 2026
ARGS=("$@" --set "train.device=$DEVICE")

if [ ! -f "$DELPHI/model.py" ]; then
  git clone -q https://github.com/gerstung-lab/Delphi.git "$DELPHI"
  git -C "$DELPHI" checkout -q "$DELPHI_COMMIT"
fi
DELPHI="$(cd "$DELPHI" && pwd)"
if [ "$CONFIG" = configs/experiment/toy_gpt.yaml ] && [ ! -d data/toy_meds ]; then
  $M convert toy --out data/toy_meds
fi
OUT=$($PY -c 'import sys; from modlehr.cli import build_parser, load
print(load(build_parser().parse_args(["train", *sys.argv[1:]])).train.out_dir)' --config "$CONFIG" "${ARGS[@]}")

$M run --config "$CONFIG" "${ARGS[@]}"
$M delphi --config "$CONFIG" "${ARGS[@]}"
$M export-delphi --config "$CONFIG" "${ARGS[@]}"
(cd "$DELPHI" && $PY train.py "$ROOT/${OUT}_delphi_data/delphi_config.py" --device="$DEVICE")
$M evaluate --config "$CONFIG" "${ARGS[@]}" --checkpoint "${OUT}_delphi_code/ckpt.pt" --delphi-repo "$DELPHI" \
  --out "${OUT}_delphi_code/evaluate.csv"
$M delphi --config "$CONFIG" "${ARGS[@]}" --checkpoint "${OUT}_delphi_code/ckpt.pt" --delphi-repo "$DELPHI" \
  --out-dir "${OUT}_delphi_code"
$M compare "$OUT" "${OUT}_delphi_code" | tee "${OUT}_delphi_code/compare.txt"
