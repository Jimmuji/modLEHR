#!/usr/bin/env bash
# modLEHR's plain GPT, modLEHR's Delphi-2M (model.architecture: delphi2m) and Delphi-2M's own code, on the same
# prepared data and scored by the same evaluation. The two Delphi-2M runs should agree: on a CPU they end
# with the same weights.
#
#   scripts/compare_delphi_code.sh                                           # toy data, CPU, one minute
#   scripts/compare_delphi_code.sh --set model.text_encoding=none            # without the notes Delphi cannot read
#   CONFIG=configs/experiment/synthea_gpt.yaml scripts/compare_delphi_code.sh --seed 1
#   CONFIG=configs/experiment/mimic_iv_core_gpt.yaml DEVICE=cuda scripts/compare_delphi_code.sh --seed 1
#
# Clones Delphi-2M's code (github.com/gerstung-lab/Delphi; MIT licence for the code) at a fixed commit into
# .cache/delphi, unless DELPHI points at a clone. Then: `modlehr run` trains and scores the GPT, and again
# with model.architecture=delphi2m; `modlehr export-delphi` writes the same prepared data in Delphi's format
# with a config of the same size and schedule; Delphi's train.py trains on it; `modlehr evaluate` and
# `modlehr delphi` score its checkpoint; `modlehr compare` sets the three side by side. Arguments go to every
# modlehr command.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PY="${PY:-$ROOT/.venv/bin/python}"
M="$PY -m modlehr"
CONFIG="${CONFIG:-configs/experiment/toy_gpt.yaml}"
DEVICE="${DEVICE:-cpu}"
DELPHI="${DELPHI:-.cache/delphi}"
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
out_dir() {  # the run directory these arguments give (with --seed N, ending in _sN); with --base, before the seed
  $PY -c 'import sys; from modlehr.cli import build_parser, load; from modlehr.config import apply_overrides, load_config
base = sys.argv[1] == "--base"; a = build_parser().parse_args(["train", *sys.argv[1 + base:]])
print((apply_overrides(load_config(a.config), a.set) if base else load(a)).train.out_dir)' "$@"
}
OUT=$(out_dir --config "$CONFIG" "${ARGS[@]}")
OURS=("${ARGS[@]}" --set model.architecture=delphi2m --set "train.out_dir=$(out_dir --base --config "$CONFIG" "${ARGS[@]}")_delphi2m")
OUT2=$(out_dir --config "$CONFIG" "${OURS[@]}")

$M run --config "$CONFIG" "${ARGS[@]}"
$M delphi --config "$CONFIG" "${ARGS[@]}"
$M run --config "$CONFIG" "${OURS[@]}"
$M delphi --config "$CONFIG" "${OURS[@]}"
$M export-delphi --config "$CONFIG" "${ARGS[@]}"
(cd "$DELPHI" && $PY train.py "$ROOT/${OUT}_delphi_data/delphi_config.py" --device="$DEVICE")
$M evaluate --config "$CONFIG" "${ARGS[@]}" --checkpoint "${OUT}_delphi_code/ckpt.pt" \
  --out "${OUT}_delphi_code/evaluate.csv"
$M delphi --config "$CONFIG" "${ARGS[@]}" --checkpoint "${OUT}_delphi_code/ckpt.pt" \
  --out-dir "${OUT}_delphi_code"
$M compare "$OUT" "$OUT2" "${OUT}_delphi_code" | tee "${OUT}_delphi_code/compare.txt"
