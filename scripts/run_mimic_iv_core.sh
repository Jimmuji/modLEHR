#!/usr/bin/env bash
# MIMIC-IV diagnoses, end to end: convert the three core hosp tables to MEDS, then prepare, train,
# evaluate under both control conventions, fit the classical baselines and compare.
#
#   RAW=~/mimic_dl/physionet.org/files/mimiciv/3.1 scripts/run_mimic_iv_core.sh
#
# This path needs only hosp/patients.csv.gz, hosp/admissions.csv.gz and hosp/diagnoses_icd.csv.gz
# (about 50 MB), and gives the Delphi-2M data shape on MIMIC: sex, disease codes and death. It is
# not a replacement for scripts/run_mimic_iv.sh, which runs the official ETL over every table and
# adds labs, medications, procedures and ICU stays; use this one while that download finishes, and
# as the no-measurements comparison afterwards.
#
# Credentialed data: run this in your own terminal, and do not paste its row-level output into
# online tools (the PhysioNet data use agreement). Summary numbers such as mean AUC are fine.
#
# Extra arguments go to `modlehr run`, for example: scripts/run_mimic_iv_core.sh --set train.device=cuda
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
RAW="${RAW:?set RAW to the MIMIC-IV 3.1 directory that holds hosp/}"
OUT="${OUT:-data/mimic_iv_core_meds}"
SEED="${SEED:-1}"
CONFIG=configs/experiment/mimic_iv_core_gpt.yaml
M=".venv/bin/python -m modlehr"

missing=0
for f in hosp/patients.csv.gz hosp/admissions.csv.gz hosp/diagnoses_icd.csv.gz; do
  [ -f "$RAW/$f" ] || { echo "missing: $RAW/$f"; missing=1; }
done
[ "$missing" = 1 ] && { echo "Fetch the missing tables with PN_USER=... scripts/fetch_mimic_iv.sh"; exit 1; }

[ -f "$OUT/metadata/codes.parquet" ] || $M convert mimic-iv-core --raw-dir "$RAW" --out "$OUT"
$M validate --meds-dir "$OUT"
$M run --config $CONFIG --seed "$SEED" "$@"
$M info --config $CONFIG

# The same checkpoint under Delphi-2M's control definition (controls never have the disease),
# written beside the default one, which keeps only subjects still at risk at the end of the horizon.
RUN="runs/mimic_iv_core_gpt_s${SEED}"
$M evaluate --config $CONFIG --seed "$SEED" --set evaluate.controls=never --out "$RUN/evaluate_never.csv"

$M baseline --config $CONFIG
$M compare "$RUN" runs/mimic_iv_core_gpt_logistic runs/mimic_iv_core_gpt_gbm
