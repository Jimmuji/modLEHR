#!/usr/bin/env bash
# Full MIMIC-IV, end to end: convert with the official MIMIC_IV_MEDS ETL, then prepare, train,
# evaluate, fit the classical baselines and compare.
#
#   RAW=~/mimic_dl/physionet.org/files/mimiciv/3.1 scripts/run_mimic_iv.sh
#
# With clinical notes, also set NOTES to the MIMIC-IV-Note folder that holds discharge.csv.gz and
# radiology.csv.gz (a separate PhysioNet module); the notes are merged into the MEDS data and the
# notes experiment runs next to the plain one:
#   RAW=... NOTES=~/mimic_dl/physionet.org/files/mimic-iv-note/2.2/note scripts/run_mimic_iv.sh
#
# RAW is the MIMIC-IV 3.1 directory that holds hosp/ and icu/. Credentialed data: run this in
# your own terminal, and do not paste its data or row-level output into online tools (the
# PhysioNet data use agreement). Summary numbers such as mean AUC are fine to share.
#
# The ETL takes about 32 minutes and 20 GB of memory. Extra arguments go to `modlehr run`,
# for example: scripts/run_mimic_iv.sh --set train.precision=bf16 --set train.device=cuda
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
RAW="${RAW:?set RAW to the MIMIC-IV 3.1 directory that holds hosp/ and icu/}"
OUT="${OUT:-data/mimic_iv_meds}"
M=".venv/bin/python -m modlehr"

missing=0
for f in hosp/patients.csv.gz hosp/admissions.csv.gz hosp/diagnoses_icd.csv.gz hosp/labevents.csv.gz \
         hosp/prescriptions.csv.gz icu/icustays.csv.gz icu/chartevents.csv.gz; do
  [ -f "$RAW/$f" ] || { echo "missing: $RAW/$f"; missing=1; }
done
if [ "$missing" = 1 ]; then
  cat <<'MSG'
The raw MIMIC-IV download is incomplete. Resume it with your PhysioNet account (wget asks for the
password and skips files that are already complete):
  cd ~/mimic_dl && wget -r -N -c -np --user YOUR_PHYSIONET_USERNAME --ask-password https://physionet.org/files/mimiciv/3.1/
Or let the ETL download and verify the files itself (see the MIMIC_IV_MEDS README).
MSG
  exit 1
fi

if [ ! -f "$OUT/metadata/codes.parquet" ]; then
  if [ ! -x .venv-etl/bin/meds-extract-run ]; then
    uv venv --python 3.11 .venv-etl
    uv pip install --python .venv-etl/bin/python MIMIC_IV_MEDS
  fi
  PATH="$PWD/.venv-etl/bin:$PATH" meds-extract-run spec=MIMIC-IV output_dir="$PWD/$OUT" do_download=false input_dir="$RAW"
fi

$M validate --meds-dir "$OUT"
$M run --config configs/experiment/mimic_iv_gpt.yaml --seed 1 "$@"
$M info --config configs/experiment/mimic_iv_gpt.yaml
$M baseline --config configs/experiment/mimic_iv_gpt.yaml
runs=(runs/mimic_iv_gpt_s1 runs/mimic_iv_gpt_logistic runs/mimic_iv_gpt_gbm)

if [ -n "${NOTES:-}" ]; then
  if [ ! -f data/mimic_iv_meds_notes/metadata/codes.parquet ]; then
    $M convert mimic-iv-note --raw-dir "$NOTES" --meds-dir "$OUT" --out data/mimic_iv_meds_notes
  fi
  $M run --config configs/experiment/mimic_iv_gpt_notes.yaml --seed 1 "$@"
  runs+=(runs/mimic_iv_gpt_notes_s1)
fi
$M compare "${runs[@]}"
