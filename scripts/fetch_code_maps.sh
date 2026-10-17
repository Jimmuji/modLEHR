#!/usr/bin/env bash
# Download the public disease-code maps that the icd_gems and phecode groupings read, into
# data/maps/ (or MAPS), and check each against the hash of the copy the configs were made with.
#
#   scripts/fetch_code_maps.sh
#
#   data/maps/gems/2018_I9gem.txt                            CMS 2018 ICD-9-CM → ICD-10-CM GEMs
#   data/maps/phecode/Phecode_map_v1_2_icd9_icd10cm.csv      PheWAS catalog phecode map v1.2,
#                                                            ICD-9-CM and ICD-10-CM in one file
#
# A hash mismatch is a warning, not an error: the publisher may have revised the file, and the
# numbers in the README were made with the hashed version.
set -uo pipefail
cd "$(dirname "$0")/.."
MAPS="${MAPS:-data/maps}"

GEMS_URL=https://www.cms.gov/Medicare/Coding/ICD10/Downloads/2018-ICD-10-CM-General-Equivalence-Mappings.zip
GEMS_SHA=1c5e5f14026ace48437a0d1c485d282fb5d171c357c6ab4f7798fc0a1e3624a2
# The PheWAS catalog serves its files from a Shiny app; if this address stops working, the file is
# under "Phecode Map 1.2 (Combined)" at https://phewascatalog.org/phewas/
PHECODE_URL=https://phewascatalog.org/phewas/_w_e6c65ff10cc8441baeafb41174923a10/data/Phecode_map_v1_2_icd9_icd10cm.csv.zip
PHECODE_SHA=561468d40352da2b4af6a3e3e89582f1a00de7a70736fad741259861f75958eb

fetch() {  # fetch <dir> <url> <sha256> <file to extract>
  local dir="$MAPS/$1" zip="$MAPS/$1/$(basename "$2")"
  mkdir -p "$dir"
  if [ -f "$dir/$4" ]; then echo "$dir/$4 is already there"; return 0; fi
  curl -sS -L --fail --max-time 300 -o "$zip" "$2" || { echo "could not download $2"; return 1; }
  [ "$(shasum -a 256 "$zip" | cut -d' ' -f1)" = "$3" ] || echo "warning: $zip differs from the version the configs were made with"
  unzip -q -o "$zip" "$4" -d "$dir" && echo "$dir/$4"
}

status=0
fetch gems "$GEMS_URL" "$GEMS_SHA" 2018_I9gem.txt || status=1
fetch phecode "$PHECODE_URL" "$PHECODE_SHA" Phecode_map_v1_2_icd9_icd10cm.csv || status=1
exit $status
