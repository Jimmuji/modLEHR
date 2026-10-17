#!/usr/bin/env bash
# Resume the raw MIMIC-IV 3.1 download from PhysioNet, then check it against SHA256SUMS.txt.
#
#   PN_USER=your_physionet_username scripts/fetch_mimic_iv.sh
#
# The password is typed at the prompt and kept in this shell only. MIMIC-IV is credentialed data:
# run this yourself, on a machine you control. Files that are already complete are skipped, and a
# partial file resumes from where it stopped, so the script is safe to re-run after a dropped
# connection (physionet.org drops long transfers often enough that re-running is expected).
#
# RAW is where the files go; it is the directory to hand to scripts/run_mimic_iv.sh.
set -uo pipefail
RAW="${RAW:-$HOME/mimic_dl/physionet.org/files/mimiciv/3.1}"
BASE=https://physionet.org/files/mimiciv/3.1
PN_USER="${PN_USER:?set PN_USER to your PhysioNet username}"
read -r -s -p "PhysioNet password for $PN_USER: " PN_PASS; echo
mkdir -p "$RAW"

sha() { shasum -a 256 "$1" | cut -d' ' -f1; }

get() {  # get <relative path>: resume a partial file. --fail keeps an error page (403) out of the file.
  local rel="$1" dest="$RAW/$1" esc
  mkdir -p "$(dirname "$dest")"
  esc="$PN_USER:$PN_PASS"; esc="${esc//\\/\\\\}"; esc="${esc//\"/\\\"}"
  for attempt in 1 2 3 4 5; do
    # the credentials go to curl on stdin, so they never appear in the process list
    printf 'user = "%s"\n' "$esc" | curl -K - -sS -L --fail -C - --retry 3 --retry-delay 5 --connect-timeout 30 \
         -o "$dest" "$BASE/$rel" && return 0
    echo "  retry $attempt: $rel"
    sleep 10
  done
  echo "  FAILED: $rel"
  return 1
}

# Always fetched fresh (an old copy could be an error page saved by an earlier run); the old one is
# put back if the fetch fails.
[ -f "$RAW/SHA256SUMS.txt" ] && mv -f "$RAW/SHA256SUMS.txt" "$RAW/SHA256SUMS.txt.old"
if ! get SHA256SUMS.txt; then
  [ -f "$RAW/SHA256SUMS.txt.old" ] && mv -f "$RAW/SHA256SUMS.txt.old" "$RAW/SHA256SUMS.txt"
  echo "Cannot read SHA256SUMS.txt: the username or password is wrong, or this account has not signed the MIMIC-IV data use agreement."
  exit 1
fi
rm -f "$RAW/SHA256SUMS.txt.old"

# Everything the MIMIC_IV_MEDS ETL reads, which is every table in the release.
files=()
while read -r rel; do files+=("$rel"); done < <(awk '{print $2}' "$RAW/SHA256SUMS.txt")
echo "${#files[@]} files in the release"

for rel in "${files[@]}"; do
  want=$(awk -v f="$rel" '$2==f {print $1}' "$RAW/SHA256SUMS.txt")
  if [ -f "$RAW/$rel" ] && [ "$(sha "$RAW/$rel")" = "$want" ]; then
    continue
  fi
  echo "fetching $rel"
  get "$rel"
  if [ -f "$RAW/$rel" ] && [ "$(sha "$RAW/$rel")" != "$want" ]; then
    # resuming onto a corrupt partial file cannot fix it; start this one over
    echo "  hash does not match after resuming; downloading $rel again from the start"
    rm -f "$RAW/$rel"
    get "$rel"
  fi
done

echo "=== checking ==="
bad=0
for rel in "${files[@]}"; do
  want=$(awk -v f="$rel" '$2==f {print $1}' "$RAW/SHA256SUMS.txt")
  if [ ! -f "$RAW/$rel" ]; then echo "missing  $rel"; bad=1
  elif [ "$(sha "$RAW/$rel")" != "$want" ]; then echo "bad hash $rel"; bad=1; fi
done
if [ "$bad" = 0 ]; then
  echo "all ${#files[@]} files complete. Next:"
  echo "  RAW=$RAW scripts/run_mimic_iv.sh"
else
  echo "Some files are still incomplete. Re-run this script; it resumes."
  exit 1
fi
