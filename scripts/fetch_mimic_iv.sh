#!/usr/bin/env bash
# Resume the raw MIMIC-IV 3.1 download from PhysioNet, then check it against SHA256SUMS.txt.
#
#   scripts/fetch_mimic_iv.sh                               # MIMIC-IV 3.1
#   FILES="hosp/omr.csv.gz" scripts/fetch_mimic_iv.sh        # only these files of the release
#   RELEASE=mimic-iv-note/2.2 scripts/fetch_mimic_iv.sh     # MIMIC-IV-Note (its own data use agreement)
#
# The username (or PN_USER) and password are typed at the prompt and kept in this shell only.
# MIMIC-IV is credentialed data: run this yourself, on a machine you control. Files that are
# already complete are skipped, and a partial file resumes from where it stopped, so the script is
# safe to re-run after a dropped connection (physionet.org drops long transfers often enough that
# re-running is expected).
#
# RAW is where the files go; it is the directory to hand to scripts/run_mimic_iv.sh.
set -uo pipefail
RELEASE="${RELEASE:-mimiciv/3.1}"
RAW="${RAW:-$HOME/mimic_dl/physionet.org/files/$RELEASE}"
BASE="https://physionet.org/files/$RELEASE"
PN_USER="${PN_USER:-}"
[ -n "$PN_USER" ] || read -r -p "PhysioNet username: " PN_USER
read -r -s -p "PhysioNet password for $PN_USER: " PN_PASS; echo
mkdir -p "$RAW"

sha() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

# physionet.org answers curl with 403 before it looks at the credentials, so this uses wget, which its
# own instructions use. The credentials go to wget in a file only this user can read, removed on exit,
# so they never appear in the process list.
command -v wget >/dev/null || { echo "wget is needed (physionet.org refuses curl); on a Mac: brew install wget"; exit 1; }
CRED=$(umask 077 && mktemp "${TMPDIR:-/tmp}/physionet.XXXXXX")
trap 'rm -f "$CRED"' EXIT
printf 'user = %s\npassword = %s\n' "$PN_USER" "$PN_PASS" > "$CRED"
unset PN_PASS

get() {  # get <relative path>: resume a partial file; an error page (403, 401) is never saved as the file
  local rel="$1" dest="$RAW/$1"
  mkdir -p "$(dirname "$dest")"
  for attempt in 1 2 3 4 5; do
    wget -q -c --tries=3 --timeout=60 --config="$CRED" -O "$dest" "$BASE/$rel" && return 0
    [ -s "$dest" ] || rm -f "$dest"  # wget leaves an empty file behind when the request fails
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
  echo "Cannot read $BASE/SHA256SUMS.txt: the username or password is wrong, or this account has not signed this project's data use agreement on physionet.org."
  exit 1
fi
rm -f "$RAW/SHA256SUMS.txt.old"

# Every file in the release (the MIMIC_IV_MEDS ETL reads every table).
files=()
while read -r rel; do files+=("$rel"); done < <(awk '{print $2}' "$RAW/SHA256SUMS.txt")
echo "${#files[@]} files in the release"
if [ -n "${FILES:-}" ]; then  # only the files asked for, each checked against the release's list
  wanted=()
  for rel in $FILES; do
    grep -q " $rel\$" "$RAW/SHA256SUMS.txt" || { echo "$rel is not in this release"; exit 1; }
    wanted+=("$rel")
  done
  files=("${wanted[@]}")
  echo "fetching ${#files[@]} of them: $FILES"
fi

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
  echo "all ${#files[@]} files complete in $RAW"
  if [ "$RELEASE" = mimiciv/3.1 ]; then echo "Next: RAW=$RAW scripts/run_mimic_iv.sh"; fi
else
  echo "Some files are still incomplete. Re-run this script; it resumes."
  exit 1
fi
