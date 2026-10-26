#!/usr/bin/env bash
# Resume the raw MIMIC-IV 3.1 download from PhysioNet, then check it against SHA256SUMS.txt.
#
#   scripts/fetch_mimic_iv.sh                               # MIMIC-IV 3.1
#   FILES="hosp/omr.csv.gz" scripts/fetch_mimic_iv.sh        # only these files of the release
#   RELEASE=mimic-iv-note/2.2 scripts/fetch_mimic_iv.sh     # MIMIC-IV-Note (its own data use agreement)
#   PARALLEL=6 scripts/fetch_mimic_iv.sh                    # six files at a time
#   scripts/push_mimic_to_fir.sh                            # meanwhile, move finished files to Fir
#
# physionet.org serves each connection at about 50 KB/s (October 2026), so the 10 GB release takes about
# two days one file at a time; six files at a time reach about 200 KB/s in all.
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
BASE="${BASE:-https://physionet.org/files/$RELEASE}"
PARALLEL="${PARALLEL:-1}"
PN_USER="${PN_USER:-}"
mkdir -p "$RAW"
# One run at a time: two runs resume the same partial file at once and interleave their bytes into it
# (on 2026-10-26 a run left in a closed terminal did this to the two largest files).
if [ -f "$RAW/.fetch.pid" ] && kill -0 "$(cat "$RAW/.fetch.pid")" 2>/dev/null; then
  echo "fetch_mimic_iv.sh is already running (pid $(cat "$RAW/.fetch.pid")); stop it before starting another"
  exit 1
fi
echo $$ > "$RAW/.fetch.pid"
[ -n "$PN_USER" ] || read -r -p "PhysioNet username: " PN_USER
read -r -s -p "PhysioNet password for $PN_USER: " PN_PASS; echo

sha() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

# physionet.org answers curl's own user agent with 403 before it looks at the credentials, and a wget
# user agent with 401 (it then asks for them): this uses wget when there is one, else curl presenting
# itself as wget (a Mac has curl but no wget). The credentials go to either in a file only this user can
# read, removed on exit, so they never appear in the process list.
CRED=$(umask 077 && mktemp "${TMPDIR:-/tmp}/physionet.XXXXXX")
trap 'rm -f "$CRED" "$RAW/.fetch.pid"' EXIT
if command -v wget >/dev/null; then
  printf 'user = %s\npassword = %s\n' "$PN_USER" "$PN_PASS" > "$CRED"
  fetch() { wget -q -c --tries=3 --timeout=60 --config="$CRED" -O "$1" "$2"; }
else
  esc() { local v=${1//\\/\\\\}; printf '%s' "${v//\"/\\\"}"; }  # for a quoted curl config value
  printf 'user = "%s:%s"\n' "$(esc "$PN_USER")" "$(esc "$PN_PASS")" > "$CRED"
  fetch() { curl -sS -f -L -C - --retry 3 --connect-timeout 60 -A "Wget/1.21.4" -K "$CRED" -o "$1" "$2"; }
fi
unset PN_PASS

get() {  # get <relative path>: resume a partial file; an error page (403, 401) is never saved as the file.
  # A dropped connection keeps the part already here: retry as long as the file keeps growing, and give up
  # (keeping the part, for the next run) only after 5 attempts in a row add nothing.
  local rel="$1" dest="$RAW/$1" stalled=0 before after
  mkdir -p "$(dirname "$dest")"
  while [ "$stalled" -lt 5 ]; do
    before=$(stat -f %z "$dest" 2>/dev/null || stat -c %s "$dest" 2>/dev/null || echo 0)
    fetch "$dest" "$BASE/$rel" && return 0
    [ -s "$dest" ] || rm -f "$dest"  # an empty file is left behind when the request fails
    after=$(stat -f %z "$dest" 2>/dev/null || stat -c %s "$dest" 2>/dev/null || echo 0)
    if [ "$after" -gt "$before" ]; then stalled=0; else stalled=$((stalled + 1)); fi
    echo "  retry: $rel ($((after / 1000000)) MB so far)"
    sleep 10
  done
  echo "  FAILED for now: $rel (the part downloaded is kept; run this script again to resume)"
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

pushed() { grep -qxF "$1" "$RAW/.pushed" 2>/dev/null; }  # moved to the server by push_mimic_to_fir.sh

fetch_one() {  # fetch_one <relative path>: skip it when complete, else fetch it and check its hash
  local rel="$1" want
  pushed "$rel" && return 0
  want=$(awk -v f="$rel" '$2==f {print $1}' "$RAW/SHA256SUMS.txt")
  if [ -f "$RAW/$rel" ] && [ "$(sha "$RAW/$rel")" = "$want" ]; then
    return 0
  fi
  echo "fetching $rel"
  get "$rel" || return 1  # incomplete: keep the part, do not judge its hash
  if [ -f "$RAW/$rel" ] && [ "$(sha "$RAW/$rel")" != "$want" ]; then
    # resuming onto a corrupt partial file cannot fix it; start this one over
    echo "  hash does not match after resuming; downloading $rel again from the start"
    rm -f "$RAW/$rel"
    get "$rel"
  fi
  echo "done $rel"
}

for rel in "${files[@]}"; do  # PARALLEL files at a time
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 2; done
  fetch_one "$rel" &
done
wait

echo "=== checking ==="
bad=0
for rel in "${files[@]}"; do
  pushed "$rel" && continue
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
