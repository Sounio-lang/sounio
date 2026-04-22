#!/usr/bin/env bash
# Download raw BrainVision rsEEG for a list of LEMON subjects from GWDG mirror.
#
# Usage:
#   lemon_download.sh <subject_list.txt> <dest_dir> [N_PARALLEL]
# Each subject yields three files (vhdr/vmrk/eeg), ~317 MB total.
# Skips files already present with matching Content-Length.
set -euo pipefail

LIST="${1:-/workspace/data/lemon/subjects_all.txt}"
DEST="${2:-/workspace/data/lemon/raw_eeg}"
N="${3:-3}"
BASE="https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID"

mkdir -p "$DEST"
export DEST BASE

fetch_one() {
    local sub="$1"
    local out="$DEST/$sub/RSEEG"
    mkdir -p "$out"
    for ext in vhdr vmrk eeg; do
        local url="$BASE/$sub/RSEEG/$sub.$ext"
        local f="$out/$sub.$ext"
        # skip if size matches remote
        local remote_len
        remote_len=$(curl -sIL --max-time 30 "$url" \
            | tr -d '\r' \
            | awk 'BEGIN{IGNORECASE=1} /^content-length:/ {l=$2} END {print l+0}')
        if [[ -f "$f" && -n "$remote_len" && "$(stat -c%s "$f")" == "$remote_len" ]]; then
            echo "[skip] $sub/$ext already complete ($remote_len bytes)"
            continue
        fi
        echo "[get ] $sub/$ext ($remote_len bytes)"
        curl -sSL --max-time 1800 --retry 3 --retry-delay 5 -o "$f" "$url"
    done
}
export -f fetch_one

cat "$LIST" | xargs -n1 -P"$N" -I{} bash -c 'fetch_one "$@"' _ {}
