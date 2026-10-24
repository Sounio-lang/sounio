#!/usr/bin/env bash
# Runs every lexical probe; one TSV record per probe and engine:
#   probe  engine  rc  condensed output (program output on success, diagnostics on failure)
# Usage: bash run_probes.sh [results.tsv]
#   MADAROS_ELF=path  run that Madaros ELF directly instead of whatever bin/souc
#                     resolves (bin/souc prefers a local build over the committed
#                     ELF; SOUNIO_REQUIRE_COMMITTED_MADAROS=1 forces the committed one)
#   ENGINES="madaros lean_single"  the engines to run (default: both)
# Probes run from a scratch directory, so `run` leaves no a.out in the tree.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SOUC="$ROOT/bin/souc"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
OUT="${1:-$HERE/results.tsv}"
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac
ENGINES="${ENGINES:-madaros lean_single}"
MADAROS_ELF="${MADAROS_ELF:-}"
[ -n "$MADAROS_ELF" ] && MADAROS_ELF="$(cd "$(dirname "$MADAROS_ELF")" && pwd)/$(basename "$MADAROS_ELF")"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK" || exit 1
: > "$OUT"
for f in "$HERE"/probes/*.sio; do
  p=$(basename "$f" .sio)
  for eng in $ENGINES; do
    if [ "$eng" = madaros ]; then
      if [ -n "$MADAROS_ELF" ]; then
        res=$(timeout 60 "$MADAROS_ELF" run "$f" 2>&1 </dev/null); rc=$?
      else
        res=$(timeout 60 "$SOUC" run "$f" 2>&1 </dev/null); rc=$?
      fi
    else
      res=$(SOUNIO_SOUC_ENGINE=lean_single timeout 60 "$SOUC" run "$f" 2>&1 </dev/null); rc=$?
    fi
    if [ $rc -eq 0 ]; then
      if [ "$eng" = madaros ] && [ -n "$MADAROS_ELF" ]; then
        body=$(printf '%s\n' "$res" | sed -n '/^native_v2_compile: emitted/,$p' | tail -n +2 | head -6 | paste -sd '|' -)
      elif [ "$eng" = madaros ]; then
        body=$(printf '%s\n' "$res" | sed -n '/^ *Output:/,$p' | tail -n +2 | head -6 | paste -sd '|' -)
      else
        body=$(printf '%s\n' "$res" | grep -v -e '^provenance:' -e '^madaros prebuilt:' | head -6 | paste -sd '|' -)
      fi
    else
      body=$(printf '%s\n' "$res" | grep -i -e 'error' -e 'unexpected' -e '-->' | grep -v 'compiler exited with status' | head -4 | tr '\t' ' ' | paste -sd '|' -)
    fi
    printf '%s\t%s\t%s\t%s\n' "$p" "$eng" "$rc" "$body" >> "$OUT"
  done
done
wc -l "$OUT"
