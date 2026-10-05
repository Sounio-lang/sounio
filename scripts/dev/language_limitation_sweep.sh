#!/usr/bin/env bash
# scripts/dev/language_limitation_sweep.sh — measure which .sio sources the
# default engine actually accepts, and why the rest are rejected.
#
# Produces one TSV row per file:
#   file  rc  first_error_code  error_count  parse_errors  first_error_line  all_error_codes
#
# `first_error_code` is what the compiler reported first; `all_error_codes` is
# the full multiset (`E035x3,E019x2`, sorted by descending count). Sizing a work
# front off the first code alone undercounts every class that tends to be
# reported later in a file.
#
# This is a measurement, not a gate: it never fails on a rejected file. It exists
# so that claims about language limitations are derived from the compiler's
# behaviour rather than from documentation drift.
#
# Usage:
#   scripts/dev/language_limitation_sweep.sh [--out PATH] [--jobs N] [TREE ...]
#
# TREE defaults to: stdlib examples tests/run-pass
#
# Heavy CPU: this fans out one compiler process per file over ~1.5k files.
# Per SOUNIO_HEAVY_CPU_POLICY.md, run it on Slurm `cpu-ops`, not in the
# workspace control pod.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

OUT=""
JOBS="${SWEEP_JOBS:-8}"
PER_FILE_TIMEOUT="${SWEEP_TIMEOUT:-120}"
TREES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)  OUT="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --timeout) PER_FILE_TIMEOUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) TREES+=("$1"); shift ;;
  esac
done

if [[ ${#TREES[@]} -eq 0 ]]; then
  TREES=(stdlib examples tests/run-pass)
fi

if [[ -z "$OUT" ]]; then
  OUT="$ROOT_DIR/artifacts/audit/language_limitation_sweep_$(date -u +%Y%m%d).tsv"
fi

sounio_require_souc

# Pin the compiler for the whole run. A sweep that starts on one binary and
# finishes on another is not a measurement; if a local build lands mid-run the
# TSV silently mixes two engines. SOUNIO_MADAROS_BIN keeps the bin/souc wrapper
# (and its ulimit handling) while fixing which ELF it execs.
: "${SOUNIO_MADAROS_BIN:=$ROOT_DIR/bin/madaros-linux-x86_64}"
export SOUNIO_MADAROS_BIN
if [[ ! -x "$SOUNIO_MADAROS_BIN" ]]; then
  echo "error: pinned compiler not executable: $SOUNIO_MADAROS_BIN" >&2
  exit 2
fi
ENGINE_ID="$(basename "$SOUNIO_MADAROS_BIN")@$(md5sum "$SOUNIO_MADAROS_BIN" 2>/dev/null | cut -c1-8)"

mkdir -p "$(dirname "$OUT")"

# Classify one file. Written to be safe under `xargs -P`: each invocation emits
# exactly one line, short enough for an atomic append.
sweep_one() {
  local f="$1"
  local out msg rc code count parse line

  msg="$(cd "$ROOT_DIR" && timeout "$PER_FILE_TIMEOUT" "$SOUC_BIN" check "$f" 2>&1)" && rc=0 || rc=$?

  # `error[E019\n]` — the code is rendered with an embedded newline, so match the
  # digits directly rather than a well-formed `error[E019]`.
  code="$(printf '%s' "$msg" | grep -o 'error\[E[0-9]\+' | head -1 | sed 's/error\[//')"
  count="$(printf '%s' "$msg" | grep -c 'error\[E[0-9]\+' || true)"
  # The driver reports parse failure in three different shapes.
  parse="$(printf '%s' "$msg" | grep -c -e 'parse error' -e 'failed to parse' -e 'parse_failed=true' || true)"
  line="$(printf '%s' "$msg" | grep -o 'at line [0-9]\+' | head -1 | sed 's/at line //')"

  # Full multiset of codes, most frequent first, as CODExN joined by commas.
  all="$(printf '%s' "$msg" \
    | grep -o 'error\[E[0-9]\+' | sed 's/error\[//' \
    | sort | uniq -c | sort -k1 -rn \
    | awk '{printf "%s%sx%s", (NR>1 ? "," : ""), $2, $1} END {print ""}')"
  if [[ "$parse" -gt 0 ]]; then
    if [[ -n "$all" ]]; then all="PARSEx$parse,$all"; else all="PARSEx$parse"; fi
  fi

  if [[ -z "$code" && "$parse" -gt 0 ]]; then
    code="PARSE"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$f" "$rc" "${code:-none}" "${count:-0}" "${parse:-0}" "${line:-}" "${all:-}"
}
export -f sweep_one
export ROOT_DIR SOUC_BIN PER_FILE_TIMEOUT

echo "[sweep] souc: $SOUC_BIN"
echo "[sweep] engine: $ENGINE_ID"
echo "[sweep] trees: ${TREES[*]}"
echo "[sweep] jobs: $JOBS  timeout: ${PER_FILE_TIMEOUT}s"

FILE_LIST="$(mktemp)"
trap 'rm -f "$FILE_LIST"' EXIT
(cd "$ROOT_DIR" && find "${TREES[@]}" -name '*.sio' -type f 2>/dev/null | sort) > "$FILE_LIST"
TOTAL="$(wc -l < "$FILE_LIST")"
echo "[sweep] files: $TOTAL"

BODY="$(mktemp)"
# shellcheck disable=SC2016
xargs -a "$FILE_LIST" -P "$JOBS" -I{} bash -c 'sweep_one "$@"' _ {} > "$BODY"

{
  printf '# engine\t%s\n' "$ENGINE_ID"
  printf '# trees\t%s\n' "${TREES[*]}"
  printf 'file\trc\tfirst_error_code\terror_count\tparse_errors\tfirst_error_line\tall_error_codes\n'
  sort "$BODY"
} > "$OUT"
rm -f "$BODY"

echo "[sweep] wrote $OUT"
echo
echo "[sweep] accepted / rejected:"
awk -F'\t' '/^#/ || /^file\t/ {next} {n[$2]++} END {for (k in n) printf "  rc=%s\t%d\n", k, n[k]}' "$OUT" | sort

echo
echo "[sweep] rejection classes (first error per file):"
awk -F'\t' '/^#/ || /^file\t/ {next} $2!=0 {n[$3]++} END {for (k in n) printf "  %s\t%d\n", k, n[k]}' "$OUT" | sort -k2 -rn

echo
echo "[sweep] files affected per class (any position in the file):"
awk -F'\t' '/^#/ || /^file\t/ {next} $2!=0 && $7 != "" {
  m = split($7, parts, ",")
  for (i = 1; i <= m; i++) { split(parts[i], kv, "x"); files[kv[1]]++; hits[kv[1]] += kv[2] }
} END {
  for (k in files) printf "  %-8s %5d files  %6d occurrences\n", k, files[k], hits[k]
}' "$OUT" | sort -k2 -rn
