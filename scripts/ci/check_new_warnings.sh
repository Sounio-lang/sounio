#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

if [[ "$SKIP_BUILD" = "1" ]]; then
  echo "[warnings] skipped (SKIP_BUILD=1, requires cargo)"
  exit 0
fi

BASELINE_FILE="${ROOT_DIR}/scripts/baselines/lib_warning_signatures.txt"
MODE="${1:-check}"

if [[ "$MODE" != "check" && "$MODE" != "--refresh" ]]; then
  echo "usage: $0 [check|--refresh]"
  exit 2
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
RAW_WARNINGS="$TMP_DIR/raw_warnings.log"
CURRENT_SIG="$TMP_DIR/current_signatures.txt"
NORMALIZED_BASELINE="$TMP_DIR/baseline_signatures.txt"
NEW_WARNINGS="$TMP_DIR/new_warnings.txt"
RESOLVED_WARNINGS="$TMP_DIR/resolved_warnings.txt"

extract_warning_signatures() {
  local input_file="$1"
  local output_file="$2"
  rg ': warning: ' "$input_file" \
    | sed -E 's#^([^:]+):[0-9]+:[0-9]+: warning: (.*)$#\1: warning: \2#' \
    | sort -u >"$output_file"
}

echo "[warnings] collecting lib warning signatures"
(cd "$ROOT_DIR" && cargo test -p souc --lib --no-run --message-format=short) 2>&1 | tee "$RAW_WARNINGS"
extract_warning_signatures "$RAW_WARNINGS" "$CURRENT_SIG"

mkdir -p "$(dirname "$BASELINE_FILE")"

if [[ "$MODE" == "--refresh" ]]; then
  cp "$CURRENT_SIG" "$BASELINE_FILE"
  echo "[warnings] baseline refreshed: $BASELINE_FILE"
  exit 0
fi

if [[ ! -f "$BASELINE_FILE" ]]; then
  echo "[warnings] missing baseline file: $BASELINE_FILE"
  echo "[warnings] run '$0 --refresh' to create it."
  exit 1
fi

sort -u "$BASELINE_FILE" >"$NORMALIZED_BASELINE"

comm -13 "$NORMALIZED_BASELINE" "$CURRENT_SIG" >"$NEW_WARNINGS" || true
comm -23 "$NORMALIZED_BASELINE" "$CURRENT_SIG" >"$RESOLVED_WARNINGS" || true

if [[ -s "$NEW_WARNINGS" ]]; then
  echo "[warnings] new warning signatures detected:"
  cat "$NEW_WARNINGS"
  echo
  echo "[warnings] if this is intentional, update baseline with:"
  echo "  scripts/ci/check_new_warnings.sh --refresh"
  exit 1
fi

if [[ -s "$RESOLVED_WARNINGS" ]]; then
  echo "[warnings] resolved warning signatures (baseline can be refreshed):"
  cat "$RESOLVED_WARNINGS"
fi

echo "[warnings] no new lib warning signatures."
