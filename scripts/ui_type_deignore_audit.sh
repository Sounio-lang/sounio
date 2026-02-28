#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

OUT_JSON="${UI_TYPE_DEIGNORE_AUDIT_JSON:-$ROOT_DIR/artifacts/omega/ui_type_deignore_audit.v1.json}"
OUT_MD="${UI_TYPE_DEIGNORE_AUDIT_MD:-$ROOT_DIR/artifacts/omega/ui_type_deignore_audit.md}"

mkdir -p "$(dirname "$OUT_JSON")"
mkdir -p "$(dirname "$OUT_MD")"

tmp_jsonl="$(mktemp)"
trap 'rm -f "$tmp_jsonl"' EXIT

total_ignored=0
safe_candidates=0

for f in tests/ui/type/*.sio; do
  if ! rg -q '^\s*//@\s*ignore' "$f"; then
    continue
  fi
  total_ignored=$((total_ignored + 1))

  pattern="$(rg -m1 '^\s*//@\s*error-pattern:' "$f" | sed -E 's/.*error-pattern:\s*//' || true)"
  if [[ -z "$pattern" ]]; then
    pattern="(none)"
  fi

  check_log="$(mktemp)"
  rc=0
  "$SOUC_BIN" check "$f" >"$check_log" 2>&1 || rc=$?

  pattern_hit=false
  if [[ "$pattern" != "(none)" ]] && rg -qiF "$pattern" "$check_log"; then
    pattern_hit=true
  fi

  safe=false
  if [[ "$rc" -ne 0 && "$pattern_hit" == "true" ]]; then
    safe=true
    safe_candidates=$((safe_candidates + 1))
  fi

  first_line="$(awk 'NF {print; exit}' "$check_log" | tr -d '\r' | sed 's/"/\\"/g')"
  jq -cn \
    --arg file "$f" \
    --arg pattern "$pattern" \
    --argjson rc "$rc" \
    --argjson pattern_hit "$pattern_hit" \
    --argjson safe "$safe" \
    --arg first_line "$first_line" \
    '{
      file: $file,
      exit_code: $rc,
      error_pattern: $pattern,
      pattern_hit: $pattern_hit,
      safe_deignore: $safe,
      first_output_line: $first_line
    }' >>"$tmp_jsonl"
  rm -f "$check_log"
done

jq -s \
  --arg generated_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg souc_bin "$SOUC_BIN" \
  --argjson total_ignored "$total_ignored" \
  --argjson safe_candidates "$safe_candidates" \
  '{
    schema: "sounio.ui-type.deignore-audit.v1",
    generated_at_utc: $generated_at_utc,
    souc_bin: $souc_bin,
    totals: {
      ignored_files: $total_ignored,
      safe_deignore_candidates: $safe_candidates
    },
    entries: .
  }' "$tmp_jsonl" >"$OUT_JSON"

{
  echo "# UI Type De-ignore Audit"
  echo
  echo "- generated_at_utc: $(jq -r '.generated_at_utc' "$OUT_JSON")"
  echo "- souc_bin: $(jq -r '.souc_bin' "$OUT_JSON")"
  echo "- ignored_files: $(jq -r '.totals.ignored_files' "$OUT_JSON")"
  echo "- safe_deignore_candidates: $(jq -r '.totals.safe_deignore_candidates' "$OUT_JSON")"
  echo
  echo "## Safe Candidates"
  jq -r '.entries[] | select(.safe_deignore == true) | "- " + .file + " (pattern: " + .error_pattern + ")"' "$OUT_JSON"
  echo
  echo "## Blocked Samples"
  jq -r '.entries[] | select(.safe_deignore == false) | "- " + .file + " (exit=" + (.exit_code|tostring) + ", pattern_hit=" + (.pattern_hit|tostring) + ", first_line=" + .first_output_line + ")"' "$OUT_JSON" | head -n 20
} >"$OUT_MD"

echo "UI_TYPE_DEIGNORE_AUDIT_PASS"
echo "JSON: $OUT_JSON"
echo "MD:   $OUT_MD"
