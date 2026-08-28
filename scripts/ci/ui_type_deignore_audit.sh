#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

OUT_JSON="${UI_TYPE_DEIGNORE_AUDIT_JSON:-$ROOT_DIR/artifacts/omega/ui_type_deignore_audit.v1.json}"
OUT_MD="${UI_TYPE_DEIGNORE_AUDIT_MD:-$ROOT_DIR/artifacts/omega/ui_type_deignore_audit.md}"
OUT_BUCKET_JSON="${UI_TYPE_DEIGNORE_BUCKET_JSON:-$ROOT_DIR/artifacts/omega/ui_type_deignore_backlog_buckets.v1.json}"

mkdir -p "$(dirname "$OUT_JSON")"
mkdir -p "$(dirname "$OUT_MD")"
mkdir -p "$(dirname "$OUT_BUCKET_JSON")"

tmp_jsonl="$(mktemp)"
trap 'rm -f "$tmp_jsonl"' EXIT

total_ignored=0
safe_candidates=0
ready_count=0
needs_fix_count=0
still_blocked_count=0

for f in tests/ui/type/*.sio; do
  if ! rg -q '^\s*//@\s*ignore' "$f"; then
    continue
  fi
  total_ignored=$((total_ignored + 1))

  pattern="$(rg -m1 '^\s*//@\s*error-pattern:' "$f" | sed -E 's/.*error-pattern:\s*//' || true)"
  description="$(rg -m1 '^\s*//@\s*description:' "$f" | sed -E 's/.*description:\s*//' || true)"
  if [[ -z "$pattern" ]]; then
    pattern="(none)"
  fi
  if [[ -z "$description" ]]; then
    description="(none)"
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

  description_blocked=false
  if [[ "$description" == BLOCKED:* ]]; then
    description_blocked=true
  fi

  classification="needs-fix"
  if [[ "$safe" == "true" ]]; then
    classification="ready"
    ready_count=$((ready_count + 1))
  elif [[ "$description_blocked" == "true" ]]; then
    classification="still-blocked"
    still_blocked_count=$((still_blocked_count + 1))
  else
    needs_fix_count=$((needs_fix_count + 1))
  fi

  first_line="$(awk 'NF {print; exit}' "$check_log" | tr -d '\r' | sed 's/"/\\"/g')"
  jq -cn \
    --arg file "$f" \
    --arg pattern "$pattern" \
    --arg description "$description" \
    --arg classification "$classification" \
    --argjson rc "$rc" \
    --argjson pattern_hit "$pattern_hit" \
    --argjson safe "$safe" \
    --argjson description_blocked "$description_blocked" \
    --arg first_line "$first_line" \
    '{
      file: $file,
      exit_code: $rc,
      error_pattern: $pattern,
      description: $description,
      pattern_hit: $pattern_hit,
      safe_deignore: $safe,
      classification: $classification,
      description_blocked: $description_blocked,
      first_output_line: $first_line,
      evidence: {
        exit_code: $rc,
        pattern: $pattern,
        pattern_hit: $pattern_hit,
        first_output_line: $first_line
      }
    }' >>"$tmp_jsonl"
  rm -f "$check_log"
done

jq -s \
  --arg generated_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg souc_bin "$SOUC_BIN" \
  --argjson total_ignored "$total_ignored" \
  --argjson safe_candidates "$safe_candidates" \
  --argjson ready_count "$ready_count" \
  --argjson needs_fix_count "$needs_fix_count" \
  --argjson still_blocked_count "$still_blocked_count" \
  '{
    schema: "sounio.ui-type.deignore-audit.v1",
    generated_at_utc: $generated_at_utc,
    souc_bin: $souc_bin,
    totals: {
      ignored_files: $total_ignored,
      safe_deignore_candidates: $safe_candidates,
      ready_count: $ready_count,
      needs_fix_count: $needs_fix_count,
      still_blocked_count: $still_blocked_count
    },
    entries: .
  }' "$tmp_jsonl" >"$OUT_JSON"

jq \
  '{
    schema: "sounio.ui-type.deignore.backlog-buckets.v1",
    generated_at_utc,
    souc_bin,
    totals: .totals,
    buckets: {
      ready: (.entries | map(select(.classification == "ready"))),
      needs_fix: (.entries | map(select(.classification == "needs-fix"))),
      still_blocked: (.entries | map(select(.classification == "still-blocked")))
    }
  }' "$OUT_JSON" >"$OUT_BUCKET_JSON"

{
  echo "# UI Type De-ignore Audit"
  echo
  echo "- generated_at_utc: $(jq -r '.generated_at_utc' "$OUT_JSON")"
  echo "- souc_bin: $(jq -r '.souc_bin' "$OUT_JSON")"
  echo "- ignored_files: $(jq -r '.totals.ignored_files' "$OUT_JSON")"
  echo "- safe_deignore_candidates: $(jq -r '.totals.safe_deignore_candidates' "$OUT_JSON")"
  echo "- ready_count: $(jq -r '.totals.ready_count' "$OUT_JSON")"
  echo "- needs_fix_count: $(jq -r '.totals.needs_fix_count' "$OUT_JSON")"
  echo "- still_blocked_count: $(jq -r '.totals.still_blocked_count' "$OUT_JSON")"
  echo
  echo "## Bucket Summary"
  echo
  echo "- ready: $(jq -r '.totals.ready_count' "$OUT_JSON")"
  echo "- needs-fix: $(jq -r '.totals.needs_fix_count' "$OUT_JSON")"
  echo "- still-blocked: $(jq -r '.totals.still_blocked_count' "$OUT_JSON")"
  echo
  echo "## Ready"
  jq -r '.entries[] | select(.classification == "ready") | "- " + .file + " (pattern: " + .error_pattern + ")"' "$OUT_JSON"
  echo
  echo "## Needs Fix"
  jq -r '.entries[] | select(.classification == "needs-fix") | "- " + .file + " (exit=" + (.exit_code|tostring) + ", pattern_hit=" + (.pattern_hit|tostring) + ", first_line=" + .first_output_line + ")"' "$OUT_JSON" | head -n 40
  echo
  echo "## Still Blocked"
  jq -r '.entries[] | select(.classification == "still-blocked") | "- " + .file + " (" + .description + ")"' "$OUT_JSON" | head -n 40
  echo
  echo "## Safe Candidates"
  jq -r '.entries[] | select(.safe_deignore == true) | "- " + .file + " (pattern: " + .error_pattern + ")"' "$OUT_JSON"
  echo
  echo "## Blocked Samples"
  jq -r '.entries[] | select(.safe_deignore == false) | "- " + .file + " (exit=" + (.exit_code|tostring) + ", pattern_hit=" + (.pattern_hit|tostring) + ", first_line=" + .first_output_line + ")"' "$OUT_JSON" | head -n 20
} >"$OUT_MD"

echo "UI_TYPE_DEIGNORE_AUDIT_PASS"
echo "JSON: $OUT_JSON"
echo "BUCKET JSON: $OUT_BUCKET_JSON"
echo "MD:   $OUT_MD"
