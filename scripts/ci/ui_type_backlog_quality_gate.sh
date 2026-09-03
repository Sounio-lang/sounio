#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

IN_JSON="${UI_TYPE_DEIGNORE_AUDIT_JSON:-$ROOT_DIR/artifacts/omega/ui_type_deignore_audit.v1.json}"
OUT_JSON="${UI_TYPE_BACKLOG_QUALITY_JSON:-$ROOT_DIR/artifacts/omega/ui_type_backlog_quality.v1.json}"

mkdir -p "$(dirname "$OUT_JSON")"

if [[ ! -f "$IN_JSON" ]]; then
  echo "error: missing ui/type audit file: $IN_JSON" >&2
  exit 1
fi

jq '
  def has_e_code: ((.description | try capture("E[0-9]{3}") catch null) != null);
  def has_blocked_prefix: (.description | startswith("BLOCKED:"));

  .entries as $entries |
  {
    schema: "sounio.ui-type.backlog-quality.v1",
    generated_at_utc: (now | todateiso8601),
    source_audit: $IN_JSON,
    totals: {
      entries: ($entries | length),
      still_blocked: ($entries | map(select(.classification == "still-blocked")) | length),
      missing_blocked_prefix: ($entries | map(select(.classification == "still-blocked" and (has_blocked_prefix | not))) | length),
      missing_e_code: ($entries | map(select(.classification == "still-blocked" and (has_e_code | not))) | length)
    },
    violations: {
      missing_blocked_prefix: ($entries | map(select(.classification == "still-blocked" and (has_blocked_prefix | not)) | {file, classification, description})),
      missing_e_code: ($entries | map(select(.classification == "still-blocked" and (has_e_code | not)) | {file, classification, description}))
    }
  }
' --arg IN_JSON "$IN_JSON" "$IN_JSON" > "$OUT_JSON"

status="pass"
if ! jq -e '.totals.missing_blocked_prefix == 0 and .totals.missing_e_code == 0' "$OUT_JSON" >/dev/null; then
  status="fail"
fi

jq --arg status "$status" '. + {status: $status}' "$OUT_JSON" > "$OUT_JSON.tmp"
mv "$OUT_JSON.tmp" "$OUT_JSON"

if [[ "$status" != "pass" ]]; then
  echo "error: ui/type backlog quality violations detected" >&2
  jq '.violations' "$OUT_JSON" >&2
  exit 1
fi

echo "UI_TYPE_BACKLOG_QUALITY_PASS"
echo "JSON: $OUT_JSON"
