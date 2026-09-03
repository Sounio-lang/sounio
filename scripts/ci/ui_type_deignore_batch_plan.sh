#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

IN_JSON="${UI_TYPE_DEIGNORE_AUDIT_JSON:-$ROOT_DIR/artifacts/omega/ui_type_deignore_audit.v1.json}"
OUT_JSON="${UI_TYPE_DEIGNORE_BATCH_PLAN_JSON:-$ROOT_DIR/artifacts/omega/ui_type_deignore_batch_plan.v1.json}"
OUT_MD="${UI_TYPE_DEIGNORE_BATCH_PLAN_MD:-$ROOT_DIR/artifacts/omega/ui_type_deignore_batch_plan.md}"

mkdir -p "$(dirname "$OUT_JSON")" "$(dirname "$OUT_MD")"

if [[ ! -f "$IN_JSON" ]]; then
  echo "error: missing audit input: $IN_JSON" >&2
  exit 1
fi

jq '
  def extract_code:
    ((.description | try capture("(?<code>E[0-9]{3})") catch null | .code) // "NO_CODE");
  def blocker_category:
    if .description | test("cannot parse|no unit registry"; "i") then "parser_or_frontend_gap"
    elif .exit_code == 0 then "pinned_binary_semantic_gap"
    else "diagnostic_alignment_gap"
    end;
  def next_action:
    if .classification == "ready" then "deignore_now"
    elif blocker_category == "parser_or_frontend_gap" then "bootstrap_frontend_then_retest"
    elif blocker_category == "diagnostic_alignment_gap" then "fix_error_pattern_or_checker_mapping"
    else "wait_for_stage1_selfhost_enforcement"
    end;

  .entries as $entries |
  ($entries | map(
    . + {
      checker_code: extract_code,
      blocker_category: blocker_category,
      next_action: next_action,
      reclassify_candidate: (
        (.classification != "ready")
        and blocker_category != "parser_or_frontend_gap"
        and ((.exit_code != 0 and .pattern_hit == false) or (.error_pattern == "(none)"))
      )
    }
  )) as $enriched |
  {
    schema: "sounio.ui-type.deignore.batch-plan.v1",
    generated_at_utc: (now | todateiso8601),
    source_audit: $IN_JSON,
    totals: {
      ignored_files: ($enriched | length),
      ready: ($enriched | map(select(.classification == "ready")) | length),
      still_blocked: ($enriched | map(select(.classification == "still-blocked")) | length),
      needs_fix: ($enriched | map(select(.classification == "needs-fix")) | length),
      reclassify_candidates: ($enriched | map(select(.reclassify_candidate == true)) | length)
    },
    by_next_action: (
      $enriched
      | group_by(.next_action)
      | map({next_action: (.[0].next_action), count: length, files: map(.file), checker_codes: (map(.checker_code) | unique | sort)})
      | sort_by(.count) | reverse
    ),
    by_checker_code: (
      $enriched
      | group_by(.checker_code)
      | map({checker_code: (.[0].checker_code), count: length, next_actions: (map(.next_action) | unique | sort), files: map(.file)})
      | sort_by(.count) | reverse
    ),
    reclassify_candidates: ($enriched | map(select(.reclassify_candidate == true) | {file, classification, exit_code, pattern_hit, error_pattern, description, next_action})),
    entries: $enriched
  }
' --arg IN_JSON "$IN_JSON" "$IN_JSON" > "$OUT_JSON"

{
  echo "# UI Type De-ignore Batch Plan"
  echo
  echo "- generated_at_utc: $(jq -r '.generated_at_utc' "$OUT_JSON")"
  echo "- source_audit: $(jq -r '.source_audit' "$OUT_JSON")"
  echo "- ignored_files: $(jq -r '.totals.ignored_files' "$OUT_JSON")"
  echo "- ready: $(jq -r '.totals.ready' "$OUT_JSON")"
  echo "- needs_fix: $(jq -r '.totals.needs_fix' "$OUT_JSON")"
  echo "- still_blocked: $(jq -r '.totals.still_blocked' "$OUT_JSON")"
  echo "- reclassify_candidates: $(jq -r '.totals.reclassify_candidates' "$OUT_JSON")"
  echo
  echo "## Action Batches"
  jq -r '.by_next_action[] | "- " + .next_action + ": " + (.count|tostring) + " files (codes: " + (.checker_codes | join(", ")) + ")"' "$OUT_JSON"
  echo
  echo "## Top Checker Codes"
  jq -r '.by_checker_code[:10][] | "- " + .checker_code + ": " + (.count|tostring) + " files"' "$OUT_JSON"
  echo
  echo "## Reclassify Candidates"
  if jq -e '.reclassify_candidates | length > 0' "$OUT_JSON" >/dev/null; then
    jq -r '.reclassify_candidates[] | "- " + .file + " (exit=" + (.exit_code|tostring) + ", pattern_hit=" + (.pattern_hit|tostring) + ", action=" + .next_action + ")"' "$OUT_JSON"
  else
    echo "- (none)"
  fi
} > "$OUT_MD"

echo "UI_TYPE_DEIGNORE_BATCH_PLAN_PASS"
echo "JSON: $OUT_JSON"
echo "MD:   $OUT_MD"
