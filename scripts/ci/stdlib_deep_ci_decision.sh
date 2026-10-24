#!/usr/bin/env bash
# Additive required-job decision. The existing CI evaluator remains unchanged.
set -euo pipefail
script=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
if [[ ${1:-} == selftest ]]; then
 base='{"impact":{"outputs":{"compiler":"false","runtime":"false","stdlib":"true","tests":"false","full":"false"}},"stdlib-deep-science":{"result":"success"}}'
 NEEDS_JSON="$base" "$script"
 for state in failure cancelled skipped missing; do
  altered=$(jq --arg state "$state" '.["stdlib-deep-science"].result=$state' <<< "$base")
  if NEEDS_JSON="$altered" "$script" >/dev/null 2>&1; then echo "decision accepted selected $state" >&2; exit 1; fi
 done
 unselected=$(jq '.impact.outputs.stdlib="false"|.["stdlib-deep-science"].result="skipped"' <<< "$base")
 NEEDS_JSON="$unselected" "$script"
 if NEEDS_JSON='{}' "$script" >/dev/null 2>&1; then echo 'decision accepted missing impact' >&2; exit 1; fi
 printf 'STDLIB_DEEP_CI_DECISION_SELFTEST_PASS\n'
 exit 0
fi
[[ $# == 0 ]] || { echo 'usage: stdlib_deep_ci_decision.sh [selftest]' >&2; exit 64; }
: "${NEEDS_JSON:?NEEDS_JSON is required}"
selected=$(jq -r '
 if (.impact.outputs|type)!="object" then error("missing impact outputs") else
 [.impact.outputs.compiler,.impact.outputs.runtime,.impact.outputs.stdlib,.impact.outputs.tests,.impact.outputs.full] as $flags |
 if any($flags[]; .!="true" and .!="false") then error("invalid impact flags") else any($flags[]; .=="true") end end
' <<< "$NEEDS_JSON")
result=$(jq -r '.["stdlib-deep-science"].result // "missing"' <<< "$NEEDS_JSON")
if [[ "$selected" == true ]]; then
 [[ "$result" == success ]] || { printf 'CI_DECISION_FAIL: selected stdlib-deep-science ended as %s\n' "$result" >&2; exit 1; }
else
 [[ "$result" == success || "$result" == skipped ]] || { printf 'CI_DECISION_FAIL: unselected stdlib-deep-science ended as %s\n' "$result" >&2; exit 1; }
fi
printf 'STDLIB_DEEP_CI_DECISION_PASS selected=%s result=%s\n' "$selected" "$result"
