#!/usr/bin/env bash
# stale_never_closes_gate.sh — the stale bot may label, never close.
#
# Issues are closed only with proof: a fixing commit and its test, the
# reproduction run on current main, the canonical duplicate, or the path that
# no longer exists. On 2026-10-09..12 actions/stale (days-before-close: 7)
# closed 17 open defect reports as "not_planned" purely for inactivity
# (#2190, #2218, #2219, #2240, #2241, #2249, #2293, #2306, #2354, #2355,
# #2363, #2368, #2371, #2375, #2377, #2380, #2388). This gate fails if any
# actions/stale step in .github/workflows can close an issue or a PR.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
fail=0
found=0
for f in .github/workflows/*.yml .github/workflows/*.yaml; do
    [[ -f "$f" ]] || continue
    grep -q 'actions/stale@' "$f" || continue
    found=1
    for key in days-before-close days-before-issue-close days-before-pr-close; do
        while IFS= read -r line; do
            v="$(printf '%s' "$line" | sed -E 's/.*:[[:space:]]*//; s/[[:space:]]*(#.*)?$//; s/["'"'"']//g')"
            if [[ "$v" != "-1" ]]; then
                echo "FAIL $f: $key: $v (must be -1: the stale bot may label but never close)" >&2
                fail=1
            fi
        done < <(grep -E "^[[:space:]]*$key:" "$f" || true)
    done
    if ! grep -qE '^[[:space:]]*days-before-close:[[:space:]]*-1' "$f"; then
        echo "FAIL $f: actions/stale without days-before-close: -1 (its default closes after 7 days)" >&2
        fail=1
    fi
done
# Emptiness guard: with no actions/stale step there is nothing to measure, and a
# gate that passes on nothing is not a gate. If the workflow is deleted on
# purpose, delete this gate with it.
if [[ "$found" -lt 1 ]]; then
    echo "FAIL: nothing to measure: no workflow uses actions/stale (found_stale_step=$found)" >&2
    fail=1
fi
if [[ "$fail" == 0 ]]; then
    echo "PASS stale automation never closes (found_stale_step=$found)"
    echo "STALE_NEVER_CLOSES_GATE_OK"
    exit 0
fi
echo "STALE_NEVER_CLOSES_GATE_FAIL" >&2
exit 1
