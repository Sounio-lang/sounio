#!/usr/bin/env bash
# Is this PR actually safe to merge?  Answers the question "did the checks that
# matter for THIS diff run and pass", not "is there anything red".
#
#   scripts/dev/pr_merge_ready.sh 2110
#
# Exit 0 = safe.  Exit 1 = not safe, with the reason on stderr.
#
# WHY IT EXISTS.  On 2026-08-25 PR #2110 was merged on the criterion
# "0 pending, 0 failures".  That criterion was satisfied VACUOUSLY: retargeting
# the PR's base through the REST API had not triggered a new pipeline, so of the
# full suite only PR Triage and two others ever ran.  A 29k-line Lean file was
# merged without CI compiling it.  Main's own post-merge run turned out green,
# but that was luck, not process.
#
# "Nothing is red" and "the right things are green" differ exactly when a
# pipeline does not run — which is the case you most need to catch, because the
# evidence of it is an ABSENCE and absences do not show up as failures.

set -euo pipefail

PR="${1:-}"
[[ -n "$PR" ]] || { echo "usage: $0 <pr-number>" >&2; exit 2; }
REPO="${GH_REPO:-Sounio-lang/sounio}"

j() { gh pr view "$PR" --repo "$REPO" --json "$1" --jq "$2"; }

rc=0
fail() { printf 'NOT MERGE-READY: %s\n' "$1" >&2; rc=1; }

state="$(j state '.state')"
[[ "$state" == "OPEN" ]] || { echo "PR #$PR is $state, nothing to check"; exit 0; }

# 1. Nothing still running.
pending="$(j statusCheckRollup '[.statusCheckRollup[]? | select(.status=="IN_PROGRESS" or .status=="QUEUED")] | length')"
[[ "$pending" == "0" ]] || fail "$pending check(s) still running"

# 2. Nothing red.
failed="$(j statusCheckRollup '[.statusCheckRollup[]? | select(.conclusion=="FAILURE" or .conclusion=="CANCELLED" or .conclusion=="TIMED_OUT")] | length')"
[[ "$failed" == "0" ]] || {
    fail "$failed check(s) failed:"
    j statusCheckRollup '.statusCheckRollup[]? | select(.conclusion=="FAILURE" or .conclusion=="CANCELLED" or .conclusion=="TIMED_OUT") | "    \(.name // .context)"' >&2
}

# 3. The pipeline ran at all.  This is the check the other two cannot make.
names="$(j statusCheckRollup '[.statusCheckRollup[]? | (.name // .context)] | join("\n")')"
has() { grep -qxF "$1" <<<"$names"; }
has "Impact" || fail "the Impact gate never ran — the suite it unlocks did not run either, so a green rollup here means nothing"

# 4. The gates this diff's paths require are PRESENT and green.
files="$(gh pr view "$PR" --repo "$REPO" --json files --jq '.files[].path')"
require() {   # require <gate name> <reason>
    local g="$1" why="$2" concl
    if ! has "$g"; then fail "$why, but the '$g' gate is absent from this PR's checks"; return; fi
    concl="$(j statusCheckRollup "[.statusCheckRollup[]? | select((.name // .context)==\"$g\")][0].conclusion")"
    [[ "$concl" == "SUCCESS" ]] || fail "'$g' is required here and concluded '$concl'"
}
grep -q '^formal/lean4/'          <<<"$files" && require "Lean Proofs"  "this PR changes formal/lean4/"
# The active Sounio compiler implementation is under self-hosted/.
grep -qE '^(src/|compiler/|self-hosted/)' <<<"$files" && require "Full Test Suite" "this PR changes compiler sources"
grep -q '^stdlib/'                <<<"$files" && require "Full Test Suite" "this PR changes stdlib/"

if [[ $rc -eq 0 ]]; then
    echo "MERGE-READY: #$PR — Impact ran, $(wc -l <<<"$names") checks reported, none pending, none failed"
fi
exit $rc
