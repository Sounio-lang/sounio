#!/usr/bin/env bash
# Changelog fragment gate (ADR-018 §4).
#
# A pull request that touches the compiler, runtime or stdlib (as classified by
# classify_ci_impact.sh) must add exactly the fragments it means to publish:
# one or more files changelog.d/<slug>.<kind>.md. `internal` is the explicit
# way to say "no user-visible effect". Fragments are separate files, so they
# never conflict in the merge queue; only the release manager edits
# CHANGELOG.md (scripts/release/changelog_assemble.sh).
#
# PRs opened before CUTOFF are exempt: the gate must not fail work that other
# agents already had in flight when the rule was introduced.
#
# Usage:
#   changelog_fragment_gate.sh            # reads CI env (see below)
#   changelog_fragment_gate.sh --selftest
#
# Env: CI_BASE_SHA, CI_HEAD_SHA, IMPACT_COMPILER, IMPACT_RUNTIME,
#      IMPACT_STDLIB, PR_CREATED_AT (ISO 8601, UTC).
set -euo pipefail

CUTOFF="2026-10-25T03:00:00Z"
KINDS="breaking refusal semantics numerical-correction numerical added changed deprecated removed fixed performance security internal"
NAME_RE='^changelog\.d/[a-z0-9][a-z0-9._-]*\.([a-z-]+)\.md$'

# decide <needs:true|false> <created_at> <added paths, newline-separated>
# Prints a verdict line; returns 0 on pass, 1 on fail.
decide() {
  local needs=$1 created=$2 added=$3 path kind n=0 bad=""
  while IFS= read -r path; do
    [[ -n "$path" && "$path" == changelog.d/* && "$path" != changelog.d/README.md ]] || continue
    if [[ "$path" =~ $NAME_RE ]]; then
      kind=${BASH_REMATCH[1]}
      if [[ " $KINDS " == *" $kind "* ]]; then n=$((n + 1)); else bad+=" $path(kind=$kind)"; fi
    else
      bad+=" $path(name)"
    fi
  done <<< "$added"
  if [[ -n "$bad" ]]; then
    echo "CHANGELOG_FRAGMENT_FAIL invalid fragment(s):$bad; kinds: $KINDS"
    return 1
  fi
  if [[ "$needs" != true ]]; then
    echo "CHANGELOG_FRAGMENT_PASS not-required fragments=$n"; return 0
  fi
  if [[ -n "$created" && "$created" < "$CUTOFF" ]]; then
    echo "CHANGELOG_FRAGMENT_PASS exempt opened=$created cutoff=$CUTOFF fragments=$n"; return 0
  fi
  if (( n == 0 )); then
    echo "CHANGELOG_FRAGMENT_FAIL compiler/runtime/stdlib changed and no changelog.d/<slug>.<kind>.md was added (use kind=internal if nothing is user-visible; see changelog.d/README.md)"
    return 1
  fi
  echo "CHANGELOG_FRAGMENT_PASS fragments=$n"
}

selftest() {
  local fails=0
  check() { # <expect pass|fail> <label> <needs> <created> <added>
    local out rc=0
    out=$(decide "$3" "$4" "$5") || rc=$?
    if [[ ( $1 == pass && $rc == 0 ) || ( $1 == fail && $rc != 0 ) ]]; then
      echo "ok   $2: $out"
    else
      echo "FAIL $2: expected $1, got rc=$rc: $out"; fails=$((fails + 1))
    fi
  }
  local new=2026-11-01T00:00:00Z old=2026-10-20T00:00:00Z
  check pass "compiler + added fragment"     true  $new $'self-hosted/x.sio\nchangelog.d/fix-x.fixed.md'
  check pass "compiler + internal fragment"  true  $new 'changelog.d/refactor-y.internal.md'
  check fail "compiler, no fragment"         true  $new 'self-hosted/x.sio'
  check pass "compiler, no fragment, old PR" true  $old 'self-hosted/x.sio'
  check pass "docs only, no fragment"        false $new 'docs/a.md'
  check fail "unknown kind"                  true  $new 'changelog.d/a.bugfix.md'
  check fail "unknown kind, even if exempt"  true  $old 'changelog.d/a.bugfix.md'
  check fail "bad name (uppercase)"          false $new 'changelog.d/Fix.fixed.md'
  check fail "README is not a fragment"      true  $new 'changelog.d/README.md'
  check pass "numerical-correction kind"     true  $new 'changelog.d/atan.numerical-correction.md'
  check fail "fragment outside changelog.d"  true  $new 'docs/fix.fixed.md'
  if (( fails )); then echo "CHANGELOG_FRAGMENT_SELFTEST_FAIL failures=$fails"; return 1; fi
  echo "CHANGELOG_FRAGMENT_SELFTEST_PASS cases=11"
}

if [[ "${1:-}" == "--selftest" ]]; then selftest; exit; fi

base=${CI_BASE_SHA:?CI_BASE_SHA is required}
head=${CI_HEAD_SHA:?CI_HEAD_SHA is required}
needs=false
for v in "${IMPACT_COMPILER:-false}" "${IMPACT_RUNTIME:-false}" "${IMPACT_STDLIB:-false}"; do
  [[ "$v" == true ]] && needs=true
done
added=$(git diff --name-only --diff-filter=A "$base...$head")
decide "$needs" "${PR_CREATED_AT:-}" "$added"
