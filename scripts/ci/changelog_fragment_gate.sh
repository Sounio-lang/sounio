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
# What this gate enforces beyond the name/kind census:
#   1. A compiler/runtime/stdlib PR may not edit CHANGELOG.md. The release
#      notes are assembled from fragments at RC time; a hand edit bypasses the
#      fragment review and collides with the assembler. Release-assembly PRs
#      touch only CHANGELOG.md and changelog.d/, classify as docs-only and
#      are not bound by this rule.
#   2. A renamed or copied file that lands a new changelog.d/ path counts as
#      an added fragment (diff-filter ACR, not just A): with git's default
#      rename detection a pure rename never appears under A.
#   3. Every fragment the diff adds must be present and non-empty at the head
#      commit; an empty file satisfies the census while saying nothing.
#   4. PR_CREATED_AT must be an ISO 8601 UTC timestamp (YYYY-MM-DDTHH:MM:SSZ)
#      to earn the pre-cutoff exemption; a set-but-malformed value fails
#      closed instead of being string-compared into an accidental exemption.
#
# The CHANGELOG.md audit runs with rename detection DISABLED: with it on,
# `git diff --name-only` reports only the destination of a rename, so
# renaming CHANGELOG.md into changelog.d/ would hide the source path from
# the audit. With it off, a rename appears as delete+add and both paths are
# audited. The ACR fragment census keeps rename detection, on purpose.
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
CREATED_RE='^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$'

# decide <needs:true|false> <created_at> <added paths, newline-separated> <changed paths, newline-separated>
# `added` is the ACR-filtered fragment census; `changed` is the full
# base...head path list (all diff filters, rename detection disabled so a
# renamed-away CHANGELOG.md still names its source path). Prints a verdict
# line; returns 0 on pass, 1 on fail.
decide() {
  local needs=$1 created=$2 added=$3 changed=$4 path kind n=0 bad=""
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
  if [[ "$needs" == true ]] && grep -qxF "CHANGELOG.md" <<< "$changed"; then
    echo "CHANGELOG_FRAGMENT_FAIL CHANGELOG.md edited by a compiler/runtime/stdlib PR; add changelog.d/<slug>.<kind>.md instead (ADR-018 §4: only the release manager edits CHANGELOG.md, via scripts/release/changelog_assemble.sh)"
    return 1
  fi
  if [[ "$needs" != true ]]; then
    echo "CHANGELOG_FRAGMENT_PASS not-required fragments=$n"; return 0
  fi
  if [[ -n "$created" ]]; then
    if [[ "$created" =~ $CREATED_RE ]]; then
      if [[ "$created" < "$CUTOFF" ]]; then
        echo "CHANGELOG_FRAGMENT_PASS exempt opened=$created cutoff=$CUTOFF fragments=$n"; return 0
      fi
    else
      echo "CHANGELOG_FRAGMENT_WARN malformed PR_CREATED_AT='$created'; pre-cutoff exemption not applied" >&2
    fi
  fi
  if (( n == 0 )); then
    echo "CHANGELOG_FRAGMENT_FAIL compiler/runtime/stdlib changed and no changelog.d/<slug>.<kind>.md was added (use kind=internal if nothing is user-visible; see changelog.d/README.md)"
    return 1
  fi
  echo "CHANGELOG_FRAGMENT_PASS fragments=$n"
}

# Every fragment the diff adds must exist at <head> and carry non-whitespace
# content. Uses git cat-file so it does not depend on the checkout state.
# Returns 0 when all fragments are non-empty, 1 otherwise.
check_fragments_nonempty() {
  local head=$1 added=$2 path size content fails=0
  while IFS= read -r path; do
    [[ -n "$path" && "$path" == changelog.d/* && "$path" != changelog.d/README.md ]] || continue
    if ! git cat-file -e "$head:$path" 2>/dev/null; then
      echo "CHANGELOG_FRAGMENT_FAIL $path: not present at head $head" >&2
      fails=$((fails + 1)); continue
    fi
    size=$(git cat-file -s "$head:$path")
    content=$(git cat-file blob "$head:$path" | tr -d '[:space:]')
    if [[ "$size" -eq 0 || -z "$content" ]]; then
      echo "CHANGELOG_FRAGMENT_FAIL $path: fragment is empty; say what changed, even for kind=internal (changelog.d/README.md)" >&2
      fails=$((fails + 1))
    fi
  done <<< "$added"
  (( fails == 0 ))
}

selftest() {
  local fails=0
  check() { # <expect pass|fail> <label> <needs> <created> <added> <changed>
    local out rc=0
    out=$(decide "$3" "$4" "$5" "$6") || rc=$?
    if [[ ( $1 == pass && $rc == 0 ) || ( $1 == fail && $rc != 0 ) ]]; then
      echo "ok   $2: $out"
    else
      echo "FAIL $2: expected $1, got rc=$rc: $out"; fails=$((fails + 1))
    fi
  }
  local new=2026-11-01T00:00:00Z old=2026-10-20T00:00:00Z
  check pass "compiler + added fragment"     true  $new $'self-hosted/x.sio\nchangelog.d/fix-x.fixed.md' $'self-hosted/x.sio\nchangelog.d/fix-x.fixed.md'
  check pass "compiler + internal fragment"  true  $new 'changelog.d/refactor-y.internal.md' 'changelog.d/refactor-y.internal.md'
  check fail "compiler, no fragment"         true  $new 'self-hosted/x.sio' 'self-hosted/x.sio'
  check pass "compiler, no fragment, old PR" true  $old 'self-hosted/x.sio' 'self-hosted/x.sio'
  check pass "docs only, no fragment"        false $new 'docs/a.md' 'docs/a.md'
  check fail "unknown kind"                  true  $new 'changelog.d/a.bugfix.md' 'changelog.d/a.bugfix.md'
  check fail "unknown kind, even if exempt"  true  $old 'changelog.d/a.bugfix.md' 'changelog.d/a.bugfix.md'
  check fail "bad name (uppercase)"          false $new 'changelog.d/Fix.fixed.md' 'changelog.d/Fix.fixed.md'
  check fail "README is not a fragment"      true  $new 'changelog.d/README.md' 'changelog.d/README.md'
  check pass "numerical-correction kind"     true  $new 'changelog.d/atan.numerical-correction.md' 'changelog.d/atan.numerical-correction.md'
  check fail "fragment outside changelog.d"  true  $new 'docs/fix.fixed.md' 'docs/fix.fixed.md'
  # Hardening: CHANGELOG.md hand edits.
  check fail "compiler PR edits CHANGELOG.md"        true  $new $'changelog.d/fix-x.fixed.md' $'CHANGELOG.md\nchangelog.d/fix-x.fixed.md'
  check fail "compiler PR edits CHANGELOG.md, no fragment" true $new 'self-hosted/x.sio' $'CHANGELOG.md\nself-hosted/x.sio'
  check fail "old PR still cannot edit CHANGELOG.md" true  $old 'changelog.d/fix-x.fixed.md' $'CHANGELOG.md\nchangelog.d/fix-x.fixed.md'
  check pass "release assembly edits CHANGELOG.md"   false $new $'CHANGELOG.md\nchangelog.d/a.fixed.md' $'CHANGELOG.md\nchangelog.d/a.fixed.md'
  # The audit sees delete+add with rename detection off, so a CHANGELOG.md
  # renamed INTO a fragment still names its source path and is refused.
  check fail "CHANGELOG.md renamed into a fragment"  true  $new 'changelog.d/archive.fixed.md' $'CHANGELOG.md\nchangelog.d/archive.fixed.md'
  # Hardening: malformed PR_CREATED_AT fails closed (no exemption).
  check fail "malformed created_at, no exemption"    true  'Oct 20 2026' 'self-hosted/x.sio' 'self-hosted/x.sio'
  check fail "offset created_at, no exemption"       true  '2026-10-20T00:00:00+00:00' 'self-hosted/x.sio' 'self-hosted/x.sio'
  check pass "malformed created_at, fragment added"  true  'not-a-date' 'changelog.d/fix-x.fixed.md' 'changelog.d/fix-x.fixed.md'

  # Hardening: fragment non-emptiness against a synthetic git history.
  local repo rc=0
  repo=$(mktemp -d)
  (
    cd "$repo"
    git init -q
    git config user.email selftest@example.invalid
    git config user.name selftest
    mkdir -p changelog.d
    echo "Fixed the thing." > changelog.d/good.fixed.md
    : > changelog.d/empty.fixed.md
    printf '  \n\t\n' > changelog.d/blank.internal.md
    git add changelog.d
    git commit -qm fragments
  )
  local head
  head=$(git -C "$repo" rev-parse HEAD)
  if (cd "$repo" && check_fragments_nonempty "$head" 'changelog.d/good.fixed.md') >/dev/null 2>&1; then
    echo "ok   non-empty fragment accepted"
  else
    echo "FAIL non-empty fragment accepted"; fails=$((fails + 1))
  fi
  if (cd "$repo" && check_fragments_nonempty "$head" 'changelog.d/empty.fixed.md') >/dev/null 2>&1; then
    echo "FAIL empty fragment refused"; fails=$((fails + 1))
  else
    echo "ok   empty fragment refused"
  fi
  if (cd "$repo" && check_fragments_nonempty "$head" 'changelog.d/blank.internal.md') >/dev/null 2>&1; then
    echo "FAIL whitespace-only fragment refused"; fails=$((fails + 1))
  else
    echo "ok   whitespace-only fragment refused"
  fi
  if (cd "$repo" && check_fragments_nonempty "$head" 'changelog.d/missing.fixed.md') >/dev/null 2>&1; then
    echo "FAIL missing fragment refused"; fails=$((fails + 1))
  else
    echo "ok   missing fragment refused"
  fi
  rm -rf "$repo"

  if (( fails )); then echo "CHANGELOG_FRAGMENT_SELFTEST_FAIL failures=$fails"; return 1; fi
  echo "CHANGELOG_FRAGMENT_SELFTEST_PASS cases=23"
}

if [[ "${1:-}" == "--selftest" ]]; then selftest; exit; fi

base=${CI_BASE_SHA:?CI_BASE_SHA is required}
head=${CI_HEAD_SHA:?CI_HEAD_SHA is required}
needs=false
for v in "${IMPACT_COMPILER:-false}" "${IMPACT_RUNTIME:-false}" "${IMPACT_STDLIB:-false}"; do
  [[ "$v" == true ]] && needs=true
done

# The diff is this gate's only evidence; a failed git diff must be loud, not
# an empty census that reads as "no fragment added" (or, worse, "docs-only").
diff_err=$(mktemp)
if ! added=$(git diff --name-only --diff-filter=ACR "$base...$head" 2>"$diff_err"); then
  echo "CHANGELOG_FRAGMENT_FAIL git diff $base...$head failed -- cannot census fragments:" >&2
  sed 's/^/  git: /' "$diff_err" >&2
  rm -f "$diff_err"
  exit 3
fi
if ! changed=$(git -c diff.renames=false diff --name-only "$base...$head" 2>"$diff_err"); then
  echo "CHANGELOG_FRAGMENT_FAIL git diff $base...$head failed -- cannot audit CHANGELOG.md:" >&2
  sed 's/^/  git: /' "$diff_err" >&2
  rm -f "$diff_err"
  exit 3
fi
rm -f "$diff_err"

decide "$needs" "${PR_CREATED_AT:-}" "$added" "$changed"
check_fragments_nonempty "$head" "$added"
