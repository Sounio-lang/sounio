#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

FILE=".claude/PARALLEL_BLOCKER_CONTRACT.md"

if [[ ! -f "$FILE" ]]; then
  echo "error: required file missing: $FILE" >&2
  exit 1
fi

search_fixed() {
  local file="$1"
  local literal="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -F -q "$literal" "$file"
  else
    grep -F -q -- "$literal" "$file"
  fi
}

assert_literal() {
  local literal="$1"
  if ! search_fixed "$FILE" "$literal"; then
    echo "error: missing required literal in $FILE: $literal" >&2
    exit 1
  fi
}

assert_literal "# Parallel Blocker Contract"
assert_literal "Status: active"
assert_literal "## Evidence Levels"
assert_literal "## Severity"
assert_literal "## Failure Classes"
assert_literal "## Required Blocker Record"
assert_literal "## Ownership Rules"
assert_literal "## Parallel Lane Contract"
assert_literal "## Merge Contract"
assert_literal "## Handoff Contract"
assert_literal "## Mandatory Stop Conditions"
assert_literal "Blocker-ID: BLK-YYYYMMDD-<lane>-<slug>"
assert_literal "Severity: B0 | B1 | B2 | B3 | B4"
assert_literal "Evidence-Level: E0 | E1 | E2 | E3 | E4"
assert_literal "Class: state-sync | ownership-conflict | gate-regression | harness-routing | compiler-semantics | bootstrap-runtime | platform-resource | dependency-secret | evidence-gap | governance-offload | doc-claim | security-privacy"
assert_literal "Files-Owned:"
assert_literal "Acceptance-Gate:"
assert_literal "Next-Action:"
assert_literal "git rev-parse HEAD main origin/main"

assert_literal ".claude/PARALLEL_BLOCKER_CONTRACT.md"

if ! search_fixed AGENTS.md ".claude/PARALLEL_BLOCKER_CONTRACT.md"; then
  echo "error: AGENTS.md must reference $FILE" >&2
  exit 1
fi

if ! search_fixed .claude/AGENT_HANDOFF.md ".claude/PARALLEL_BLOCKER_CONTRACT.md"; then
  echo "error: .claude/AGENT_HANDOFF.md must reference $FILE" >&2
  exit 1
fi

if ! search_fixed .claude/OPERATIONAL_CANONICAL_INDEX.md ".claude/PARALLEL_BLOCKER_CONTRACT.md"; then
  echo "error: .claude/OPERATIONAL_CANONICAL_INDEX.md must reference $FILE" >&2
  exit 1
fi

echo "PARALLEL_BLOCKER_CONTRACT_PASS"
