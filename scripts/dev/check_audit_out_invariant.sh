#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

AUDIT="scripts/dev/worktree_branch_audit.sh"

if [[ ! -f "$AUDIT" ]]; then
  echo "missing audit script: $AUDIT" >&2
  exit 1
fi

failed=0

# Invariant 1: the vulnerable pre-bc0513783 form must not return. That version
# parsed the output path as OUT="${1:-}", so an invocation like
# `worktree_branch_audit.sh --check` silently set OUT="--check" and wrote the
# TSV inventory into ./--check at the repo root.
if grep -nE 'OUT="\$\{1:-\}"' "$AUDIT" >/dev/null; then
  echo "regression: $AUDIT uses OUT=\"\${1:-}\" which swallows flags as the output path"
  failed=1
fi

# Invariant 2: --check must be intercepted as a real flag (CHECK_MODE), not
# reachable as a positional OUT path. A dedicated case guarantees the token
# can never become the write target.
if ! grep -nE '^[[:space:]]*--check\)' "$AUDIT" >/dev/null; then
  echo "regression: $AUDIT has no dedicated --check) flag case"
  failed=1
fi

# Invariant 3: unknown leading-dash tokens must be rejected so they cannot
# become OUT through some future positional fallback.
if ! grep -nE '^[[:space:]]*-\*\)' "$AUDIT" >/dev/null; then
  echo "regression: $AUDIT has no -*) unknown-option rejection"
  failed=1
fi

# Dynamic check (fast, no full audit run): an unknown flag must exit non-zero
# before any worktree scanning happens. This proves the -*) branch is live.
if bash "$AUDIT" --bogus-audit-invariant >/dev/null 2>&1; then
  echo "regression: $AUDIT accepted an unknown --bogus flag"
  failed=1
fi

# Dynamic check (fast): --help must succeed without writing any TSV and without
# leaving a ./--check residue behind.
bash "$AUDIT" --help >/dev/null 2>&1 || {
  echo "regression: $AUDIT --help did not exit 0"
  failed=1
}
if [[ -e "$ROOT_DIR/--check" ]]; then
  echo "regression: $AUDIT left a ./--check residue at the repo root"
  failed=1
fi

if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

echo "audit OUT-path invariant check passed"
