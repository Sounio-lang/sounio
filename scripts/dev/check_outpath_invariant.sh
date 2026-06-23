#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Output-path dev scripts that share the `[--check] [OUT.tsv]` convention.
# Each must reject leading-dash tokens so a flag like --check can never be
# silently captured as the write target (the class of bug that once wrote the
# worktree inventory to ./--check at the repo root; see bc0513783 and
# docs/audit/WORKTREE_AUDIT_CHECK_LEAK_2026-06-23.md).
OUTPATH_SCRIPTS=(
  scripts/dev/worktree_branch_audit.sh
  scripts/dev/madaros_pr_resolution_queue.sh
)

failed=0

check_script() {
  local script="$1"

  if [[ ! -f "$script" ]]; then
    echo "missing output-path script: $script" >&2
    failed=1
    return
  fi

  # Invariant 1: the vulnerable positional form must not (re)appear. The
  # pre-fix worktree audit used OUT="${1:-}", so `... --check` set OUT=--check.
  if grep -nE 'OUT="\$\{1:-\}"' "$script" >/dev/null; then
    echo "regression: $script uses OUT=\"\${1:-}\" which swallows flags as the output path"
    failed=1
  fi

  # Invariant 2: --check must be intercepted as a real flag, not reachable as
  # a positional OUT path.
  if ! grep -nE '^[[:space:]]*--check\)' "$script" >/dev/null; then
    echo "regression: $script has no dedicated --check) flag case"
    failed=1
  fi

  # Invariant 3: unknown leading-dash tokens must be rejected.
  if ! grep -nE '^[[:space:]]*-\*\)' "$script" >/dev/null; then
    echo "regression: $script has no -*) unknown-option rejection"
    failed=1
  fi

  # Dynamic check (fast, no network/full scan): an unknown flag must exit
  # non-zero during arg parsing. Proves the -*) branch is live.
  if bash "$script" --bogus-outpath-invariant >/dev/null 2>&1; then
    echo "regression: $script accepted an unknown --bogus flag"
    failed=1
  fi

  # Dynamic check (fast): --help must succeed without leaving a ./--check
  # residue behind.
  bash "$script" --help >/dev/null 2>&1 || {
    echo "regression: $script --help did not exit 0"
    failed=1
  }
}

for script in "${OUTPATH_SCRIPTS[@]}"; do
  check_script "$script"
done

if [[ -e "$ROOT_DIR/--check" ]]; then
  echo "regression: a checked script left a ./--check residue at the repo root"
  failed=1
fi

if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

echo "output-path invariant check passed (${#OUTPATH_SCRIPTS[@]} scripts)"
