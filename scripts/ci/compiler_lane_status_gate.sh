#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCANNER="$ROOT_DIR/scripts/dev/compiler_lane_status.sh"

bash -n "$SCANNER"
# actions/checkout checks out a synthetic merge commit without creating
# origin/main. HEAD is the correct integration snapshot for this isolated gate.
current_output="$(bash "$SCANNER" --main-ref HEAD --current-only)"
global_output="$(bash "$SCANNER" --main-ref HEAD)"

grep -q '^Sounio compiler lane status$' <<< "$current_output"
grep -Eq '^main_ref=HEAD main_sha=[0-9a-f]{10}$' <<< "$current_output"
grep -q '^scanner_mode=current-only$' <<< "$current_output"
# The checked-out HEAD may itself include compiler files. That is still the
# integrated product in CI because this gate pins --main-ref HEAD. What must not
# leak into this read-only gate is an active/stale/review compiler lane.
if grep -Eq '^state=(ACTIVE|STALE_WITH_RESIDUE|SCRATCH_COPY|REVIEW_READY|FRONTIER|FRONTIER_INTEGRATED|UNCLASSIFIED)' <<< "$current_output"; then
  echo 'compiler-lanes: current worktree was reported as an outstanding compiler lane' >&2
  exit 1
fi
grep -q '^scanner_mode=all-worktrees$' <<< "$global_output"
grep -q '^== Summary ==$' <<< "$global_output"

echo '[compiler-lanes] PASS: read-only lane classification is executable'
