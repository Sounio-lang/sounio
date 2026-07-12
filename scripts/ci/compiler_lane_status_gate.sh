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
# This gate's own HEAD changes coordination files only. It must not be
# misreported as a compiler review lane.
if grep -q '^state=' <<< "$current_output"; then
  echo 'compiler-lanes: non-compiler current worktree leaked into lane output' >&2
  exit 1
fi
grep -q '^scanner_mode=all-worktrees$' <<< "$global_output"
grep -q '^== Summary ==$' <<< "$global_output"

echo '[compiler-lanes] PASS: read-only lane classification is executable'
