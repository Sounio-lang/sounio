#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCANNER="$ROOT_DIR/scripts/dev/compiler_lane_status.sh"

bash -n "$SCANNER"
current_output="$(bash "$SCANNER" --current-only)"
global_output="$(bash "$SCANNER")"

grep -q '^Sounio compiler lane status$' <<< "$current_output"
grep -q '^scanner_mode=current-only$' <<< "$current_output"
grep -q '^scanner_mode=all-worktrees$' <<< "$global_output"
grep -q '^== Summary ==$' <<< "$global_output"

echo '[compiler-lanes] PASS: read-only lane classification is executable'
