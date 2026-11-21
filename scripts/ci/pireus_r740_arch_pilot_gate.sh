#!/usr/bin/env bash
# R740 admitted-lowering pilot: PTX targets, per-architecture gain rule, launcher.
# Does not submit Slurm and does not weaken the Spark admission gate.
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9
. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "pireus_r740_arch_pilot"

require_file "tools/pireus/continuity/r740/test_r740_pilot.py"
require_file "tools/pireus/continuity/r740/arch_gain_decision.sio"
require_file "tools/pireus/continuity/r740/arch_gain_rule.lock.json"
require_tool python3

LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT
if ! python3 tools/pireus/continuity/r740/test_r740_pilot.py > "$LOG" 2>&1; then
  tail -40 "$LOG" >&2
  gate_fail "r740 pilot controls failed"
fi
gate_pass "r740 pilot controls passed; no job was submitted"
