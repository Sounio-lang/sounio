#!/usr/bin/env bash
# Gate for ir::effects (ADR-013): the single authority for IR memory effects.
#
# Scope: this gate protects exactly what feat/ir-effects-authority changed --
# the new self-hosted/ir/effects.sio module and its one live caller,
# ocp_sink_loads in opt_cleanup.sio. It does NOT enforce full migration of the
# other four legacy side-effect tables (optimize.sio, const_prop.sio, dce.sio,
# auto_vectorize.sio) or the memory_analysis.sio MemPtrOp table found during
# review -- ADR-013 lists retiring those as later, separate migration steps
# (2-4), not part of this change. A gate that failed on their continued
# existence would be checking a claim this PR never made.
#
# Checks, in the style of scripts/ci/exact_bitwise_rebracket_authority_gate.sh
# (text-anchor statics + a real compiler run), narrowed to this PR's scope:
#
#   1. `check`: the compiler accepts effects.sio and the self-test runner on
#      their own (catches a type error before wasting a run).
#   2. static anchor: opt_cleanup.sio still imports ir::effects and
#      ocp_sink_loads still delegates to ir_effect_swap_ok, i.e. the guard
#      was not quietly reverted to its old inline src1/src2-only form.
#   3. `run`: self-hosted/ir/effects_self_test_runner.sio (T01-T09) against a
#      real compiler, require exit 0, the literal "[effects] ALL PASS" line,
#      and reject a "defect reproduced" T09 witness -- that string is what
#      the runner prints if ocp_sink_loads regresses to sinking a load past
#      an IrIndexSet that reads it through imm_i64 (the #1682-class bug this
#      whole change exists to close).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MADAROS="${MADAROS_BIN:-$ROOT_DIR/bin/madaros}"
RUNNER="$ROOT_DIR/self-hosted/ir/effects_self_test_runner.sio"
EFFECTS="$ROOT_DIR/self-hosted/ir/effects.sio"
OPT_CLEANUP="$ROOT_DIR/self-hosted/ir/opt_cleanup.sio"
WORK="${SOUNIO_IR_EFFECTS_GATE_DIR:-$(mktemp -d /tmp/sounio-ir-effects-authority.XXXXXX)}"
KEEP_WORK="${SOUNIO_IR_EFFECTS_GATE_KEEP:-0}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi
mkdir -p "$WORK"

fail() { echo "[ir-effects-authority] FAIL: $*" >&2; exit 1; }

[[ -f "$EFFECTS" ]] || fail "self-hosted/ir/effects.sio is missing"
[[ -f "$RUNNER" ]] || fail "self-hosted/ir/effects_self_test_runner.sio is missing"
[[ -x "$MADAROS" ]] || fail "compiler is missing or not executable: $MADAROS (build it first, e.g. \`make build-madaros\`)"

# --- 1. check: effects.sio and the runner must typecheck on their own. ------
check_log="$WORK/check.log"
set +e
"$MADAROS" check "$EFFECTS" >"$check_log" 2>&1
effects_check_rc=$?
"$MADAROS" check "$RUNNER" >>"$check_log" 2>&1
runner_check_rc=$?
set -e
if [[ "$effects_check_rc" -ne 0 || "$runner_check_rc" -ne 0 ]]; then
  cat "$check_log" >&2
  fail "compiler check failed (effects.sio rc=$effects_check_rc, runner rc=$runner_check_rc)"
fi

# --- 2. static anchor: ocp_sink_loads still delegates to ir::effects. -------
grep -q 'use ir::effects::\*' "$OPT_CLEANUP" ||
  fail "self-hosted/ir/opt_cleanup.sio no longer imports ir::effects"
grep -q 'ir_effect_swap_ok' "$OPT_CLEANUP" ||
  fail "ocp_sink_loads no longer delegates to ir_effect_swap_ok"

# --- 3. run: the self-test runner against a real compiler; require ALL PASS.
run_log="$WORK/effects_self_test_runner.log"
set +e
"$MADAROS" run "$RUNNER" >"$run_log" 2>&1
rc=$?
set -e
cat "$run_log"

if [[ "$rc" -ne 0 ]]; then
  fail "effects_self_test_runner.sio exited rc=$rc"
fi
grep -q '^\[effects\] ALL PASS$' "$run_log" || {
  fail "effects_self_test_runner.sio did not print '[effects] ALL PASS'"
}
if grep -q 'defect reproduced' "$run_log"; then
  fail "T09 witness reports 'defect reproduced' -- ocp_sink_loads regressed to the pre-ir::effects defect"
fi
grep -q 'defect not reproduced' "$run_log" || {
  fail "T09 witness line is missing entirely (expected 'defect not reproduced')"
}

echo "[ir-effects-authority] PASS check=ok opt_cleanup_delegates=1 runner=ALL_PASS t09=defect_not_reproduced"
