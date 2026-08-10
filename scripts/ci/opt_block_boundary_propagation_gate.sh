#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="$ROOT_DIR/tests/run-pass/opt_block_boundary_propagation.sio"
SOUC="${SOUNIO_OPT_BLOCK_BOUNDARY_SOUC:-$ROOT_DIR/bin/madaros}"
WORK_DIR="${SOUNIO_OPT_BLOCK_BOUNDARY_GATE_DIR:-$(mktemp -d /tmp/sounio-opt-block-boundary.XXXXXX)}"
STACK_KB="${SOUNIO_OPT_BLOCK_BOUNDARY_STACK_KB:-524288}"

fail() {
    echo "opt_block_boundary_propagation_gate: FAIL: $*" >&2
    exit 1
}

[[ -f "$SOURCE" ]] || fail "missing witness: $SOURCE"
[[ -x "$SOUC" ]] || fail "compiler is not executable: $SOUC"
[[ "$STACK_KB" =~ ^[1-9][0-9]*$ && ${#STACK_KB} -le 9 ]] || fail "invalid stack target: $STACK_KB"

stack_soft_before="$(ulimit -S -s)"
stack_hard="$(ulimit -H -s)"
if [[ "$stack_soft_before" != "unlimited" ]] && ((stack_soft_before < STACK_KB)); then
    if [[ "$stack_hard" != "unlimited" ]] && ((stack_hard < STACK_KB)); then
        fail "stack hard limit is too low: requested=$STACK_KB hard=$stack_hard"
    fi
    ulimit -S -s "$STACK_KB" || fail "could not raise stack to $STACK_KB KiB"
fi
stack_soft_after="$(ulimit -S -s)"
if [[ "$stack_soft_after" != "unlimited" ]] && ((stack_soft_after < STACK_KB)); then
    fail "stack raise was ineffective: requested=$STACK_KB actual=$stack_soft_after"
fi
echo "OPT_BLOCK_BOUNDARY_STACK status=ready requested_kb=$STACK_KB soft_before_kb=$stack_soft_before soft_after_kb=$stack_soft_after hard_kb=$stack_hard"

mkdir -p "$WORK_DIR"

run_case() {
    local label="$1"
    shift
    local elf="$WORK_DIR/$label.elf"
    local compile_log="$WORK_DIR/$label.compile.log"
    local run_log="$WORK_DIR/$label.run.log"

    if ! "$SOUC" "$SOURCE" "$@" -o "$elf" >"$compile_log" 2>&1; then
        fail "$label compile failed; see $compile_log"
    fi
    [[ -s "$elf" ]] || fail "$label compile produced no ELF"
    chmod +x "$elf"

    set +e
    timeout 15 "$elf" >"$run_log" 2>&1
    local rc=$?
    set -e
    [[ $rc -eq 0 ]] || fail "$label runtime rc=$rc; see $run_log"
    echo "PASS  $label runtime rc=0"
}

run_case no-opt
run_case opt -O

echo "OPT_BLOCK_BOUNDARY_PROPAGATION_PASS no_opt_rc=0 opt_rc=0"
