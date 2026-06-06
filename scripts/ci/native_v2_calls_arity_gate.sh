#!/usr/bin/env bash
# native_v2_calls_arity_gate.sh
#
# Verifies 5- and 6-argument calls in the modular native-v2 x86-64 backend.
# Exercises SysV arg regs 4 (r8) and 5 (r9), which require a REX.R prefix
# in the load encoding — distinct from rdi/rsi/rdx/rcx (regs 0-3).
#
# Witnesses:
#   call5 -> add5(1,2,4,8,16)=31    5 args: rdi/rsi/rdx/rcx/r8
#   call6 -> add6(1,2,4,8,16,32)=63 6 args: rdi/rsi/rdx/rcx/r8/r9
#
# Exit codes are power-of-two sums: a dropped or misencoded arg yields a
# distinct wrong value (e.g. missing r8 arg -> 15 for call5, not 31).
#
# Builds the modular compiler from self-hosted/compiler/main.sio using the
# bootstrap ./bin/souc, then emits witnesses via --native-v2-emit-call5/6.
# This is the ONLY sanctioned build path — no native_compile_driver.sio.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ulimit -s 1048576 2>/dev/null || true

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

BOOTSTRAP_SOUC="${SOUNIO_BOOTSTRAP_SOUC:-./bin/souc}"
MODULAR_SOUC="${SOUNIO_MODULAR_SOUC:-}"

if [[ -z "$MODULAR_SOUC" ]]; then
    echo "[native-v2-calls-arity] building modular compiler from self-hosted/compiler/main.sio (bootstrap=$BOOTSTRAP_SOUC)"
    MODULAR_SOUC="$WORK/mc.elf"
    if ! "$BOOTSTRAP_SOUC" self-hosted/compiler/main.sio "$MODULAR_SOUC" > "$WORK/build.log" 2>&1; then
        echo "[native-v2-calls-arity] FAIL: could not build modular compiler" >&2
        tail -5 "$WORK/build.log" >&2
        exit 2
    fi
    chmod +x "$MODULAR_SOUC"
fi

if [[ ! -x "$MODULAR_SOUC" ]]; then
    echo "[native-v2-calls-arity] FAIL: modular compiler not executable: $MODULAR_SOUC" >&2
    exit 2
fi

FAILED=0

emit_and_check() {
    local expected="$1"; shift
    local label="$1"; shift
    local elf="$WORK/witness_${label}_$$.elf"
    rm -f "$elf"

    "$MODULAR_SOUC" "$@" "$elf" > "$WORK/emit_${label}.log" 2>&1 || true

    if [[ ! -f "$elf" ]]; then
        echo "[native-v2-calls-arity] FAIL[$label]: compiler did not emit $elf" >&2
        tail -5 "$WORK/emit_${label}.log" >&2 || true
        FAILED=1
        return
    fi

    local magic
    magic="$(head -c4 "$elf" | od -An -tx1 | tr -d ' ')"
    if [[ "$magic" != "7f454c46" ]]; then
        echo "[native-v2-calls-arity] FAIL[$label]: emitted file is not ELF (magic=$magic)" >&2
        FAILED=1
        return
    fi

    chmod +x "$elf"
    "$elf"
    local got=$?
    if [[ "$got" -ne "$expected" ]]; then
        echo "[native-v2-calls-arity] FAIL[$label]: exit=$got expected=$expected" >&2
        FAILED=1
        return
    fi
    echo "[native-v2-calls-arity] PASS[$label]: exit $got (expected $expected)"
}

echo "[native-v2-calls-arity] 5- and 6-arg call witness gate (r8/r9 SysV regs)"

emit_and_check 31 "call5" --native-v2-emit-call5
emit_and_check 63 "call6" --native-v2-emit-call6

if [[ "$FAILED" -ne 0 ]]; then
    echo "[native-v2-calls-arity] FAIL: 5/6-arg call witness regressed"
    exit 5
fi

echo "[native-v2-calls-arity] PASS: 5-arg (r8) and 6-arg (r9) calls emit correct exit codes"
exit 0
