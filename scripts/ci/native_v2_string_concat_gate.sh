#!/usr/bin/env bash
# native_v2_string_concat_gate.sh
#
# Regression gate for native-v2 string concatenation:
#   str_len(str_concat("ab","cde")) -> exit 5
#   str_char_at(str_concat("ab","cde"), 3) -> exit 100 ('d')
#
# Uses lean_single to build the slim suite emitter (main.sio is out of scope).
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ulimit -s 1048576 2>/dev/null || true

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

BOOTSTRAP_SOUC="${SOUNIO_BOOTSTRAP_SOUC:-./bin/souc}"
EMIT_SRC="${SOUNIO_NATIVE_V2_SUITE_EMIT_SRC:-self-hosted/compiler/native_v2_codegen_suite_emit.sio}"
MODULAR_SOUC="${SOUNIO_MODULAR_SOUC:-}"

if [[ -z "$MODULAR_SOUC" ]]; then
    echo "[gate] building suite emitter from $EMIT_SRC (bootstrap=$BOOTSTRAP_SOUC)"
    MODULAR_SOUC="$WORK/suite_emit.elf"
    if ! "$BOOTSTRAP_SOUC" self-hosted/compiler/lean_single.sio "$WORK/lean_single.elf" > "$WORK/lean_build.log" 2>&1; then
        echo "[gate] FAIL: could not build lean_single bootstrap"
        tail -5 "$WORK/lean_build.log"
        exit 2
    fi
    chmod +x "$WORK/lean_single.elf"
    if ! "$WORK/lean_single.elf" "$EMIT_SRC" "$MODULAR_SOUC" > "$WORK/build.log" 2>&1; then
        echo "[gate] FAIL: could not build suite emitter"
        tail -10 "$WORK/build.log"
        exit 2
    fi
    chmod +x "$MODULAR_SOUC"
fi

if [[ ! -x "$MODULAR_SOUC" ]]; then
    echo "[gate] FAIL: suite emitter not executable: $MODULAR_SOUC"
    exit 2
fi

FAILED=0

emit_and_check() {
    local expected="$1"; shift
    local label="$1"; shift
    local mode="$1"; shift
    local elf="$WORK/witness_${label}_$$.elf"
    rm -f "$elf"

    "$MODULAR_SOUC" "$mode" "$elf" > "$WORK/emit_${label}.log" 2>&1 || true

    if [[ ! -f "$elf" ]]; then
        echo "[gate] FAIL[$label]: compiler did not emit $elf"
        grep -a 'native_v2_emit_exit_code' "$WORK/emit_${label}.log" | tr -cd '[:print:]\n' || true
        FAILED=1
        return
    fi

    local magic
    magic="$(head -c4 "$elf" | od -An -tx1 | tr -d ' ')"
    if [[ "$magic" != "7f454c46" ]]; then
        echo "[gate] FAIL[$label]: emitted file is not ELF (magic=$magic)"
        FAILED=1
        return
    fi

    chmod +x "$elf"
    "$elf"
    local got=$?
    if [[ "$got" -ne "$expected" ]]; then
        echo "[gate] FAIL[$label]: exit code=$got, expected $expected"
        FAILED=1
        return
    fi
    echo "[gate] PASS[$label]: exit $got (expected $expected)"
}

echo "[gate] native-v2 string concat (IR -> ELF -> exit)"

emit_and_check 5   "str_concat_len"  --native-v2-emit-string-concat-len
emit_and_check 100 "str_concat_char" --native-v2-emit-string-concat-char

if [[ "$FAILED" -ne 0 ]]; then
    echo "[gate] FAIL: string concat witnesses regressed"
    exit 5
fi

echo "[gate] PASS: native-v2 string concat len + byte-index witnesses verified"
exit 0