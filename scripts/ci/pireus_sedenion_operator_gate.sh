#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC="${SOUC:-bin/souc}"
SOURCE="tests/native/pireus_sedenion_operator.sio"
PROBE="tools/pireus/sedenion_operator_ir_contract.sio"
work="$(mktemp -d "${TMPDIR:-/tmp}/pireus-sedenion-operator.XXXXXX")"
trap 'rm -rf "$work"' EXIT

fail() {
    printf 'PIREUS_SEDENION_OPERATOR_GATE_FAIL: %s\n' "$*" >&2
    exit 1
}

"$SOUC" check "$SOURCE" >"$work/source-check.log" 2>&1 ||
    fail "ordinary Sedenion multiplication source does not typecheck"

rg -q '\(\*pa\) \* \(\*pb\)' "$SOURCE" || fail "source witness does not use ordinary multiplication"
if rg -q 'pireus_sed_xor_convolution|xor_convolution\(' "$SOURCE"; then
    fail "source witness uses a named lowering escape hatch"
fi

SOUNIO_SOUC_ENGINE=lean_single "$SOUC" run "$PROBE" >"$work/probe.log" 2>&1 ||
    fail "Sounio IR contract probe did not execute"
grep -q '^PIREUS_SEDENION_OPERATOR_IR_PASS$' "$work/probe.log" ||
    fail "Sounio IR contract probe did not pass"

rg -q 'info\.op_kind == 1 && info\.algebra_tag == 4' self-hosted/ir/lower.sio ||
    fail "AST-to-IR lowering does not select Sedenion multiplication metadata"
rg -q 'ir_xor_convolution_s_virtual_ptr3\(dst_pair\.1, lhs_pair\.1, rhs_pair\.1\)' self-hosted/ir/lower.sio ||
    fail "AST-to-IR lowering does not preserve the semantic operator"
rg -q 'abi=ptr3' self-hosted/ir/lower.sio ||
    fail "typed assignment does not preserve the three-pointer ABI"
rg -q 'pireus_native_emit_xor_convolution_s_ptr3' self-hosted/native/codegen_x86_linux.sio ||
    fail "native-v2 does not materialize the semantic operator"
rg -q 'IR_A_LABEL_ID.*!= 1' self-hosted/native/codegen_x86_linux.sio ||
    fail "native-v2 does not reject a generic virtual operator without ptr3 ABI"
rg -q 'return hlir_oct_record_deferred_xor\(s, lhs_ptr, rhs_ptr\)' self-hosted/hlir/lower.sio ||
    fail "HLIR still lacks deferred Sedenion operator preservation"

printf 'PIREUS_SEDENION_OPERATOR_GATE_PASS source=%s probe=%s\n' "$SOURCE" "$PROBE"
