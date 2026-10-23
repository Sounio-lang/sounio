#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC="${SOUC:-bin/souc}"
AUTHORITY="tools/pireus/xor_basis4_semantic_authority.sio"
FROZEN="tools/pireus/xor_basis4_semantics.values.v1"
EXPECTED_SOURCE_HASH="064f4613311e810a465530e282153129a2760e04d199ad3676e865b7b54693c8"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

fail() {
    printf 'PIREUS_XOR_HLIR_CONTRACT_FAIL: %s\n' "$*" >&2
    exit 1
}

actual_hash="$(sha256sum "$AUTHORITY" | cut -d' ' -f1)"
[ "$actual_hash" = "$EXPECTED_SOURCE_HASH" ] ||
    fail "Sounio authority source changed without a new freeze"

"$SOUC" run "$AUTHORITY" >"$TMP_DIR/authority.log" 2>&1 ||
    fail "Sounio semantic authority did not execute"
sed -n '/^schema=pireus-xor-basis4-semantics-v1$/,$p' \
    "$TMP_DIR/authority.log" >"$TMP_DIR/authority.values"
cmp -s "$FROZEN" "$TMP_DIR/authority.values" ||
    fail "frozen semantics diverge from Sounio execution"

rg -q 'pub enum HlirXorTwist' self-hosted/hlir/ir.sio ||
    fail "typed twist contract missing"
rg -q 'pub operator_kind: i64' self-hosted/hlir/ir.sio ||
    fail "HLIR operator identity missing"
rg -q 'instr.operator_kind == HLIR_OPERATOR_XOR_CONVOLUTION' \
    self-hosted/gpu/hlir_to_gpu.sio ||
    fail "GPU lowering does not consume semantic identity"

if rg -q 'hlir_name_is_pireus_sed_xor|HlirXorCocycle|HlirCocycleCayleyDickson' \
    self-hosted/hlir self-hosted/gpu; then
    fail "legacy magic-name or false cocycle contract remains"
fi

printf 'PIREUS_XOR_HLIR_CONTRACT_PASS\n'
