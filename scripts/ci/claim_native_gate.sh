#!/usr/bin/env bash
# scripts/ci/claim_native_gate.sh
#
# CI gate for native parser support of AST-native claims
# (ItemKind::ItemClaim — future rung 1 of
# docs/research/ast_native_claims_spec_2026-07-25.md).
#
# Clauses:
#   N1_PARSER_SURFACE   — ItemKind::ItemClaim, ClaimDecl, and parse_claim_item
#                         exist in the self-hosted parser sources.
#   N2_STREAM_FILTER    — the parse driver (parser/mod.sio) drops ItemClaim
#                         items and resets the claim registry.
#   N3_OLD_COMPILER_REJECTS — the checked-in (pre-change) compiler rejects the
#                         claim syntax, proving the test exercises the new
#                         parser path. SKIPPED when bin/madaros-linux-x86_64
#                         has been refreshed to a post-claim build.
#   N4_NEW_COMPILER_RUNS — a madaros rebuilt from CURRENT source compiles and
#                         runs tests/run-pass/claim_native_basic.sio, printing
#                         CLAIM_NATIVE_OK (claims ignored at codegen).
#                         Requires a claim-aware raw ELF: set MADAROS_RAW_BIN,
#                         or let this gate build one via
#                         scripts/ci/build_modular_madaros.sh (CPU-heavy).
#                         Set CLAIM_NATIVE_SKIP_BUILD=1 to reuse an existing
#                         artifacts/self-hosted/madaros-claim-native.
#
# Exit 0 = PASS (prints CLAIM_NATIVE_GATE_OK), 1 = FAIL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

TEST_SIO="$REPO_ROOT/tests/run-pass/claim_native_basic.sio"
CLAIM_ELF="$REPO_ROOT/artifacts/self-hosted/madaros-claim-native"

fail() {
    echo "CLAIM_NATIVE_GATE_FAIL: $*" >&2
    exit 1
}

# N1: parser surface exists in source.
grep -q "ItemClaim" self-hosted/parser/ast.sio \
    || fail "N1: ItemKind::ItemClaim missing from parser/ast.sio"
grep -q "pub struct ClaimDecl" self-hosted/parser/ast.sio \
    || fail "N1: ClaimDecl missing from parser/ast.sio"
grep -q "fn parse_claim_item" self-hosted/parser/items.sio \
    || fail "N1: parse_claim_item missing from parser/items.sio"
echo "N1_PARSER_SURFACE PASS"

# N2: parse driver filters claims out of the item stream.
grep -q "ItemKind::ItemClaim" self-hosted/parser/mod.sio \
    || fail "N2: parser/mod.sio does not filter ItemClaim items"
grep -q "ast_reset_claims" self-hosted/parser/mod.sio \
    || fail "N2: parser/mod.sio does not reset the claim registry"
echo "N2_STREAM_FILTER PASS"

# N3: the checked-in prebuilt compiler predates the syntax and must reject it.
# Once the fleet refreshes bin/madaros-linux-x86_64 to a post-claim build this
# clause no longer applies and is skipped.
if MADAROS_RAW_BIN="$REPO_ROOT/bin/madaros-linux-x86_64" \
    bin/madaros check "$TEST_SIO" > /dev/null 2>&1; then
    echo "N3_OLD_COMPILER_REJECTS SKIP (checked-in compiler already claim-aware)"
else
    echo "N3_OLD_COMPILER_REJECTS PASS"
fi

# N4: a claim-aware compiler (rebuilt from current source) runs the test.
if [[ -z "${MADAROS_RAW_BIN:-}" ]]; then
    if [[ "${CLAIM_NATIVE_SKIP_BUILD:-0}" == "1" ]]; then
        [[ -x "$CLAIM_ELF" ]] || fail "N4: CLAIM_NATIVE_SKIP_BUILD=1 but $CLAIM_ELF missing"
    else
        bash scripts/ci/build_modular_madaros.sh "$CLAIM_ELF" > /tmp/claim_native_build.log 2>&1 \
            || fail "N4: madaros rebuild failed (see /tmp/claim_native_build.log)"
    fi
    export MADAROS_RAW_BIN="$CLAIM_ELF"
fi

RUN_OUT="$(bin/madaros run "$TEST_SIO" 2>&1)" || fail "N4: claim-native compiler failed on $TEST_SIO"
printf '%s\n' "$RUN_OUT" | grep -q "CLAIM_NATIVE_OK" \
    || fail "N4: expected stdout CLAIM_NATIVE_OK, got: $RUN_OUT"
echo "N4_NEW_COMPILER_RUNS PASS"

echo "CLAIM_NATIVE_GATE_OK"
