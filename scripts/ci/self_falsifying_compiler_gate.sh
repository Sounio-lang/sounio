#!/usr/bin/env bash
# scripts/ci/self_falsifying_compiler_gate.sh
#
# CI gate for the self-falsifying compiler rung
# (docs/research/self_falsifying_compiler_spec_2026-07-25.md): compile-time
# execution of scientific claims behind the opt-in --verify-claims flag.
#
# Clauses:
#   F1_SURFACE        — claim executor module, --verify-claims flag/hooks, and
#                       the text-preserving registry accessors exist in source.
#   F2_INERT_DEFAULT  — the run-pass test compiles and runs WITHOUT the flag;
#                       claims are inert metadata.
#   F3_NO_CLAIMS_NOOP — flag on a claim-free source: VERIFY_CLAIMS_NOOP, exit 0.
#   F4_PASS_ONLY      — flag on a source whose gates all pass: VERIFY_CLAIMS_OK,
#                       exit 0, codegen+run proceed.
#   F5_FAIL_BLOCKS    — flag on the mixed test source: CLAIM_PASS for the
#                       passing claim, CLAIM_SKIP for the archived claim,
#                       CLAIM_FAIL for the failing claim,
#                       VERIFY_CLAIMS_FALSIFIED, non-zero exit, no codegen.
#   F6_TIMEOUT        — optional (SFC_TEST_TIMEOUT=1): a sleeping gate is
#                       killed and reported CLAIM_TIMEOUT.
#
# F2-F6 require a claim-aware Madaros built from CURRENT source. Set
# MADAROS_RAW_BIN to provide one, or let this gate build
# artifacts/self-hosted/madaros-self-falsifying via
# scripts/ci/build_modular_madaros.sh (CPU-heavy). Set SFC_SKIP_BUILD=1 to
# reuse an existing artifact.
#
# Exit 0 = PASS (prints SELF_FALSIFYING_COMPILER_GATE_OK), 1 = FAIL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

TEST_SIO="$REPO_ROOT/tests/run-pass/self_falsifying_compiler_test.sio"
FIX="$REPO_ROOT/scripts/ci/fixtures"
CLAIM_ELF="$REPO_ROOT/artifacts/self-hosted/madaros-self-falsifying"

fail() {
    echo "SELF_FALSIFYING_COMPILER_GATE_FAIL: $*" >&2
    exit 1
}

# F1: source surface.
grep -q "module claim_executor" self-hosted/compiler/claim_executor.sio \
    || fail "F1: claim_executor module missing"
grep -q "claim_executor_verify" self-hosted/compiler/main.sio \
    || fail "F1: claim_executor_verify not wired into compiler/main.sio"
grep -q -- "--verify-claims" self-hosted/compiler/main.sio \
    || fail "F1: --verify-claims flag missing from compiler/main.sio"
grep -q "ast_claim_slot_field" self-hosted/parser/ast.sio \
    || fail "F1: text-preserving claim accessors missing from parser/ast.sio"
[[ -f docs/research/self_falsifying_compiler_spec_2026-07-25.md ]] \
    || fail "F1: spec doc missing"
echo "F1_SURFACE PASS"

# Build/select the claim-aware compiler.
if [[ -z "${MADAROS_RAW_BIN:-}" ]]; then
    if [[ "${SFC_SKIP_BUILD:-0}" == "1" ]]; then
        [[ -x "$CLAIM_ELF" ]] || fail "build: SFC_SKIP_BUILD=1 but $CLAIM_ELF missing"
    else
        bash scripts/ci/build_modular_madaros.sh "$CLAIM_ELF" > /tmp/sfc_madaros_build.log 2>&1 \
            || fail "build: madaros rebuild failed (see /tmp/sfc_madaros_build.log)"
    fi
    RAW="$CLAIM_ELF"
else
    RAW="$MADAROS_RAW_BIN"
fi
echo "build: using raw compiler $RAW"

ulimit -s unlimited 2>/dev/null || true
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$REPO_ROOT/stdlib}"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sfc_gate.XXXXXX")"
cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

# run_case <name> <expect_rc:zero|nonzero> <args...>; sets OUT and RC.
run_case() {
    local name="$1"; shift
    local expect="$1"; shift
    set +e
    OUT="$("$RAW" "$@" 2>&1)"
    RC=$?
    set -e
    if [[ "$expect" == "zero" && $RC -ne 0 ]]; then
        echo "$OUT" >&2
        fail "$name: expected exit 0, got $RC"
    fi
    if [[ "$expect" == "nonzero" && $RC -eq 0 ]]; then
        echo "$OUT" >&2
        fail "$name: expected non-zero exit, got 0"
    fi
}

# F2: without the flag claims are inert — compile and run normally.
run_case F2 zero run "$TEST_SIO" -o "$TMP_DIR/f2.elf"
grep -q "SELF_FALSIFYING_COMPILER_TEST_OK" <<<"$OUT" \
    || fail "F2: expected run marker without --verify-claims, got: $OUT"
grep -q "CLAIM_FAIL" <<<"$OUT" \
    && fail "F2: claim gates ran without --verify-claims: $OUT"
echo "F2_INERT_DEFAULT PASS"

# F3: no claims → no-op.
run_case F3 zero run "$FIX/self_falsifying_no_claims.sio" -o "$TMP_DIR/f3.elf" --verify-claims
grep -q "VERIFY_CLAIMS_NOOP" <<<"$OUT" \
    || fail "F3: expected VERIFY_CLAIMS_NOOP, got: $OUT"
grep -q "SELF_FALSIFYING_NO_CLAIMS_OK" <<<"$OUT" \
    || fail "F3: expected compile+run to proceed, got: $OUT"
echo "F3_NO_CLAIMS_NOOP PASS"

# F4: all gates pass → verification succeeds and codegen+run proceed.
run_case F4 zero run "$FIX/self_falsifying_claims_pass_only.sio" -o "$TMP_DIR/f4.elf" --verify-claims
grep -q "CLAIM_PASS sfc_only_pass" <<<"$OUT" \
    || fail "F4: expected CLAIM_PASS sfc_only_pass, got: $OUT"
grep -q "CLAIM_SKIP sfc_metadata_only (no-gate)" <<<"$OUT" \
    || fail "F4: expected CLAIM_SKIP sfc_metadata_only (no-gate), got: $OUT"
grep -q "VERIFY_CLAIMS_OK" <<<"$OUT" \
    || fail "F4: expected VERIFY_CLAIMS_OK, got: $OUT"
grep -q "SELF_FALSIFYING_PASS_ONLY_OK" <<<"$OUT" \
    || fail "F4: expected codegen+run to proceed, got: $OUT"
echo "F4_PASS_ONLY PASS"

# F5: a failing gate falsifies its claim and blocks codegen; the archived
# claim's (failing) gate must be skipped.
run_case F5 nonzero run "$TEST_SIO" -o "$TMP_DIR/f5.elf" --verify-claims
grep -q "CLAIM_PASS sfc_gate_pass" <<<"$OUT" \
    || fail "F5: expected CLAIM_PASS sfc_gate_pass, got: $OUT"
grep -q "CLAIM_FAIL sfc_gate_fail" <<<"$OUT" \
    || fail "F5: expected CLAIM_FAIL sfc_gate_fail, got: $OUT"
grep -q "CLAIM_SKIP sfc_archived (archived)" <<<"$OUT" \
    || fail "F5: expected CLAIM_SKIP sfc_archived (archived), got: $OUT"
grep -q "VERIFY_CLAIMS_FALSIFIED" <<<"$OUT" \
    || fail "F5: expected VERIFY_CLAIMS_FALSIFIED, got: $OUT"
grep -q "SELF_FALSIFYING_COMPILER_TEST_OK" <<<"$OUT" \
    && fail "F5: codegen+run proceeded despite falsified claim: $OUT"
[[ ! -f "$TMP_DIR/f5.elf" ]] \
    || fail "F5: ELF emitted despite falsified claim"
echo "F5_FAIL_BLOCKS PASS"

# F7: the default lane (no mode keyword) must also block codegen on a
# falsified claim and exit non-zero.
rm -f "$TMP_DIR/f7.elf"
run_case F7 nonzero "$TEST_SIO" -o "$TMP_DIR/f7.elf" --verify-claims
grep -q "VERIFY_CLAIMS_FALSIFIED" <<<"$OUT" \
    || fail "F7: expected VERIFY_CLAIMS_FALSIFIED, got: $OUT"
grep -q "Compilation successful" <<<"$OUT" \
    && fail "F7: default lane compiled despite falsified claim: $OUT"
[[ ! -f "$TMP_DIR/f7.elf" ]] \
    || fail "F7: default lane emitted ELF despite falsified claim"
echo "F7_DEFAULT_LANE_BLOCKS PASS"

# F6 (optional): timeout enforcement — costs ~30s wall-clock.
if [[ "${SFC_TEST_TIMEOUT:-0}" == "1" ]]; then
    run_case F6 nonzero run "$FIX/self_falsifying_claims_timeout.sio" -o "$TMP_DIR/f6.elf" --verify-claims
    grep -q "CLAIM_TIMEOUT sfc_gate_timeout" <<<"$OUT" \
        || fail "F6: expected CLAIM_TIMEOUT sfc_gate_timeout, got: $OUT"
    grep -q "VERIFY_CLAIMS_FALSIFIED" <<<"$OUT" \
        || fail "F6: expected VERIFY_CLAIMS_FALSIFIED, got: $OUT"
    echo "F6_TIMEOUT PASS"
else
    echo "F6_TIMEOUT SKIP (set SFC_TEST_TIMEOUT=1 to enable)"
fi

echo "SELF_FALSIFYING_COMPILER_GATE_OK"
