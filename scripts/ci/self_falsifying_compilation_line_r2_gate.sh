#!/usr/bin/env bash
# scripts/ci/self_falsifying_compilation_line_r2_gate.sh
#
# CI gate for rung R2 of the self-falsifying compilation line
# (docs/research/self_falsifying_compilation_line_r2_2026-07-26.md):
# verdict-token binding in the compiler.
#
# ORDER MATTERS. The contract's T5_BEHAVIOUR_RECEIPT refuses to certify
# implementation without evidence that the mechanism actually ran, so the
# compile arm runs FIRST and writes the receipt, and the contract is evaluated
# afterwards. Without the compile arm, T5 fails and says how to fix it — which
# is the honest state: source surface is not behaviour.
#
# Compile arm (SFCL_R2_RUN_COMPILE=1) — needs a token-binding Madaros
# (MADAROS_RAW_BIN, or artifacts/self-hosted/madaros-token-binding):
#   D1_MATCH_PASSES      declared token == emitted token -> CLAIM_PASS, ELF
#   D2_DRIFT_BLOCKS      gate exits 0 but emits another token ->
#                        CLAIM_TOKEN_MISMATCH, no ELF   <- the whole point
#   D3_ABSENT_BLOCKS     gate exits 0 and emits no token ->
#                        CLAIM_TOKEN_ABSENT, no ELF
#   D4_BACKWARD_COMPAT   a claim WITHOUT verdict_token behaves exactly as
#                        before (exit-code gating only)
#
# Contract clauses:
#   T1_EXECUTOR_SURFACE  verdict_token field, capture, extraction, outcomes
#   T2_FIXTURES          probes discriminate match / drift / absent
#   T3_NO_SHELL_STRING   capture is open+dup2, argv stays fixed
#   T4_REACH             how much of the corpus could be token-bound
#   T5_BEHAVIOUR_RECEIPT D1..D4 were observed on this exact executor source
#
# Exit 0 = PASS (prints SELF_FALSIFYING_COMPILATION_LINE_R2_GATE_OK), 1 = FAIL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=../lib/classify_compile.sh
source "$REPO_ROOT/scripts/lib/classify_compile.sh"

CONTRACT="scripts/research/self_falsifying_compilation_line_r2_contract.py"
SPEC="docs/research/self_falsifying_compilation_line_r2_2026-07-26.md"
EXECUTOR="self-hosted/compiler/claim_executor.sio"
FIX="scripts/ci/fixtures"
TB_ELF="$REPO_ROOT/artifacts/self-hosted/madaros-token-binding"
RECEIPT="$REPO_ROOT/artifacts/self_falsifying_r2_receipt.txt"

fail() {
    echo "SELF_FALSIFYING_COMPILATION_LINE_R2_GATE_FAIL: $*" >&2
    exit 1
}

[[ -f "$SPEC" ]] || fail "spec doc missing: $SPEC"
[[ -f "$CONTRACT" ]] || fail "contract missing: $CONTRACT"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sfcl_r2.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

claim_refusal() {
    local rc="$1" log="$2" marker="$3"
    # The verdict marker can be printed before a crash or timeout. Only an
    # ordinary non-zero exit carrying that marker witnesses a refusal (#2425).
    sounio_expect_failed_exit "$rc" "$log" "$marker"
}

refusal_classifier_self_test() {
    local marker="CLAIM_TOKEN_MISMATCH sfc_token_drifts"
    local valid="$TMP_DIR/self-test-valid.log"
    local crash="$TMP_DIR/self-test-crash.log"

    printf '%s\n' "$marker" >"$valid"
    claim_refusal 1 "$valid" "$marker" \
        || fail "refusal_self_test_valid: $(sounio_refusal_reason)"

    cp "$valid" "$crash"
    printf '%s\n' 'Segmentation fault (core dumped)' >>"$crash"
    if claim_refusal 139 "$crash" "$marker"; then
        fail "refusal_self_test_crash_accepted"
    fi
    if claim_refusal 124 "$valid" "$marker"; then
        fail "refusal_self_test_timeout_accepted"
    fi
    if claim_refusal 1 "$crash" "$marker"; then
        fail "refusal_self_test_fatal_text_accepted"
    fi
    if claim_refusal 0 "$valid" "$marker"; then
        fail "refusal_self_test_rc0_accepted"
    fi

    echo "SELF_FALSIFYING_R2_REFUSAL_SELFTEST_OK valid=1 crash=1 fatal_text=1 timeout=1 rc0=1"
}

refusal_classifier_self_test
if [[ "${SOUNIO_SFCL_R2_SELF_TEST_ONLY:-0}" == "1" ]]; then
    exit 0
fi

# ---------------------------------------------------------------- compile arm

if [[ "${SFCL_R2_RUN_COMPILE:-0}" == "1" ]]; then
    RAW="${MADAROS_RAW_BIN:-$TB_ELF}"
    [[ -x "$RAW" ]] || fail "compile arm: no token-binding compiler at $RAW"
    echo "compile arm: using $RAW"

    ulimit -s unlimited 2>/dev/null || true
    export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$REPO_ROOT/stdlib}"

    # run_probe <name> <source> <expect_rc:zero|nonzero> <marker> <elf:yes|no>
    run_probe() {
        local name="$1" src="$2" expect_rc="$3" marker="$4" want_elf="$5"
        local elf="$TMP_DIR/${name}.elf"
        local log="$TMP_DIR/${name}.log"
        local rc=0
        rm -f "$elf"
        sounio_capture_rc rc "$log" -- "$RAW" run "$src" -o "$elf" --verify-claims
        if [[ "$expect_rc" == "zero" && $rc -ne 0 ]]; then
            tail -20 "$log" >&2; fail "$name: expected exit 0, got $rc"
        fi
        if [[ "$expect_rc" == "nonzero" ]]; then
            claim_refusal "$rc" "$log" "$marker" || {
                tail -20 "$log" >&2
                fail "$name: not a clean refusal: $(sounio_refusal_reason)"
            }
        fi
        grep -q "$marker" "$log" \
            || { tail -20 "$log" >&2; fail "$name: expected marker '$marker'"; }
        if [[ "$want_elf" == "yes" && ! -f "$elf" ]]; then
            fail "$name: expected an ELF, none emitted"
        fi
        if [[ "$want_elf" == "no" && -f "$elf" ]]; then
            fail "$name: ELF emitted despite a falsified claim"
        fi
    }

    run_probe d1 "$FIX/self_falsifying_token_pass.sio"   zero    "CLAIM_PASS sfc_token_agrees" yes
    echo "D1_MATCH_PASSES PASS — declared token == emitted token, codegen proceeds"

    run_probe d2 "$FIX/self_falsifying_token_drift.sio"  nonzero "CLAIM_TOKEN_MISMATCH sfc_token_drifts" no
    echo "D2_DRIFT_BLOCKS PASS — gate exits 0 with a different token, build refused"

    run_probe d3 "$FIX/self_falsifying_token_absent.sio" nonzero "CLAIM_TOKEN_ABSENT sfc_token_absent" no
    echo "D3_ABSENT_BLOCKS PASS — gate exits 0 emitting no token, build refused"

    run_probe d4 "$FIX/self_falsifying_claims_pass_only.sio" zero "VERIFY_CLAIMS_OK" yes
    echo "D4_BACKWARD_COMPAT PASS — claims without verdict_token unaffected"

    mkdir -p "$(dirname "$RECEIPT")"
    {
        echo "# Behaviour receipt for self-falsifying compilation R2."
        echo "# Written only after D1..D4 were observed. Bound to the executor"
        echo "# source hash, so editing the executor invalidates it."
        echo "executor_sha256=$(sha256sum "$EXECUTOR" | awk '{print $1}')"
        echo "compiler=$RAW"
        echo "D1=PASS"
        echo "D2=PASS"
        echo "D3=PASS"
        echo "D4=PASS"
    } > "$RECEIPT"
    echo "receipt written: $RECEIPT"
fi

# ---------------------------------------------------------------- contract

OUT="$(python3 "$CONTRACT" 2>&1)" || { echo "$OUT"; fail "contract exited non-zero"; }
echo "$OUT"

for clause in T1_EXECUTOR_SURFACE T2_FIXTURES T3_NO_SHELL_STRING T4_REACH T5_BEHAVIOUR_RECEIPT; do
    grep -q "^${clause} PASS" <<<"$OUT" || fail "${clause} did not PASS"
done

# Drift guard, applied to this rung's own spec — header and prose alike.
SPEC_TOKEN="$(grep -m1 -oE '^\*\*Status:\*\* `[^`]*` — `[A-Za-z0-9_]+`' "$SPEC" \
    | grep -oE '`[A-Za-z0-9_]+`$' | tr -d '`' || true)"
[[ -n "$SPEC_TOKEN" ]] || fail "spec declares no verdict token in its Status line"
CONTRACT_TOKEN="$(grep -m1 '^SELF_FALSIFYING_R2_VERDICT ' <<<"$OUT" | awk '{print $2}')"
[[ "$SPEC_TOKEN" == "$CONTRACT_TOKEN" ]] \
    || fail "verdict drift: spec says '${SPEC_TOKEN}', contract emits '${CONTRACT_TOKEN}'"
while read -r prose_token; do
    [[ -z "$prose_token" ]] && continue
    [[ "$prose_token" == "$CONTRACT_TOKEN" ]] \
        || fail "verdict drift in prose: found '${prose_token}', contract emits '${CONTRACT_TOKEN}'"
done < <(grep -oE 'SELF_FALSIFYING_R2_VERDICT [A-Za-z0-9_]+' "$SPEC" | awk '{print $2}')

echo "SELF_FALSIFYING_COMPILATION_LINE_R2_GATE_OK"
