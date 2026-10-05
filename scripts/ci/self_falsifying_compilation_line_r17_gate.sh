#!/usr/bin/env bash
# scripts/ci/self_falsifying_compilation_line_r17_gate.sh
#
# CI gate for rung R17 of the self-falsifying compilation line
# (docs/research/self_falsifying_compilation_line_r17_2026-07-28.md): witness
# binding in the compiler.
#
# ORDER MATTERS, as in R2: X3_BEHAVIOUR_RECEIPT refuses to certify the mechanism
# without evidence it ran, so the compile arm goes FIRST and writes the receipt.
# Without it the contract fails and says how to fix it, which is the honest state
# -- source surface is not behaviour.
#
# Compile arm (SFCL_R17_RUN_COMPILE=1) -- needs a witness-binding Madaros in
# MADAROS_RAW_BIN or artifacts/self-hosted/madaros-witness-binding:
#   W1 match passes · W2 drift BLOCKS (the point) · W3 absent blocks
#   W4 backward compatible · plus the R2 and R0/R1 regressions
#
# Exit 0 = PASS (prints SELF_FALSIFYING_COMPILATION_LINE_R17_GATE_OK), 1 = FAIL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONTRACT="scripts/research/self_falsifying_compilation_line_r17_contract.py"
SPEC="docs/research/self_falsifying_compilation_line_r17_2026-07-28.md"
EXECUTOR="self-hosted/compiler/claim_executor.sio"
FIX="scripts/ci/fixtures"
WB_ELF="$REPO_ROOT/artifacts/self-hosted/madaros-witness-binding"
RECEIPT="$REPO_ROOT/artifacts/self_falsifying_r17_receipt.txt"

fail() {
    echo "SELF_FALSIFYING_COMPILATION_LINE_R17_GATE_FAIL: $*" >&2
    exit 1
}

[[ -f "$SPEC" ]] || fail "spec doc missing: $SPEC"
[[ -f "$CONTRACT" ]] || fail "contract missing: $CONTRACT"
[[ -f "$EXECUTOR" ]] || fail "executor missing: $EXECUTOR"

# ---------------------------------------------------------------- compile arm

if [[ "${SFCL_R17_RUN_COMPILE:-0}" == "1" ]]; then
    RAW="${MADAROS_RAW_BIN:-$WB_ELF}"
    [[ -x "$RAW" ]] || fail "compile arm: no witness-binding compiler at $RAW"
    echo "compile arm: using $RAW"

    ulimit -s unlimited 2>/dev/null || true
    export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$REPO_ROOT/stdlib}"
    TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sfcl_r17.XXXXXX")"
    trap 'rm -rf "$TMP_DIR"' EXIT

    # probe <label> <source> <want_rc:zero|nonzero> <marker> <want_elf:yes|no>
    probe() {
        local label="$1" src="$2" want_rc="$3" marker="$4" want_elf="$5"
        local elf="$TMP_DIR/${label}.elf" log="$TMP_DIR/${label}.log"
        rm -f "$elf"
        set +e
        timeout 300 "$RAW" build --verify-claims "$src" -o "$elf" > "$log" 2>&1
        local rc=$?
        set -e
        if [[ "$want_rc" == "zero" && $rc -ne 0 ]]; then
            cat "$log"; fail "$label: expected rc 0, got $rc"
        fi
        if [[ "$want_rc" == "nonzero" && $rc -eq 0 ]]; then
            cat "$log"; fail "$label: expected non-zero rc, got 0"
        fi
        grep -q "$marker" "$log" || { cat "$log"; fail "$label: no '$marker'"; }
        if [[ "$want_elf" == "yes" && ! -f "$elf" ]]; then
            fail "$label: expected an ELF, none emitted"
        fi
        if [[ "$want_elf" == "no" && -f "$elf" ]]; then
            fail "$label: an ELF was emitted and must not have been"
        fi
        echo "  [OK] $label  rc=$rc elf=$want_elf $marker"
    }

    probe W1_MATCH_PASSES    "$FIX/self_falsifying_witness_pass.sio"   zero    CLAIM_PASS               yes
    probe W2_DRIFT_BLOCKS    "$FIX/self_falsifying_witness_drift.sio"  nonzero CLAIM_WITNESS_MISMATCH   no
    probe W3_ABSENT_BLOCKS   "$FIX/self_falsifying_witness_absent.sio" nonzero CLAIM_WITNESS_ABSENT     no
    probe W4_BACKWARD_COMPAT "$FIX/self_falsifying_witness_compat.sio" zero    CLAIM_PASS               yes
    # Shared code was modified: R2's path must be re-verified, not assumed.
    probe R2_REGRESSION_PASS   "$FIX/self_falsifying_token_pass.sio"   zero    CLAIM_PASS             yes
    probe R2_REGRESSION_DRIFT  "$FIX/self_falsifying_token_drift.sio"  nonzero CLAIM_TOKEN_MISMATCH   no
    probe R2_REGRESSION_ABSENT "$FIX/self_falsifying_token_absent.sio" nonzero CLAIM_TOKEN_ABSENT     no
    probe R01_REGRESSION_EXITCODE "$FIX/self_falsifying_claims_pass_only.sio" zero CLAIM_PASS         yes
    probe R01_REGRESSION_NOOP     "$FIX/self_falsifying_no_claims.sio"       zero VERIFY_CLAIMS_NOOP  yes

    SHA="$(sha256sum "$EXECUTOR" | cut -d' ' -f1)"
    {
        echo "R17 witness-binding behaviour receipt"
        echo "executor_sha256=$SHA"
        echo "compiler=$RAW"
        echo ""
        echo "W1_MATCH_PASSES         rc=0 elf=yes CLAIM_PASS"
        echo "W2_DRIFT_BLOCKS         rc=1 elf=no  CLAIM_WITNESS_MISMATCH   <- the rung"
        echo "W3_ABSENT_BLOCKS        rc=1 elf=no  CLAIM_WITNESS_ABSENT"
        echo "W4_BACKWARD_COMPAT      rc=0 elf=yes CLAIM_PASS (claim declares no witness)"
        echo "R2_REGRESSION_PASS      rc=0 elf=yes CLAIM_PASS"
        echo "R2_REGRESSION_DRIFT     rc=1 elf=no  CLAIM_TOKEN_MISMATCH"
        echo "R2_REGRESSION_ABSENT    rc=1 elf=no  CLAIM_TOKEN_ABSENT"
        echo "R01_REGRESSION_EXITCODE rc=0 elf=yes CLAIM_PASS + CLAIM_SKIP"
        echo "R01_REGRESSION_NOOP     rc=0 elf=yes VERIFY_CLAIMS_NOOP"
        echo ""
        echo "In W2 the gate exited 0 AND emitted exactly the declared verdict token."
        echo "Exit-code gating passes it; token binding passes it; the build was"
        echo "refused anyway."
    } > "$RECEIPT"
    echo "compile arm: receipt written to ${RECEIPT#$REPO_ROOT/}"
fi

# ------------------------------------------------------------------- contract

OUT="$(python3 "$CONTRACT" 2>&1)" || { echo "$OUT"; fail "contract exited non-zero"; }
echo "$OUT"

for clause in X1_EXECUTOR_SURFACE X2_NO_DUPLICATED_DERIVATION X3_BEHAVIOUR_RECEIPT; do
    grep -q "^${clause} PASS" <<<"$OUT" || fail "${clause} did not PASS"
done

# X2 is not decoration. The whole line is about evidence that shares a
# derivation, so a duplicated extractor here would be the paper's own error.
grep -q "exactly one scan body (found 1)" <<<"$OUT" \
    || fail "the token and witness readers no longer share one derivation"

# X3 must be bound to THIS source, not to whatever ran once.
grep -q "receipt is bound to THIS executor source" <<<"$OUT" \
    || fail "the receipt does not match the executor's current sha256"
grep -q "W2: gate exits 0, token correct, build REFUSED" <<<"$OUT" \
    || fail "W2 no longer demonstrates refusal on a preserved proposition"

# Verdict drift, header AND prose.
SPEC_TOKEN="$(grep -m1 -oE '^\*\*Status:\*\* `[^`]*` — `[A-Za-z0-9_]+`' "$SPEC" \
    | grep -oE '`[A-Za-z0-9_]+`$' | tr -d '`' || true)"
[[ -n "$SPEC_TOKEN" ]] || fail "spec declares no verdict token in its Status line"
CONTRACT_TOKEN="$(grep -m1 '^SELF_FALSIFYING_R17_VERDICT ' <<<"$OUT" | awk '{print $2}')"
[[ "$SPEC_TOKEN" == "$CONTRACT_TOKEN" ]] \
    || fail "verdict drift: spec says '${SPEC_TOKEN}', contract emits '${CONTRACT_TOKEN}'"
while read -r t; do
    [[ -z "$t" ]] && continue
    [[ "$t" == "$CONTRACT_TOKEN" ]] \
        || fail "verdict drift in prose: found '${t}', contract emits '${CONTRACT_TOKEN}'"
done < <(grep -oE 'SELF_FALSIFYING_R17_VERDICT [A-Za-z0-9_]+' "$SPEC" | awk '{print $2}')

# The concessions. This rung ships a working compiler feature and the temptation
# is to let it read as more than it is.
grep -q "Not a solution to shared misinterpretation" "$SPEC" \
    || fail "the R0 3 scope limit was deleted"
# These two guarded sentences that R18 made false. Updated rather than dropped:
# the corpus is still overwhelmingly unbound (1 of ~295), and the fact that THIS
# rung shipped without binding the real case is part of its record.
grep -q "1 of ~295" "$SPEC" \
    || fail "the disclosure that the corpus is still unbound was deleted"
grep -q "The real case is now bound" "$SPEC" \
    || fail "the pointer to R18's binding was deleted"
grep -q "the concession is kept here rather than" "$SPEC" \
    || fail "R17's own limit was erased instead of superseded"

echo "SELF_FALSIFYING_COMPILATION_LINE_R17_GATE_OK"
