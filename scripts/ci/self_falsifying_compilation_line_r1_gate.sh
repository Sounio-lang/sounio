#!/usr/bin/env bash
# scripts/ci/self_falsifying_compilation_line_r1_gate.sh
#
# CI gate for rung R1 of the self-falsifying compilation line
# (docs/research/self_falsifying_compilation_line_r1_2026-07-26.md): binding
# real CI gates to native claims, and the module-closure wall.
#
# Static clauses (always run, via the contract):
#   B1_MANIFEST_BOUND   real source carries claims bound to real gates
#   B2_GATES_EXIST      every bound gate exists and is executable
#   B3_MODULE_CLOSURE   the module-closure probe is shaped to be decisive
#   B4_TIMEOUT_BUDGET   no over-budget gate is bound
#   B5_HERMETIC         no bound gate mutates the working tree
#
# Compile arm (SFCL_R1_RUN_COMPILE=1) — re-measures the compile-time facts.
# Needs a claim-aware Madaros (MADAROS_RAW_BIN, or
# artifacts/self-hosted/madaros-self-falsifying) and runs every bound gate, so
# it costs roughly a minute:
#   C1_MANIFEST_VERIFIES  all bound claims pass; ELF emitted
#   C2_RED_GATE_BLOCKS    one gate swapped for an always-failing fixture:
#                         VERIFY_CLAIMS_FALSIFIED, non-zero exit, NO ELF
#   C3_MODULE_CLOSURE     importer of a module with a FALSE claim still
#                         compiles ⇒ imported claims are never executed
#
# Exit 0 = PASS (prints SELF_FALSIFYING_COMPILATION_LINE_R1_GATE_OK), 1 = FAIL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONTRACT="scripts/research/self_falsifying_compilation_line_r1_contract.py"
SPEC="docs/research/self_falsifying_compilation_line_r1_2026-07-26.md"
MANIFEST="examples/epistemic/rupture_claims_verified.sio"
MC_MAIN="scripts/ci/fixtures/self_falsifying_modclosure_main.sio"
CLAIM_ELF="$REPO_ROOT/artifacts/self-hosted/madaros-self-falsifying"

fail() {
    echo "SELF_FALSIFYING_COMPILATION_LINE_R1_GATE_FAIL: $*" >&2
    exit 1
}

[[ -f "$SPEC" ]] || fail "spec doc missing: $SPEC"
[[ -f "$CONTRACT" ]] || fail "contract missing: $CONTRACT"

OUT="$(python3 "$CONTRACT" 2>&1)" || { echo "$OUT"; fail "contract exited non-zero"; }
echo "$OUT"

for clause in B1_MANIFEST_BOUND B2_GATES_EXIST B3_MODULE_CLOSURE B4_TIMEOUT_BUDGET B5_HERMETIC; do
    grep -q "^${clause} PASS" <<<"$OUT" || fail "${clause} did not PASS"
done

# Drift guard: the spec's declared verdict token must be what the contract emits.
SPEC_TOKEN="$(grep -m1 -oE '^\*\*Status:\*\* `[^`]*` — `[A-Za-z0-9_]+`' "$SPEC" \
    | grep -oE '`[A-Za-z0-9_]+`$' | tr -d '`' || true)"
[[ -n "$SPEC_TOKEN" ]] || fail "spec declares no verdict token in its Status line"
CONTRACT_TOKEN="$(grep -m1 '^SELF_FALSIFYING_R1_VERDICT ' <<<"$OUT" | awk '{print $2}')"
[[ "$SPEC_TOKEN" == "$CONTRACT_TOKEN" ]] \
    || fail "verdict drift: spec says '${SPEC_TOKEN}', contract emits '${CONTRACT_TOKEN}'"

if [[ "${SFCL_R1_RUN_COMPILE:-0}" != "1" ]]; then
    echo "compile arm: SKIPPED (set SFCL_R1_RUN_COMPILE=1 to re-measure)"
    echo "SELF_FALSIFYING_COMPILATION_LINE_R1_GATE_OK"
    exit 0
fi

RAW="${MADAROS_RAW_BIN:-$CLAIM_ELF}"
[[ -x "$RAW" ]] || fail "compile arm: no claim-aware compiler at $RAW"
echo "compile arm: using $RAW"

ulimit -s unlimited 2>/dev/null || true
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$REPO_ROOT/stdlib}"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sfcl_r1.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

# Take the count from the contract's own summary line rather than re-deriving
# it by regex over the listing (claim names contain digits: g2_, cd_tower_...).
BOUND_N="$(grep -m1 -oE 'claims bound to real gates : [0-9]+' <<<"$OUT" | grep -oE '[0-9]+$')"
[[ -n "$BOUND_N" && "$BOUND_N" -gt 0 ]] || fail "could not read bound-claim count from contract output"

# C1: the manifest verifies and codegen proceeds.
set +e
C1_OUT="$("$RAW" run "$MANIFEST" -o "$TMP_DIR/manifest.elf" --verify-claims 2>&1)"
C1_RC=$?
set -e
[[ $C1_RC -eq 0 ]] || { echo "$C1_OUT" >&2; fail "C1: manifest failed to compile (rc=$C1_RC)"; }
grep -q "VERIFY_CLAIMS_OK pass=${BOUND_N}" <<<"$C1_OUT" \
    || fail "C1: expected VERIFY_CLAIMS_OK pass=${BOUND_N}, got: $(grep VERIFY_CLAIMS <<<"$C1_OUT")"
[[ -f "$TMP_DIR/manifest.elf" ]] || fail "C1: no ELF emitted despite all claims passing"
echo "C1_MANIFEST_VERIFIES PASS — ${BOUND_N} real gates verified before codegen"

# C2: swapping one bound gate for an always-failing fixture must block codegen.
# Target the FIRST real gate binding rather than a named gate, so renaming or
# removing any single claim cannot silently turn this into a no-op test.
awk '!done && /^[[:space:]]*gate[[:space:]]*=[[:space:]]*"scripts\/ci\// {
        sub(/"scripts\/ci\/[^"]*"/, "\"scripts/ci/fixtures/self_falsifying_claim_gate_fail.sh\"")
        done = 1
     } { print }' "$MANIFEST" > "$TMP_DIR/red.sio"
if diff -q "$MANIFEST" "$TMP_DIR/red.sio" >/dev/null; then
    fail "C2: gate substitution was a no-op — the manifest has no real gate binding to redden"
fi
set +e
C2_OUT="$("$RAW" run "$TMP_DIR/red.sio" -o "$TMP_DIR/red.elf" --verify-claims 2>&1)"
C2_RC=$?
set -e
[[ $C2_RC -ne 0 ]] || { echo "$C2_OUT" >&2; fail "C2: red gate did not block compilation"; }
grep -q "VERIFY_CLAIMS_FALSIFIED" <<<"$C2_OUT" || fail "C2: expected VERIFY_CLAIMS_FALSIFIED"
[[ ! -f "$TMP_DIR/red.elf" ]] || fail "C2: ELF emitted despite a falsified claim"
echo "C2_RED_GATE_BLOCKS PASS — falsified claim aborts before codegen, no ELF"

# C3: module closure. The importer carries one passing claim; the module it
# imports carries a claim whose gate always fails. A clean compile reporting
# pass=1 proves the imported claim was never executed.
set +e
C3_OUT="$("$RAW" run "$MC_MAIN" -o "$TMP_DIR/mc.elf" --verify-claims 2>&1)"
C3_RC=$?
set -e
# R29 (2026-08-01) closed this wall: claim_executor_verify now walks the module
# closure, so the imported false claim executes and blocks the build. R1's
# original outcome — MODULE_CLOSURE_BLOCKS, measured 2026-07-26 — was a true
# observation of the compiler as it stood then; it is superseded, not retracted.
# The polarity below is therefore inverted ON PURPOSE. A return to BLOCKS is now
# the regression, and is what this clause fails on.
if [[ $C3_RC -ne 0 ]] && grep -q "VERIFY_CLAIMS_FALSIFIED" <<<"$C3_OUT"; then
    grep -q "^CLAIM_FAIL mcl_library_claim_that_is_false" <<<"$C3_OUT" \
        || fail "C3: build blocked, but not by the imported claim — probe no longer decisive"
    [[ ! -f "$TMP_DIR/mc.elf" ]] || fail "C3: ELF emitted despite a falsified imported claim"
    echo "C3_MODULE_CLOSURE PASS — MODULE_CLOSURE_PASSES: imported claims execute (R29)"
elif [[ $C3_RC -eq 0 ]] && grep -q "VERIFY_CLAIMS_OK pass=1" <<<"$C3_OUT"; then
    fail "C3: MODULE_CLOSURE_BLOCKS — the import wall is back. R29 closed it; \
either claim_executor_verify lost its closure walk or this binary predates it. \
Do not edit the token to match: re-measure, then re-derive."
else
    echo "$C3_OUT" >&2
    fail "C3: probe outcome unrecognised (rc=$C3_RC)"
fi

echo "SELF_FALSIFYING_COMPILATION_LINE_R1_GATE_OK"
