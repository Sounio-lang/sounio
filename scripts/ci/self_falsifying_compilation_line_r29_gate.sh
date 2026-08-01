#!/usr/bin/env bash
# Self-falsifying compilation line, R29 — claim verification walks the module
# closure.  docs/research/self_falsifying_compilation_line_r29_2026-08-01.md
#
# Static arm (always): the change is present in the source, and the probes that
# measured it still have the shape that makes them decisive.
# Compile arm (SFCL_R29_RUN_COMPILE=1): re-measures D1-D4 against a real binary.
#
# Every grep here is prefix-anchored. print_int newline-terminates, so a count
# never shares a line with the field after it; an unanchored "pass=2" would
# match "pass=20" and this gate would pass while the mechanism regressed.
set -uo pipefail

cd "$(git rev-parse --show-toplevel)" || exit 9

EXEC="self-hosted/compiler/claim_executor.sio"
MC_MAIN="scripts/ci/fixtures/self_falsifying_modclosure_main.sio"
MC_LIB="scripts/ci/fixtures/self_falsifying_modclosure_lib.sio"
GREEN_MAIN="scripts/ci/fixtures/mcl_green_main.sio"
GREEN_LIB="scripts/ci/fixtures/mcl_green_lib.sio"
MANIFEST="examples/epistemic/rupture_claims_verified.sio"

fail() { echo "SELF_FALSIFYING_COMPILATION_LINE_R29_GATE_FAIL: $*" >&2; exit 1; }

# ------------------------------------------------------------------ A1
# The executor collects the closure and iterates it. Checked as three separate
# facts: a zero-result census of any one of them would otherwise look like a
# pass.
grep -q 'module_frontend_collect_ast_closure_into(source_path, &!closure)' "$EXEC" \
    || fail "A1: $EXEC no longer collects the module closure"
grep -q 'while mod_idx < scope_count' "$EXEC" \
    || fail "A1: $EXEC no longer iterates the closure"
grep -q 'let module_path = closure.paths\[mod_idx\]' "$EXEC" \
    || fail "A1: $EXEC no longer verifies per closure node"
echo "A1_CLOSURE_WALK PASS — executor collects the closure and verifies each node"

# ------------------------------------------------------------------ A2
# A truncated scope must announce itself rather than print a smaller total under
# the same label.
grep -q 'VERIFY_CLAIMS_SCOPE_PARTIAL' "$EXEC" \
    || fail "A2: an incomplete closure would report its total without saying it was cut"
grep -q 'VERIFY_CLAIMS_SCOPE modules=' "$EXEC" \
    || fail "A2: the verified scope is no longer reported"
echo "A2_SCOPE_REPORTED PASS — scope printed, truncation announced"

# ------------------------------------------------------------------ A3
# The probes stay decisive only while the red fixture is red, the green fixture
# is green, and each importer carries a claim of its own — without that last
# one, "the imported claim ran" is indistinguishable from "nothing ran".
grep -q 'self_falsifying_claim_gate_fail.sh' "$MC_LIB" \
    || fail "A3: the red probe's imported claim no longer binds a failing gate"
grep -q '^claim ' "$MC_MAIN" \
    || fail "A3: $MC_MAIN carries no claim of its own — probe not decisive"
grep -q 'self_falsifying_claim_gate_pass.sh' "$GREEN_LIB" \
    || fail "A3: the green probe's imported claim no longer binds a passing gate"
grep -q '^claim ' "$GREEN_MAIN" \
    || fail "A3: $GREEN_MAIN carries no claim of its own — probe not decisive"
grep -q '^use mcl_green_lib::' "$GREEN_MAIN" \
    || fail "A3: the green probe no longer imports its library"
echo "A3_PROBES_DECISIVE PASS — red and green import pairs both intact"

# ------------------------------------------------------------------ A4
# D4's invariance arm is only an invariance arm while the manifest has no
# imports. If one is ever added, its closure stops being one node and the
# clause silently changes meaning.
if grep -qE '^use ' "$MANIFEST"; then
    fail "A4: $MANIFEST gained an import — D4 is no longer a one-node invariance test"
fi
echo "A4_INVARIANCE_ARM PASS — manifest still import-free, closure is one node"

echo
echo "SELF_FALSIFYING_R29_VERDICT CLOSURE_WALKED__MODULE_CLOSURE_PASSES"

# ------------------------------------------------------------------ compile arm
if [[ "${SFCL_R29_RUN_COMPILE:-0}" != "1" ]]; then
    echo "compile arm: SKIPPED (set SFCL_R29_RUN_COMPILE=1 to re-measure D1-D4)"
    echo "SELF_FALSIFYING_COMPILATION_LINE_R29_GATE_OK"
    exit 0
fi

RAW="${MADAROS_RAW_BIN:-artifacts/self-hosted/madaros}"
[[ -x "$RAW" ]] || fail "compile arm: no usable compiler at $RAW"
echo "compile arm: using $RAW"
ulimit -s 524288 2>/dev/null || true
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

run_probe() { # name source [args...]
    local n="$1" s="$2"; shift 2
    "$RAW" compile "$s" "$@" -o "$TMP_DIR/$n.elf" > "$TMP_DIR/$n.txt" 2>&1
    echo $?
}

# D1 — a red claim one import away blocks the build.
rc=$(run_probe d1 "$MC_MAIN" --verify-claims)
[[ $rc -ne 0 ]] || fail "D1: red imported claim did not block (rc=0)"
grep -q '^VERIFY_CLAIMS_FALSIFIED fail=1$' "$TMP_DIR/d1.txt" \
    || fail "D1: expected VERIFY_CLAIMS_FALSIFIED fail=1"
grep -q '^CLAIM_FAIL mcl_library_claim_that_is_false' "$TMP_DIR/d1.txt" \
    || fail "D1: blocked, but not by the imported claim"
[[ ! -f "$TMP_DIR/d1.elf" ]] || fail "D1: ELF emitted despite a falsified imported claim"
echo "D1_IMPORT_BLOCKS PASS — refuted claim in an imported module aborts before codegen"

# D2 — and a green one does not. Without this, D1 is satisfied by a compiler
# that simply fails everything it imports.
rc=$(run_probe d2 "$GREEN_MAIN" --verify-claims)
[[ $rc -eq 0 ]] || { cat "$TMP_DIR/d2.txt" >&2; fail "D2: green import pair did not compile"; }
grep -q '^VERIFY_CLAIMS_OK pass=2$' "$TMP_DIR/d2.txt" \
    || fail "D2: expected VERIFY_CLAIMS_OK pass=2 (importer + imported)"
[[ -f "$TMP_DIR/d2.elf" ]] || fail "D2: no ELF despite two green claims"
echo "D2_IMPORT_PASSES PASS — both claims execute, build proceeds"

# D3 — none of this happens without the flag.
rc=$(run_probe d3 "$MC_MAIN")
[[ $rc -eq 0 ]] || fail "D3: red pair failed to build without --verify-claims"
[[ -f "$TMP_DIR/d3.elf" ]] || fail "D3: no ELF without --verify-claims"
grep -q '^VERIFY_CLAIMS' "$TMP_DIR/d3.txt" \
    && fail "D3: verification ran without --verify-claims"
echo "D3_OPT_IN PASS — verification stays opt-in"

# D4 — an import-free file must be untouched by the widening.
rc=$(run_probe d4 "$MANIFEST" --verify-claims)
grep -q '^VERIFY_CLAIMS_SCOPE modules=1$' "$TMP_DIR/d4.txt" \
    || fail "D4: import-free manifest no longer reports a one-node closure"
n_pass=$(grep -c '^CLAIM_PASS ' "$TMP_DIR/d4.txt")
echo "D4_INVARIANT PASS — one-node closure, ${n_pass} claims green (rc=$rc)"

echo "SELF_FALSIFYING_COMPILATION_LINE_R29_GATE_OK"
