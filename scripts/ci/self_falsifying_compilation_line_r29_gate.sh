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
CHAIN_MAIN="scripts/ci/fixtures/mcl_chain_main.sio"
CHAIN_LEAF="scripts/ci/fixtures/mcl_chain_leaf.sio"
DIAMOND_MAIN="scripts/ci/fixtures/mcl_diamond_main.sio"
MIXED_MAIN="scripts/ci/fixtures/mcl_mixed_main.sio"
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

# ------------------------------------------------------------------ A5
# The depth-2 and diamond probes are what take the transitive claim past one
# hop. If the chain flattens to a direct import, or the leaf stops being red,
# D5 silently becomes another copy of D1 and the claim loses its evidence.
grep -q '^use mcl_chain_mid::' "$CHAIN_MAIN" \
    || fail "A5: the chain probe no longer goes through its middle module"
grep -q '^use mcl_chain_leaf::' "scripts/ci/fixtures/mcl_chain_mid.sio" \
    || fail "A5: the chain probe's middle module no longer imports the leaf"
grep -q 'self_falsifying_claim_gate_fail.sh' "$CHAIN_LEAF" \
    || fail "A5: the chain probe's leaf claim is no longer red"
grep -q '^use mcl_diamond_a::' "$DIAMOND_MAIN" && grep -q '^use mcl_diamond_b::' "$DIAMOND_MAIN" \
    || fail "A5: the diamond probe no longer has two arms"
grep -q '^use mcl_diamond_leaf::' "scripts/ci/fixtures/mcl_diamond_a.sio" \
    || fail "A5: the diamond's arms no longer share a leaf"
echo "A5_TRANSITIVE_PROBES PASS — depth-2 chain and shared-leaf diamond intact"

# ------------------------------------------------------------------ A6
# The mixed probe is the only clause that observes discrimination directly. It
# needs BOTH imports; drop either and it degenerates into D1 or D2.
grep -q '^use mcl_green_lib::' "$MIXED_MAIN" \
    || fail "A6: the mixed probe lost its green import"
grep -q '^use self_falsifying_modclosure_lib::' "$MIXED_MAIN" \
    || fail "A6: the mixed probe lost its red import"
echo "A6_MIXED_PROBE PASS — one green and one red import in a single compilation"

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

# D5 — transitivity: the refuted claim is TWO hops away. A depth-1 walk passes
# D1 and fails here, which is the whole reason this clause exists.
rc=$(run_probe d5 "$CHAIN_MAIN" --verify-claims)
[[ $rc -ne 0 ]] || fail "D5: a claim refuted two imports away did not block"
grep -q '^VERIFY_CLAIMS_SCOPE modules=3$' "$TMP_DIR/d5.txt" \
    || fail "D5: closure is not three nodes — the walk is not transitive"
grep -q '^CLAIM_FAIL mcl_chain_leaf_claim_false' "$TMP_DIR/d5.txt" \
    || fail "D5: blocked, but not by the two-hop leaf"
[[ ! -f "$TMP_DIR/d5.elf" ]] || fail "D5: ELF emitted despite a two-hop falsified claim"
echo "D5_TRANSITIVE PASS — refuted claim two imports away aborts before codegen"

# D6 — a module reachable by two paths is verified once, not twice.
rc=$(run_probe d6 "$DIAMOND_MAIN" --verify-claims)
[[ $rc -eq 0 ]] || { cat "$TMP_DIR/d6.txt" >&2; fail "D6: all-green diamond did not compile"; }
grep -q '^VERIFY_CLAIMS_SCOPE modules=4$' "$TMP_DIR/d6.txt" \
    || fail "D6: diamond closure is not four nodes"
grep -q '^VERIFY_CLAIMS_OK pass=4$' "$TMP_DIR/d6.txt" \
    || fail "D6: shared leaf counted more than once (expected pass=4)"
echo "D6_SHARED_NODE PASS — leaf reachable two ways verified once"

# D7 — discrimination, observed in one run rather than inferred from D1+D2.
rc=$(run_probe d7 "$MIXED_MAIN" --verify-claims)
[[ $rc -ne 0 ]] || fail "D7: a red import alongside a green one did not block"
grep -q '^CLAIM_PASS mcl_green_library_claim' "$TMP_DIR/d7.txt" \
    || fail "D7: the green import was not reported as a pass — compiler may fail all imports"
grep -q '^CLAIM_FAIL mcl_library_claim_that_is_false' "$TMP_DIR/d7.txt" \
    || fail "D7: the red import was not reported as a failure"
[[ ! -f "$TMP_DIR/d7.elf" ]] || fail "D7: ELF emitted despite a falsified import"
echo "D7_DISCRIMINATES PASS — green passes and red blocks in the same compilation"

echo "SELF_FALSIFYING_COMPILATION_LINE_R29_GATE_OK"
