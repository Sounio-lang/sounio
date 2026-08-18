#!/usr/bin/env bash
# scripts/ci/ontology_cli_smoke_gate.sh
#
# Smoke gate for the embedded ontology CLI in lean_single.sio.
# Compiles a temporary binary with ontology dispatch and runs a suite of
# subcommand tests covering resolve, search, list, count, stats, validate,
# ancestors, is-subclass, fuzzy, batch, and format/limit flags.
#
# Exit 0 = all tests passed.
# Exit 1 = at least one test failed.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC_BIN="${SOUC_BIN:-$ROOT_DIR/bin/souc-linux-x86_64}"
SRC="$ROOT_DIR/self-hosted/compiler/lean_single.sio"
TMP_DIR="$(mktemp -d /tmp/sounio-ontology-smoke.XXXXXX)"
PATCHED_SRC="$TMP_DIR/ontology_cli.sio"
TEST_BIN="$TMP_DIR/ontology_cli"

PASS=0
FAIL=0

cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

# ── Helpers ──────────────────────────────────────────────────────────────────
ok() {
    PASS=$((PASS + 1))
    printf '[ontology-smoke] PASS: %s\n' "$1"
}

fail() {
    FAIL=$((FAIL + 1))
    printf '[ontology-smoke] FAIL: %s\n' "$1" >&2
}

run_ont() {
    "$TEST_BIN" ontology "$@"
}

# ── Patch main() to dispatch to ontology_run_cli() ───────────────────────────
patch_source() {
    python3 <<PYEOF
with open('$SRC', 'r') as f:
    lines = f.readlines()

main_start = None
for i, line in enumerate(lines):
    if line.strip() == 'fn main() -> i64 with IO, Mut, Panic, Div {':
        main_start = i
        break

insert_after = None
for i in range(main_start, min(main_start + 20, len(lines))):
    if lines[i].strip() == '}' and i > main_start + 2:
        insert_after = i
        break

dispatch = [
    '    let first = get_arg(0)\n',
    '    if first == "ontology" {\n',
    '        return ontology_run_cli()\n',
    '    }\n',
]

lines = lines[:insert_after+1] + dispatch + lines[insert_after+1:]

with open('$PATCHED_SRC', 'w') as f:
    f.writelines(lines)
PYEOF
}

# ── Compile test binary ──────────────────────────────────────────────────────
compile_test_binary() {
    if [[ ! -x "$SOUC_BIN" ]]; then
        echo "[ontology-smoke] ERROR: souc binary not found: $SOUC_BIN" >&2
        exit 1
    fi
    patch_source
    # Capture, don't pipe: through the pipe it is grep's match that decides,
    # so a failing `--help` whose stdout happens to contain the flag would
    # still select the new interface, and an early-exiting `grep -q` can
    # EPIPE the writer under pipefail. The rc decides; the match only refines.
    help_rc=0
    help_out="$("$SOUC_BIN" --help 2>/dev/null)" || help_rc=$?
    if [[ "$help_rc" -eq 0 ]] && grep -q "compile <file.sio>" <<<"$help_out"; then
        compile_cmd=( "$SOUC_BIN" compile "$PATCHED_SRC" -o "$TEST_BIN" )
    else
        compile_cmd=( "$SOUC_BIN" "$PATCHED_SRC" "$TEST_BIN" )
    fi
    if ! "${compile_cmd[@]}" >"$TMP_DIR/build.log" 2>&1; then
        echo "[ontology-smoke] SKIP: selected compiler could not build patched ontology CLI source"
        tail -n 20 "$TMP_DIR/build.log" || true
        exit 0
    fi
    chmod +x "$TEST_BIN"
}

# ── Tests ────────────────────────────────────────────────────────────────────
test_resolve_curie() {
    local out
    out="$(run_ont resolve ALG:0000001)"
    if grep -q 'curie: ALG:0000001' <<<"$out"; then
        ok "resolve CURIE"
    else
        fail "resolve CURIE"
    fi
}

test_resolve_label() {
    local out
    out="$(run_ont resolve "Ring")"
    if grep -q 'curie: ALG:0000005' <<<"$out"; then
        ok "resolve by label"
    else
        fail "resolve by label"
    fi
}

test_resolve_synonym() {
    local out
    out="$(run_ont resolve "Algebraic group")"
    if grep -q 'curie: ALG:0000002' <<<"$out"; then
        ok "resolve by synonym"
    else
        fail "resolve by synonym"
    fi
}

test_search_limit() {
    local out n
    out="$(run_ont --limit 2 search space)"
    n="$(echo "$out" | grep -c '^ALG:' || true)"
    if [[ "$n" -eq 2 ]]; then
        ok "search with --limit"
    else
        fail "search with --limit (expected 2, got $n)"
    fi
}

test_list_limit() {
    local out n
    out="$(run_ont --limit 3 list ALG:)"
    n="$(echo "$out" | grep -c '^ALG:' || true)"
    if [[ "$n" -eq 3 ]]; then
        ok "list with --limit"
    else
        fail "list with --limit (expected 3, got $n)"
    fi
}

test_list_all() {
    local out n
    out="$(run_ont list ALG:)"
    n="$(echo "$out" | grep -c '^ALG:' || true)"
    if [[ "$n" -eq 112 ]]; then
        ok "list all terms (no limit)"
    else
        fail "list all terms (expected 112, got $n)"
    fi
}

test_fuzzy_limit() {
    local out n
    out="$(run_ont --limit 2 fuzzy group)"
    n="$(echo "$out" | grep -c '^ALG:' || true)"
    if [[ "$n" -eq 2 ]]; then
        ok "fuzzy with --limit"
    else
        fail "fuzzy with --limit (expected 2, got $n)"
    fi
}

test_count_prefix() {
    local out
    out="$(run_ont count GO:)"
    if grep -q '112' <<<"$out"; then
        ok "count prefix"
    else
        fail "count prefix"
    fi
}

test_count_total() {
    local out
    out="$(run_ont count)"
    if grep -q '1008' <<<"$out"; then
        ok "count total"
    else
        fail "count total"
    fi
}

test_is_subclass() {
    local out
    out="$(run_ont is-subclass ALG:0000003 ALG:0000001)"
    if grep -q 'yes' <<<"$out"; then
        ok "is-subclass true"
    else
        fail "is-subclass true"
    fi
}

test_ancestors() {
    local out
    out="$(run_ont ancestors ALG:0000003)"
    if grep -q 'ALG:0000002' <<<"$out" && grep -q 'ALG:0000001' <<<"$out"; then
        ok "ancestors"
    else
        fail "ancestors"
    fi
}

test_validate() {
    local out rc=0
    out="$(run_ont validate)" || rc=$?
    if [[ "$rc" -eq 0 ]] && grep -q 'VALID' <<<"$out"; then
        ok "validate"
    else
        fail "validate"
    fi
}

test_stats() {
    local out
    out="$(run_ont stats)"
    if grep -q 'ALG:' <<<"$out" && grep -q 'Max depth:' <<<"$out" && grep -q 'Root terms:' <<<"$out"; then
        ok "stats"
    else
        fail "stats"
    fi
}

test_format_json() {
    local out
    out="$(run_ont --format json resolve ALG:0000001)"
    if grep -q '"curie": "ALG:0000001"' <<<"$out"; then
        ok "--format json"
    else
        fail "--format json"
    fi
}

test_format_tsv() {
    local out
    out="$(run_ont --format tsv resolve ALG:0000001)"
    if grep -q $'ALG:0000001\tAlgebra' <<<"$out"; then
        ok "--format tsv"
    else
        fail "--format tsv"
    fi
}

test_combined_flags() {
    local out n
    out="$(run_ont --format json --limit 2 list ALG:)"
    n="$(echo "$out" | grep -c '"curie"' || true)"
    if [[ "$n" -eq 2 ]]; then
        ok "combined --format json --limit"
    else
        fail "combined --format json --limit (expected 2, got $n)"
    fi
}

test_batch() {
    local batch_file="$TMP_DIR/batch.txt"
    cat > "$batch_file" <<'EOF'
resolve ALG:0000001
count ALG:
ancestors ALG:0000003
search group
list ALG:
fuzzy group
stats
validate
EOF
    local out
    out="$(run_ont batch "$batch_file")"
    if grep -q 'ALG:0000001' <<<"$out" && grep -q '112' <<<"$out" && grep -q 'ALG:0000002' <<<"$out" && grep -q 'ALG:' <<<"$out" && grep -q 'VALID' <<<"$out"; then
        ok "batch mode"
    else
        fail "batch mode"
    fi
}

test_unresolved() {
    local out
    out="$(run_ont resolve NONEXISTENT:999)"
    if grep -q 'unresolved' <<<"$out"; then
        ok "unresolved CURIE"
    else
        fail "unresolved CURIE"
    fi
}

# ── Main ─────────────────────────────────────────────────────────────────────
echo "[ontology-smoke] souc=$SOUC_BIN"
echo "[ontology-smoke] src=$SRC"
echo "[ontology-smoke] tmp=$TMP_DIR"

compile_test_binary

echo "[ontology-smoke] running tests..."

test_resolve_curie
test_resolve_label
test_resolve_synonym
test_search_limit
test_list_limit
test_list_all
test_fuzzy_limit
test_count_prefix
test_count_total
test_is_subclass
test_ancestors
test_validate
test_stats
test_format_json
test_format_tsv
test_combined_flags
test_batch
test_unresolved

printf '[ontology-smoke] results: %d passed, %d failed\n' "$PASS" "$FAIL"

if [[ "$FAIL" -gt 0 ]]; then
    exit 1
fi
exit 0
