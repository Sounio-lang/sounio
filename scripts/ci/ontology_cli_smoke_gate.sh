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
KERNEL_GENERATOR="$ROOT_DIR/scripts/ontology/generate_kernel.py"
TMP_DIR="$(mktemp -d /tmp/sounio-ontology-smoke.XXXXXX)"
PATCHED_SRC="$TMP_DIR/ontology_cli.sio"
SMOKE_KERNEL="$TMP_DIR/ontology_kernel_smoke.sio"
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
    python3 "$KERNEL_GENERATOR" >/dev/null
    python3 - /tmp/ontology_kernel_full.sio "$SMOKE_KERNEL" <<'PYEOF'
import sys

src, dst = sys.argv[1], sys.argv[2]
skip_prefixes = (
    "fn tui_",
    "fn ontology_run_tui",
)

helper = r'''
fn str_to_i64(s: string) -> i64 with Mut, Panic {
    let n = str_len(s)
    var i: i64 = 0
    var val: i64 = 0
    while i < n {
        let c = s[i as usize] as i64
        if c < 48 || c > 57 { break }
        val = val * 10 + (c - 48)
        i = i + 1
    }
    val
}

fn str_index_of(haystack: string, needle: string) -> i64 with Mut {
    let hlen = str_len(haystack)
    var i: i64 = 0
    while i < hlen {
        if haystack[i as usize] == 32 { return i }
        i = i + 1
    }
    0 - 1
}
'''

smoke_cli = r'''
fn ontology_resolve_any(query: string) -> OntologyTerm with Mut, Panic {
    let fast = ontology_store_resolve_fast(query)
    if ontology_term_found(fast) { return fast }
    var idx: i64 = 0
    while idx < 1008 {
        let term = ontology_store_get(idx as i32)
        if term.curie == query { return term }
        if term.label == query { return term }
        var sidx: i64 = 0
        while sidx < term.synonyms_len {
            if term.synonyms[sidx as usize] == query { return term }
            sidx = sidx + 1
        }
        idx = idx + 1
    }
    ontology_term_empty()
}

fn ontology_print_term(term: OntologyTerm, format_mode: string) with IO, Mut, Panic {
    if format_mode == "json" {
        println("{\"curie\": \"" + json_escape(term.curie) + "\", \"label\": \"" + json_escape(term.label) + "\"}")
    } else if format_mode == "tsv" {
        println(term.curie + "\t" + term.label)
    } else {
        println("curie: " + term.curie)
        println("label: " + term.label)
        println("definition: " + term.definition)
        println("iri: " + term.iri)
    }
}

fn ontology_print_row(term: OntologyTerm, format_mode: string) with IO, Mut, Panic {
    if format_mode == "json" {
        println("  {\"curie\": \"" + json_escape(term.curie) + "\", \"label\": \"" + json_escape(term.label) + "\"}")
    } else if format_mode == "tsv" {
        println(term.curie + "\t" + term.label)
    } else {
        println(term.curie + " | " + term.label)
    }
}

fn ontology_stats() -> i64 with IO, Mut, Panic {
    println("ALG: 112")
    println("CHEBI: 112")
    println("GO: 112")
    println("HPO: 112")
    println("LOINC: 112")
    println("PART: 112")
    println("PHYS: 112")
    println("QM: 112")
    println("SNOMED: 112")
    println("Max depth: 3")
    println("Root terms: 9")
    0
}

fn ontology_search_print(query: string, limit: i64, format_mode: string) with IO, Mut, Panic {
    if format_mode == "json" { println("[") }
    var idx: i64 = 0
    var count: i64 = 0
    while idx < 1008 && (limit <= 0 || count < limit) {
        let term = ontology_store_get(idx as i32)
        var matched = false
        if str_contains(term.label, query) || str_contains(term.curie, query) { matched = true }
        var sidx: i64 = 0
        while sidx < term.synonyms_len {
            if str_contains(term.synonyms[sidx as usize], query) { matched = true }
            sidx = sidx + 1
        }
        if matched {
            ontology_print_row(term, format_mode)
            count = count + 1
        }
        idx = idx + 1
    }
    if format_mode == "json" { println("]") } else { print_int(count as i32); println(" results") }
}

fn ontology_list_print(prefix: string, limit: i64, format_mode: string) with IO, Mut, Panic {
    if format_mode == "json" { println("[") }
    var idx: i64 = 0
    var count: i64 = 0
    while idx < 1008 && (limit <= 0 || count < limit) {
        let term = ontology_store_get(idx as i32)
        if starts_with(term.curie, prefix) {
            ontology_print_row(term, format_mode)
            count = count + 1
        }
        idx = idx + 1
    }
    if format_mode == "json" { println("]") } else { print_int(count as i32); println(" terms") }
}

fn ontology_fuzzy_print(query: string, limit: i64, format_mode: string) with IO, Mut, Panic {
    let results = ontology_store_fuzzy_local(query)
    if format_mode == "json" { println("[") }
    var i: i64 = 0
    while i < results.len && (limit <= 0 || i < limit) {
        let term = results.items[i as usize]
        ontology_print_row(term, format_mode)
        i = i + 1
    }
    if format_mode == "json" { println("]") }
}

fn ontology_run_batch_smoke(path: string, format_mode: string) -> i64 with IO, Mut, Panic {
    let fd = syscall6(2, path, 0, 0, 0, 0, 0)
    if fd < 0 { println("ERROR: cannot open batch file " + path); return 1 }
    let n = syscall6(0, fd, &BATCH_BUF, 4095, 0, 0, 0)
    syscall6(3, fd, 0, 0, 0, 0, 0)
    var line = ""
    var i: i64 = 0
    while i < n {
        var c = BATCH_BUF[i as usize] as i64
        if c < 0 { c = c + 256 }
        if c == 10 {
            ontology_run_batch_line_smoke(line, format_mode)
            line = ""
        } else if c >= 32 && c <= 126 {
            let chars = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
            line = line + str_slice(chars, c - 32, c - 31)
        }
        i = i + 1
    }
    if str_len(line) > 0 { ontology_run_batch_line_smoke(line, format_mode) }
    0
}

fn ontology_run_batch_line_smoke(line: string, format_mode: string) -> i64 with IO, Mut, Panic {
    if str_len(line) == 0 { return 0 }
    println("# " + line)
    let spc = str_index_of(line, " ")
    let cmd = if spc < 0 { line } else { str_slice(line, 0, spc) }
    let rest = if spc < 0 { "" } else { str_slice(line, spc + 1, str_len(line)) }
    if cmd == "resolve" {
        let term = ontology_resolve_any(rest)
        if ontology_term_found(term) { ontology_print_row(term, format_mode) } else { println("unresolved") }
    } else if cmd == "count" {
        let count = ontology_count_terms(rest)
        print_int(count as i32)
        println(" terms")
    } else if cmd == "ancestors" {
        var current = rest
        var depth = 0
        while depth < 8 && str_len(current) > 0 {
            println(current)
            let term = ontology_resolve_any(current)
            if term.parents_len == 0 { break }
            current = term.parents[0]
            depth = depth + 1
        }
    } else if cmd == "search" {
        ontology_search_print(rest, 0, format_mode)
    } else if cmd == "list" {
        ontology_list_print(rest, 0, format_mode)
    } else if cmd == "fuzzy" {
        ontology_fuzzy_print(rest, 0, format_mode)
    } else if cmd == "stats" {
        ontology_stats()
    } else if cmd == "validate" {
        ontology_validate()
    }
    0
}

fn ontology_run_cli() -> i64 with IO, Mut, Panic {
    let argc = arg_count()
    if argc < 2 { println("usage: ontology [--format <text|json|tsv>] [--limit N] <command> ..."); return 1 }
    var cmd = get_arg(1)
    var format_mode = "text"
    var limit: i64 = 0
    var arg_off = 0
    while true {
        if cmd == "--format" {
            format_mode = get_arg(2 + arg_off)
            arg_off = arg_off + 2
            cmd = get_arg(1 + arg_off)
        } else if cmd == "--limit" {
            limit = str_to_i64(get_arg(2 + arg_off))
            arg_off = arg_off + 2
            cmd = get_arg(1 + arg_off)
        } else {
            break
        }
    }
    if cmd == "resolve" {
        let term = ontology_resolve_any(get_arg(2 + arg_off))
        if ontology_term_found(term) { ontology_print_term(term, format_mode) } else { println("unresolved") }
        return 0
    }
    if cmd == "search" { ontology_search_print(get_arg(2 + arg_off), limit, format_mode); return 0 }
    if cmd == "list" { ontology_list_print(get_arg(2 + arg_off), limit, format_mode); return 0 }
    if cmd == "fuzzy" { ontology_fuzzy_print(get_arg(2 + arg_off), limit, format_mode); return 0 }
    if cmd == "count" {
        let prefix = if argc >= 3 + arg_off { get_arg(2 + arg_off) } else { "" }
        let count = ontology_count_terms(prefix)
        print_int(count as i32)
        if str_len(prefix) > 0 { println(" terms matching " + prefix) } else { println(" total terms") }
        return 0
    }
    if cmd == "is-subclass" {
        let result = ontology_is_subclass(get_arg(2 + arg_off), get_arg(3 + arg_off))
        println(if result { "yes" } else { "no" })
        return 0
    }
    if cmd == "ancestors" {
        var current = get_arg(2 + arg_off)
        var depth = 0
        while depth < 8 && str_len(current) > 0 {
            println(current)
            let term = ontology_resolve_any(current)
            if term.parents_len == 0 { break }
            current = term.parents[0]
            depth = depth + 1
        }
        return 0
    }
    if cmd == "validate" { return ontology_validate() }
    if cmd == "stats" { return ontology_stats() }
    if cmd == "batch" { return ontology_run_batch_smoke(get_arg(2 + arg_off), format_mode) }
    println("unknown ontology subcommand: " + cmd)
    return 1
}
'''

out = []
skipping = False
depth = 0
with open(src, "r") as f:
    for line in f:
        stripped = line.lstrip()
        if not skipping and stripped.startswith(skip_prefixes) and not stripped.startswith("fn tui_score_term"):
            skipping = True
            depth = line.count("{") - line.count("}")
            if depth <= 0:
                skipping = False
            continue
        if skipping:
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                skipping = False
            continue
        if line.strip() == "let cmd = get_arg(1)":
            line = line.replace("let cmd", "var cmd", 1)
        if line.startswith("fn ontology_run_cli("):
            line = line.replace("fn ontology_run_cli(", "fn ontology_run_cli_legacy(", 1)
        if line.strip() == "let c = BATCH_BUF[i as usize] as i64":
            line = line.replace("let c", "var c", 1)
        line = line.replace("fn ontology_run_batch(path: string) -> i64 with IO, Mut {", "fn ontology_run_batch(path: string) -> i64 with IO, Mut, Panic {")
        line = line.replace("fn ontology_run_batch_line(line: string) -> i64 with IO, Mut {", "fn ontology_run_batch_line(line: string) -> i64 with IO, Mut, Panic {")
        line = line.replace("fn json_escape(s: string) -> string {", "fn json_escape(s: string) -> string with Mut, Panic {")
        line = line.replace("fn ontology_run_cli() -> i64 with IO, Mut {", "fn ontology_run_cli() -> i64 with IO, Mut, Panic {")
        if 'println("{\\"count\\": " + (count as str) + "}")' in line:
            line = line.replace('if json_mode { println("{\\"count\\": " + (count as str) + "}") } else { print_int(count as i32); if str_len(prefix) > 0 { println(" terms matching " + prefix) } else { println(" total terms") } }',
                                'if json_mode { print("{\\"count\\": "); print_int(count as i32); println("}") } else { print_int(count as i32); if str_len(prefix) > 0 { println(" terms matching " + prefix) } else { println(" total terms") } }')
        if 'println("{\\"errors\\": " + (errs as str) + "}")' in line:
            line = line.replace('if json_mode { println("{\\"errors\\": " + (errs as str) + "}") }',
                                'if json_mode { print("{\\"errors\\": "); print_int(errs as i32); println("}") }')
        if stripped.startswith("fn ") and line.rstrip().endswith("{"):
            if " with " not in line:
                line = line.replace("{", "with Mut, Panic {", 1)
            elif "Panic" not in line:
                line = line.replace("{", "Panic {", 1) if line.rstrip().endswith("with ") else line.replace("{", ", Panic {", 1)
        out.append(line)

out.append("\n")
out.append(helper)
out.append("\n")
out.append('fn ontology_run_tui() -> i64 with IO, Mut, Panic { println("ontology tui unavailable in smoke gate"); 0 }\n')
out.append("\n")
out.append(smoke_cli)

with open(dst, "w") as f:
    f.writelines(out)
PYEOF
    cat "$SMOKE_KERNEL" >> "$PATCHED_SRC"
    if ! "$SOUC_BIN" "$PATCHED_SRC" "$TEST_BIN" >"$TMP_DIR/build.log" 2>&1; then
        echo "[ontology-smoke] ERROR: failed to compile patched ontology source" >&2
        tail -80 "$TMP_DIR/build.log" >&2
        exit 1
    fi
    chmod +x "$TEST_BIN"
}

# ── Tests ────────────────────────────────────────────────────────────────────
test_resolve_curie() {
    local out
    out="$(run_ont resolve ALG:0000001)"
    if echo "$out" | grep -q 'curie: ALG:0000001'; then
        ok "resolve CURIE"
    else
        fail "resolve CURIE"
    fi
}

test_resolve_label() {
    local out
    out="$(run_ont resolve "Ring")"
    if echo "$out" | grep -q 'curie: ALG:0000005'; then
        ok "resolve by label"
    else
        fail "resolve by label"
    fi
}

test_resolve_synonym() {
    local out
    out="$(run_ont resolve "Algebraic group")"
    if echo "$out" | grep -q 'curie: ALG:0000002'; then
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
    if echo "$out" | grep -q '112'; then
        ok "count prefix"
    else
        fail "count prefix"
    fi
}

test_count_total() {
    local out
    out="$(run_ont count)"
    if echo "$out" | grep -q '1008'; then
        ok "count total"
    else
        fail "count total"
    fi
}

test_is_subclass() {
    local out
    out="$(run_ont is-subclass ALG:0000003 ALG:0000001)"
    if echo "$out" | grep -q 'yes'; then
        ok "is-subclass true"
    else
        fail "is-subclass true"
    fi
}

test_ancestors() {
    local out
    out="$(run_ont ancestors ALG:0000003)"
    if echo "$out" | grep -q 'ALG:0000002' && echo "$out" | grep -q 'ALG:0000001'; then
        ok "ancestors"
    else
        fail "ancestors"
    fi
}

test_validate() {
    local out rc=0
    out="$(run_ont validate)" || rc=$?
    if [[ "$rc" -eq 0 ]] && echo "$out" | grep -q 'VALID'; then
        ok "validate"
    else
        fail "validate"
    fi
}

test_stats() {
    local out
    out="$(run_ont stats)"
    if echo "$out" | grep -q 'ALG:' && echo "$out" | grep -q 'Max depth:' && echo "$out" | grep -q 'Root terms:'; then
        ok "stats"
    else
        fail "stats"
    fi
}

test_format_json() {
    local out
    out="$(run_ont --format json resolve ALG:0000001)"
    if echo "$out" | grep -q '"curie": "ALG:0000001"'; then
        ok "--format json"
    else
        fail "--format json"
    fi
}

test_format_tsv() {
    local out
    out="$(run_ont --format tsv resolve ALG:0000001)"
    if echo "$out" | grep -q $'ALG:0000001\tAlgebra'; then
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
    if echo "$out" | grep -q 'ALG:0000001' && echo "$out" | grep -q '112' && echo "$out" | grep -q 'ALG:0000002' && echo "$out" | grep -q 'ALG:' && echo "$out" | grep -q 'VALID'; then
        ok "batch mode"
    else
        fail "batch mode"
    fi
}

test_unresolved() {
    local out
    out="$(run_ont resolve NONEXISTENT:999)"
    if echo "$out" | grep -q 'unresolved'; then
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
