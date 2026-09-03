#!/usr/bin/env bash
# scripts/ci/ontology_hash_benchmark.sh
#
# Benchmarks the hash-index fast-path for ontology CURIE resolution.
# Compiles a temporary binary with ontology dispatch, resolves 50 random
# CURIEs, and reports total/average time.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC="$ROOT/bin/souc-linux-x86_64"
SRC="$ROOT/self-hosted/compiler/lean_single.sio"
TMP_DIR="$(mktemp -d /tmp/sounio-ontology-bench.XXXXXX)"
PATCHED_SRC="$TMP_DIR/ontology_cli.sio"
TEST_BIN="$TMP_DIR/ontology_cli"

cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

# Verify binary exists
if [[ ! -x "$SOUC" ]]; then
    echo "ERROR: binary not found: $SOUC" >&2
    exit 1
fi

# Patch main() to dispatch to ontology_run_cli()
python3 - "$SRC" "$PATCHED_SRC" <<'PYEOF'
import sys
src_path, dst_path = sys.argv[1], sys.argv[2]
with open(src_path, 'r') as f:
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

with open(dst_path, 'w') as f:
    f.writelines(lines)
PYEOF

# Compile patched source
if ! "$SOUC" "$PATCHED_SRC" "$TEST_BIN" >/dev/null 2>&1; then
    echo "ERROR: failed to compile patched ontology source" >&2
    exit 1
fi
chmod +x "$TEST_BIN"

# Gather 50 random CURIEs from the ontology kernel
mapfile -t curies < <("$TEST_BIN" ontology list "" 2>/dev/null | sed 's/ | .*//' | shuf -n 50)

if [[ ${#curies[@]} -eq 0 ]]; then
    echo "ERROR: failed to collect CURIEs" >&2
    exit 1
fi

# Warm up: resolve once to ensure index is loaded
"$TEST_BIN" ontology resolve "${curies[0]}" >/dev/null 2>&1

# Benchmark: resolve all 50 CURIEs via the hash index (fast path)
TIMEFORMAT='%R'
total_time_s=$({ time {
    for curie in "${curies[@]}"; do
        "$TEST_BIN" ontology resolve "$curie" >/dev/null 2>&1
    done
}; } 2>&1)

# Report
awk -v total="$total_time_s" -v n="${#curies[@]}" 'BEGIN {
    printf "=== ontology hash benchmark ===\n"
    printf "Terms: 1008\n"
    printf "Resolutions: %d\n", n
    printf "Total time: %.3fs\n", total
    printf "Average per resolve: %.3fms\n", (total / n) * 1000
}'
