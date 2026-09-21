#!/usr/bin/env bash
# Run with a freshly rebuilt modular compiler: MADAROS_RAW_BIN=/path/madaros.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export MADAROS_RAW_BIN="${MADAROS_RAW_BIN:?set MADAROS_RAW_BIN to the freshly rebuilt modular ELF}"
export MADAROS_BIN="$ROOT/bin/madaros"
source scripts/lib/resolve_madaros.sh
sounio_require_madaros
MADAROS="$MADAROS_BIN"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
export SOUNIO_SPEC_TRACE=1
export SOUNIO_DUMP_MERGED_CALLS=1
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
compile_run() {
    local stem="$1" expected="$2"
    "$MADAROS" compile "tests/run-pass/$stem.sio" -o "$OUT/$stem.elf" >"$OUT/$stem.compile" 2>&1 || {
        cat "$OUT/$stem.compile"; return 1;
    }
    if grep -Eq 'NATIVE_REFUSAL|empty_stub_ud2|missing_lowered_body|poisoned' "$OUT/$stem.compile"; then
        cat "$OUT/$stem.compile"; return 1
    fi
    chmod +x "$OUT/$stem.elf"
    "$OUT/$stem.elf" >"$OUT/$stem.stdout"
    diff -u <(printf '%s\n' "$expected") "$OUT/$stem.stdout"
}
compile_run specializer_scalar_multi_instance SCALAR_MULTI_INSTANCE_OK
compile_run trait_bounded_dispatch_multi_call $'11\n5\n6\nmulti_call PASS'
# Inspect the production merge dump, including its post-resolution call edges.
# The single-module callsite probe bypasses specialization and is unsuitable.
for ty in i64 f64; do
    symbol="__sp_005twiceAN003${ty}EE"
    grep -Fxq "specializer: out_fn $symbol" "$OUT/specializer_scalar_multi_instance.compile"
    grep -Fq "name=$symbol ->$symbol ic=" "$OUT/specializer_scalar_multi_instance.compile"
done
if grep -Fxq 'specializer: out_fn twice' "$OUT/specializer_scalar_multi_instance.compile"; then
    echo 'FAIL: unspecialized twice survived'; exit 1
fi
# Emitted struct instances are remembered by exact mangled name. `Cell__B0` and
# `Cell__AQ` share a DJB2 hash, and a hash-only cache emitted just one of them. The
# program prints the same either way; the trace counts emitted struct instances.
compile_run specializer_emitted_instance_hash_collision 'PASS specializer_emitted_instance_hash_collision'
if ! grep -Fxq 'specializer: instances=2' "$OUT/specializer_emitted_instance_hash_collision.compile"; then
    echo 'FAIL: colliding struct instances were merged into one'
    grep -F 'specializer: instances=' "$OUT/specializer_emitted_instance_hash_collision.compile" || true
    exit 1
fi
echo MADAROS_SCALAR_MULTI_INSTANCE_GATE_OK
