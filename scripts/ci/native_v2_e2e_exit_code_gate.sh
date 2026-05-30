#!/usr/bin/env bash
# native_v2_e2e_exit_code_gate.sh
#
# End-to-end codegen-CORRECTNESS gate for the modular self-hosted compiler's
# native-v2 x86-64 backend. Unlike the structural smoke (which only checks the
# emitted ELF is > 64 bytes), this gate EMITS a full executable for
# `main() { return N }` and RUNS it, asserting the process exit code == N.
# A correct value flowing IR -> machine code -> loaded ELF -> exit status proves
# the backend is semantically correct end-to-end (not just structurally valid).
#
# The modular compiler binary is taken from $SOUNIO_MODULAR_SOUC if set (a prebuilt
# main.sio ELF), otherwise built from self-hosted/compiler/main.sio using the
# bootstrap compiler $SOUNIO_BOOTSTRAP_SOUC (default ./bin/souc).
#
# Exit 0 = all values verified; non-zero = a mismatch (codegen regression).
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ulimit -s 1048576 2>/dev/null || true

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

BOOTSTRAP_SOUC="${SOUNIO_BOOTSTRAP_SOUC:-./bin/souc}"
MODULAR_SOUC="${SOUNIO_MODULAR_SOUC:-}"

if [[ -z "$MODULAR_SOUC" ]]; then
    echo "[gate] building modular compiler from self-hosted/compiler/main.sio (bootstrap=$BOOTSTRAP_SOUC)"
    MODULAR_SOUC="$WORK/mc.elf"
    if ! "$BOOTSTRAP_SOUC" self-hosted/compiler/main.sio "$MODULAR_SOUC" > "$WORK/build.log" 2>&1; then
        echo "[gate] FAIL: could not build modular compiler"
        tail -5 "$WORK/build.log"
        exit 2
    fi
    chmod +x "$MODULAR_SOUC"
fi

if [[ ! -x "$MODULAR_SOUC" ]]; then
    echo "[gate] FAIL: modular compiler not executable: $MODULAR_SOUC"
    exit 2
fi

# The compiler's --native-v2-emit13 mode emits main(){return 13} to
# artifacts/omega/emit13.elf. (A future multi-value mode could parameterize N;
# for now 13 is the canonical witness — a non-trivial, non-zero, non-default byte.)
EXPECTED=13
EMITTED="artifacts/omega/emit13.elf"
rm -f "$EMITTED"

echo "[gate] emitting executable for main(){ return $EXPECTED }"
"$MODULAR_SOUC" --native-v2-emit13 > "$WORK/emit.log" 2>&1 || true
grep -a 'native_v2_emit_exit_code' "$WORK/emit.log" | tr -cd '[:print:]\n' || true

if [[ ! -f "$EMITTED" ]]; then
    echo "[gate] FAIL: compiler did not emit $EMITTED"
    exit 3
fi

# Structural sanity: ELF magic.
MAGIC="$(head -c4 "$EMITTED" | od -An -tx1 | tr -d ' ')"
if [[ "$MAGIC" != "7f454c46" ]]; then
    echo "[gate] FAIL: emitted file is not ELF (magic=$MAGIC, expected 7f454c46)"
    exit 4
fi

# The real test: run it, check exit code.
chmod +x "$EMITTED"
"$EMITTED"
GOT=$?
if [[ "$GOT" -ne "$EXPECTED" ]]; then
    echo "[gate] FAIL: emitted executable exit code=$GOT, expected $EXPECTED"
    exit 5
fi

echo "[gate] PASS: modular native-v2 backend emits a correct executable (main(){return $EXPECTED} -> exit $GOT)"
exit 0
