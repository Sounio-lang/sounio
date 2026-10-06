#!/usr/bin/env bash
# math::qd128 transcendentals vs the independent mpmath corpus.
#
# The oracle values in tests/vectors/qd128_transcendental/qd128_transcendental.jsonl
# were produced by mpmath 1.3.0 at 400 bits (see that file's `_meta` record and
# gen/gen_qd128_transcendental_vectors.py). This gate only REPLAYS them: the
# checker compares in exact rationals and needs nothing but the stdlib.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
SOUC="${SOUC:-$ROOT/bin/souc}"
HARNESS="tests/vectors/qd128_transcendental/gen/qd128_transcendental_harness.sio"
CHECK="scripts/dev/qd128_transcendental_accuracy_check.py"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
ELF="$OUT/qdt.elf"

echo "== qd128_transcendental_accuracy_gate =="

if ! "$SOUC" compile "$HARNESS" -o "$ELF" >"$OUT/compile.log" 2>&1; then
  echo "FAIL: compile"
  tail -40 "$OUT/compile.log" || true
  exit 1
fi
chmod +x "$ELF"
if ! "$ELF" >"$OUT/run.log" 2>&1; then
  echo "FAIL: run"
  tail -40 "$OUT/run.log" || true
  exit 1
fi

if ! python3 "$CHECK" "$OUT/run.log" "$@"; then
  echo "FAIL: accuracy check"
  exit 1
fi

# Cheap in-tree identity/round-trip proof alongside the accuracy measurement.
SRC="tests/stdlib/math/test_qd128_transcendental.sio"
ELF2="$OUT/qdt_rt.elf"
if ! "$SOUC" compile "$SRC" -o "$ELF2" >"$OUT/compile2.log" 2>&1; then
  echo "FAIL: compile run-pass proof"
  tail -40 "$OUT/compile2.log" || true
  exit 1
fi
chmod +x "$ELF2"
if ! "$ELF2" >"$OUT/run2.log" 2>&1; then
  echo "FAIL: run-pass proof"
  cat "$OUT/run2.log" || true
  exit 1
fi
grep -q 'QD128_TRANSCENDENTAL_OK' "$OUT/run2.log" || {
  echo "FAIL: missing sentinel"
  cat "$OUT/run2.log" || true
  exit 1
}

echo "QD128_TRANSCENDENTAL_ACCURACY_GATE_OK"
