#!/usr/bin/env bash
# KL-3: fixed-capacity specializer limits fail closed, and an explicit generic
# method turbofish is type-checked and monomorphized instead of being discarded.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
unset SOUNIO_SOUC_ENGINE || true
SOUC="${SOUC:-$ROOT/bin/souc}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT

echo "== madaros_kl3_specializer_gate =="

for src in \
  tests/run-pass/kl3_specializer_four_type_params.sio \
  tests/run-pass/kl3_method_turbofish.sio
do
  name="$(basename "$src" .sio)"
  elf="$OUT/$name.elf"
  log="$OUT/$name.compile.log"
  if ! "$SOUC" compile "$src" -o "$elf" >"$log" 2>&1; then
    echo "FAIL: compile $src"
    tail -40 "$log" || true
    exit 1
  fi
  chmod +x "$elf"
  if ! "$elf" >"$OUT/$name.run.log" 2>&1; then
    echo "FAIL: run $src"
    cat "$OUT/$name.run.log" || true
    exit 1
  fi
done

grep -qF 'KL3_FOUR_TYPE_PARAMS_OK' "$OUT/kl3_specializer_four_type_params.run.log" || {
  echo "FAIL: four-type-parameter control lost its sentinel"
  cat "$OUT/kl3_specializer_four_type_params.run.log" || true
  exit 1
}
grep -qF 'KL3_METHOD_TURBOFISH_OK' "$OUT/kl3_method_turbofish.run.log" || {
  echo "FAIL: method turbofish witness lost its sentinel"
  cat "$OUT/kl3_method_turbofish.run.log" || true
  exit 1
}

for spec in \
  'tests/compile-fail/kl3_specializer_five_type_params.sio|generic declarations support at most 4 type parameters' \
  'tests/compile-fail/kl3_method_turbofish_unsupported.sio|generic method specialization requires one unambiguous explicit turbofish'
do
  src="${spec%%|*}"
  pattern="${spec#*|}"
  log="$OUT/$(basename "$src" .sio).log"
  if "$SOUC" compile "$src" -o "$OUT/refused.elf" >"$log" 2>&1; then
    echo "FAIL: $src was accepted"
    tail -40 "$log" || true
    exit 1
  fi
  grep -qF "$pattern" "$log" || {
    echo "FAIL: $src was rejected for the wrong reason"
    tail -40 "$log" || true
    exit 1
  }
done

SOUC="$SOUC" bash scripts/ci/madaros_specializer_nested_targ_gate.sh
echo "MADAROS_KL3_SPECIALIZER_GATE_OK"
