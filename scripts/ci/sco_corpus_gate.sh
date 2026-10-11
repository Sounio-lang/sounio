#!/usr/bin/env bash
# SCO corpus gate (Step II) — exact runtime sentinels (V3 remains an ordinary effect smoke).
# See docs/research/sco_corpus_2026-08-01.md and
# docs/research/scientific_compilation_obligation_2026-08-01.md.
#
# Green means: default ./bin/souc multi-module native path matches the
# corpus sentinels. It does NOT discharge full SCO (needs Step III E(τ)
# and non-vacuity).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
SOUC="${SOUC:-$ROOT/bin/souc}"

OUT="${SCO_OUTPUT_DIR:-$ROOT/artifacts/sco}/corpus"
mkdir -p "$OUT"
fail=0

run_item() {
  local name="$1"
  local driver="$2"
  local expected_file="$3"

  echo "== SCO $name =="
  echo "driver: $driver"

  local compile_rc=0 run_rc=0
  rm -f "$OUT/${name}.elf"
  timeout 180 "$SOUC" compile "$driver" -o "$OUT/${name}.elf" >"$OUT/${name}.compile.out" 2>"$OUT/${name}.compile.err" || compile_rc=$?
  printf 'compile_rc=%s\n' "$compile_rc" >"$OUT/${name}.status"
  if [[ "$compile_rc" != 0 ]]; then
    echo "FAIL: compile"
    tail -20 "$OUT/${name}.compile.err" || true
    fail=1
    return
  fi

  [[ -s "$OUT/${name}.elf" ]] || { echo "FAIL: compile produced no fresh ELF"; fail=1; return; }

  # Madaros may emit a non-executable ELF mode; force +x before run.
  chmod +x "$OUT/${name}.elf" 2>/dev/null || true

  timeout 30 "$OUT/${name}.elf" >"$OUT/${name}.stdout" 2>"$OUT/${name}.stderr" || run_rc=$?
  printf 'run_rc=%s\n' "$run_rc" >>"$OUT/${name}.status"
  if [[ "$run_rc" != 0 ]]; then
    echo "FAIL: run (non-zero exit)"
    cat "$OUT/${name}.stderr" || true
    fail=1
    return
  fi

  # No stdout normalization: the fixture promises exact bytes.
  if ! cmp "$expected_file" "$OUT/${name}.stdout"; then
    echo "FAIL: sentinel mismatch"
    echo "--- expected ---"
    cat "$expected_file"
    echo "--- got ---"
    cat "$OUT/${name}.stdout"
    fail=1
    return
  fi

  echo "PASS: sentinel exact"
}

# Expected outputs (generated next to run for auditability)
mkdir -p "$OUT/expected"
printf '%s\n' '2776' >"$OUT/expected/v1.txt"
printf '%s\n' '250000' '4250000' >"$OUT/expected/v2.txt"
printf '%s\n' '1' '7001' >"$OUT/expected/v3.txt"

run_item "V1_coverage" \
  "examples/sco_corpus/v1_coverage/v1_main.sio" \
  "$OUT/expected/v1.txt"

run_item "V2_algebra" \
  "examples/sco_corpus/v2_algebra/v2_main.sio" \
  "$OUT/expected/v2.txt"

run_item "V3_observe" \
  "examples/sco_corpus/v3_observe/v3_main.sio" \
  "$OUT/expected/v3.txt"

if [[ "$fail" -ne 0 ]]; then
  echo "SCO_CORPUS_GATE_FAIL"
  exit 1
fi

echo "SCO_CORPUS_GATE_OK"
exit 0
