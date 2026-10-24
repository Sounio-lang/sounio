#!/usr/bin/env bash
# SCO corpus gate (Step II) — exact runtime sentinels and an attributed V3 observation refusal.
# See docs/research/sco_corpus_2026-08-01.md and
# docs/research/scientific_compilation_obligation_2026-08-01.md.
#
# Green means: default ./bin/souc multi-module native path matches the
# corpus sentinels. It does NOT discharge full SCO (needs Step III E(τ)
# and non-vacuity).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/ci/sco_stack_contract.sh"
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

# The negative keeps imports, payload and expression identical; only the
# callee's Observe declaration is absent. Any unrelated failure is insufficient.
: >"$OUT/V3_negative.status"
negative="examples/sco_corpus/v3_observe/v3_without_observe.sio"
for mode in check compile; do
  rc=0
  rm -f "$OUT/V3_negative.elf"
  if [[ "$mode" == check ]]; then
    timeout 180 "$SOUC" check "$negative" >"$OUT/V3_negative.$mode.log" 2>&1 || rc=$?
  else
    timeout 180 "$SOUC" compile "$negative" -o "$OUT/V3_negative.elf" >"$OUT/V3_negative.$mode.log" 2>&1 || rc=$?
  fi
  printf '%s_rc=%s\n' "$mode" "$rc" >>"$OUT/V3_negative.status"
  if [[ "$rc" != 1 ]] || ! grep -q 'error\[E036\]' "$OUT/V3_negative.$mode.log"; then
    echo "FAIL: V3 negative lacks attributed observation refusal ($mode rc=$rc)"
    fail=1
  elif [[ -s "$OUT/V3_negative.elf" ]]; then
    echo "FAIL: V3 refusal left an executable output"
    fail=1
  else
    echo "PASS: V3 $mode observation refusal E036"
  fi
done

# Bounded observer regression matrix: exact statuses, diagnostic IDs and stdout.
for spec in identity:PASS scalar_control:PASS cast_without_observe:E036 compare_without_observe:E036 implicit_return:E008 wrong_carrier:E036 nested_cast:E036 nested_compare:E004 arithmetic:E036 unknown_compare:E004 unknown_return:E008 inferred_binding:PASS inferred_binding_without_observe:E036 unknown_inner_refinement:E036; do
  name="${spec%%:*}"
  expected="${spec#*:}"
  file="examples/sco_corpus/v3_observe/observer-regressions/$name.sio"
  : >"$OUT/observer_$name.status"
  for mode in check compile; do
    rc=0
    rm -f "$OUT/observer_$name.elf"
    if [[ "$mode" == check ]]; then
      timeout 180 "$SOUC" check "$file" >"$OUT/observer_$name.$mode.log" 2>&1 || rc=$?
    else
      timeout 180 "$SOUC" compile "$file" -o "$OUT/observer_$name.elf" >"$OUT/observer_$name.$mode.log" 2>&1 || rc=$?
    fi
    printf '%s_rc=%s\n' "$mode" "$rc" >>"$OUT/observer_$name.status"
    if [[ "$expected" == PASS ]]; then
      if [[ "$rc" != 0 ]]; then echo "FAIL: observer $name $mode rc=$rc"; fail=1; fi
    elif [[ "$rc" != 1 ]] || ! grep -q "error\[$expected\]" "$OUT/observer_$name.$mode.log" || [[ -s "$OUT/observer_$name.elf" ]]; then
      echo "FAIL: observer $name $mode lacks $expected refusal (rc=$rc)"
      fail=1
    else
      echo "PASS: observer $name $mode $expected refusal"
    fi
  done
  if [[ "$expected" == PASS ]]; then
    if [[ ! -s "$OUT/observer_$name.elf" ]]; then
      echo "FAIL: observer $name missing fresh ELF"; fail=1
    else
      chmod +x "$OUT/observer_$name.elf"
      rc=0
      timeout 30 "$OUT/observer_$name.elf" >"$OUT/observer_$name.stdout" 2>"$OUT/observer_$name.stderr" || rc=$?
      printf 'run_rc=%s\n' "$rc" >>"$OUT/observer_$name.status"
      printf '7\n' >"$OUT/observer_$name.expected"
      if [[ "$rc" != 0 ]] || ! cmp "$OUT/observer_$name.expected" "$OUT/observer_$name.stdout"; then
        echo "FAIL: observer $name exact runtime"; fail=1
      else
        echo "PASS: observer $name exact runtime"
      fi
    fi
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "SCO_CORPUS_GATE_FAIL"
  exit 1
fi

echo "SCO_CORPUS_GATE_OK"
exit 0
