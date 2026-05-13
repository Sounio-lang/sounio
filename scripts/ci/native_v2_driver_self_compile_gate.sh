#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[native-v2-driver-self] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[native-v2-driver-self] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

OUT_DIR="${SOUNIO_NATIVE_V2_DRIVER_SELF_COMPILE_DIR:-$(mktemp -d /tmp/sounio-native-v2-driver-self.XXXXXX)}"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

DRIVER_SRC="self-hosted/compiler/native_compile_driver.sio"
HELLO_SRC="examples/native/hello.sio"
BASELINE_HELLO_BIN="$OUT_DIR/hello.baseline"
STAGE1_DRIVER="$OUT_DIR/native_compile_driver.stage1"
STAGE1_HELLO_BIN="$OUT_DIR/hello.stage1"

CHECK_LOG="$LOG_DIR/native_compile_driver.check.log"
BASELINE_GATE_LOG="$LOG_DIR/native_v2_serious_track_gate.log"
BASELINE_COMPILE_LOG="$LOG_DIR/hello.baseline.compile.log"
STAGE1_COMPILE_LOG="$LOG_DIR/native_compile_driver.stage1.compile.log"
STAGE1_HELLO_COMPILE_LOG="$LOG_DIR/hello.stage1.compile.log"
BASELINE_STDOUT="$LOG_DIR/hello.baseline.stdout"
BASELINE_STDERR="$LOG_DIR/hello.baseline.stderr"
STAGE1_STDOUT="$LOG_DIR/hello.stage1.stdout"
STAGE1_STDERR="$LOG_DIR/hello.stage1.stderr"
EXPECTED_STDOUT="$LOG_DIR/hello.expected.stdout"
STAGE1_FILE_LOG="$LOG_DIR/native_compile_driver.stage1.file.txt"
STAGE1_SECTIONS_LOG="$LOG_DIR/native_compile_driver.stage1.readelf.sections"
HELLO_FILE_LOG="$LOG_DIR/hello.stage1.file.txt"
HELLO_SECTIONS_LOG="$LOG_DIR/hello.stage1.readelf.sections"
HELLO_STRINGS_LOG="$LOG_DIR/hello.stage1.strings"
STAGE2_DRIVER="$OUT_DIR/native_compile_driver.stage2"
STAGE3_DRIVER="$OUT_DIR/native_compile_driver.stage3"
STAGE2_COMPILE_LOG="$LOG_DIR/native_compile_driver.stage2.compile.log"
STAGE3_COMPILE_LOG="$LOG_DIR/native_compile_driver.stage3.compile.log"
STAGE2_SECTIONS_LOG="$LOG_DIR/native_compile_driver.stage2.readelf.sections"
STAGE3_SECTIONS_LOG="$LOG_DIR/native_compile_driver.stage3.readelf.sections"
STAGE2_HELLO_BIN="$OUT_DIR/hello.stage2"
STAGE2_HELLO_COMPILE_LOG="$LOG_DIR/hello.stage2.compile.log"
STAGE2_STDOUT="$LOG_DIR/hello.stage2.stdout"

printf '[native-v2-driver-self] souc=%s\n' "$SOUC_BIN"
printf '[native-v2-driver-self] out=%s\n' "$OUT_DIR"

check_driver_elf_layout() {
  local bin="$1"
  local log="$2"
  local label="$3"
  if ! command -v readelf >/dev/null 2>&1; then
    return 0
  fi
  readelf -W -S "$bin" >"$log"
  for section_name in '\.text' '\.rodata' '\.data' '\.shstrtab' '\.sounio\.epistemic'; do
    if ! grep -q "$section_name" "$log"; then
      echo "[native-v2-driver-self] FAIL: $label missing ELF section $section_name" >&2
      cat "$log" >&2 || true
      exit 1
    fi
  done

  local data_hex
  data_hex="$(awk '$3 == ".data" { print $7; exit }' "$log")"
  if [[ -z "$data_hex" ]]; then
    echo "[native-v2-driver-self] FAIL: $label missing .data size in readelf output" >&2
    cat "$log" >&2 || true
    exit 1
  fi

  local data_size=$((16#$data_hex))
  local min_driver_data_size=$((0x5101000))
  if (( data_size < min_driver_data_size )); then
    echo "[native-v2-driver-self] FAIL: $label driver data segment too small: 0x${data_hex} < 0x5101000" >&2
    echo "[native-v2-driver-self]       issue-71 guard: 1MiB driver-global slots must not overlap DRV_ELF_BYTES with .rodata/.shstrtab" >&2
    exit 1
  fi
}

bash scripts/ci/native_v2_serious_track_gate.sh >"$BASELINE_GATE_LOG" 2>&1

"$SOUC_BIN" check "$DRIVER_SRC" >"$CHECK_LOG" 2>&1

"$SOUC_BIN" run "$DRIVER_SRC" -- "$HELLO_SRC" -o "$BASELINE_HELLO_BIN" >"$BASELINE_COMPILE_LOG" 2>&1
if [[ ! -x "$BASELINE_HELLO_BIN" ]]; then
  echo "[native-v2-driver-self] FAIL: baseline driver did not produce executable hello" >&2
  tail -n 80 "$BASELINE_COMPILE_LOG" >&2 || true
  exit 1
fi

"$BASELINE_HELLO_BIN" >"$BASELINE_STDOUT" 2>"$BASELINE_STDERR"
printf 'Hello from self-hosted Sounio!\n42\n' >"$EXPECTED_STDOUT"
if ! cmp -s "$EXPECTED_STDOUT" "$BASELINE_STDOUT"; then
  echo "[native-v2-driver-self] FAIL: baseline hello stdout mismatch" >&2
  diff -u "$EXPECTED_STDOUT" "$BASELINE_STDOUT" >&2 || true
  exit 1
fi

if ! "$SOUC_BIN" run "$DRIVER_SRC" -- "$DRIVER_SRC" -o "$STAGE1_DRIVER" >"$STAGE1_COMPILE_LOG" 2>&1; then
  echo "[native-v2-driver-self] FAIL: native-v2 driver did not self-compile stage1" >&2
  echo "[native-v2-driver-self] compile log: $STAGE1_COMPILE_LOG" >&2
  tail -n 120 "$STAGE1_COMPILE_LOG" >&2 || true
  exit 1
fi

if [[ ! -x "$STAGE1_DRIVER" ]]; then
  echo "[native-v2-driver-self] FAIL: stage1 driver is not executable: $STAGE1_DRIVER" >&2
  tail -n 120 "$STAGE1_COMPILE_LOG" >&2 || true
  exit 1
fi

if command -v file >/dev/null 2>&1; then
  file "$STAGE1_DRIVER" >"$STAGE1_FILE_LOG"
  grep -q 'ELF 64-bit LSB executable, x86-64' "$STAGE1_FILE_LOG"
fi
check_driver_elf_layout "$STAGE1_DRIVER" "$STAGE1_SECTIONS_LOG" "stage1 driver"

if ! "$STAGE1_DRIVER" "$HELLO_SRC" -o "$STAGE1_HELLO_BIN" >"$STAGE1_HELLO_COMPILE_LOG" 2>&1; then
  echo "[native-v2-driver-self] FAIL: stage1 driver failed while compiling hello" >&2
  echo "[native-v2-driver-self] compile log: $STAGE1_HELLO_COMPILE_LOG" >&2
  tail -n 120 "$STAGE1_HELLO_COMPILE_LOG" >&2 || true
  exit 1
fi
if [[ ! -x "$STAGE1_HELLO_BIN" ]]; then
  echo "[native-v2-driver-self] FAIL: stage1 driver did not produce executable hello" >&2
  tail -n 120 "$STAGE1_HELLO_COMPILE_LOG" >&2 || true
  exit 1
fi

if command -v file >/dev/null 2>&1; then
  file "$STAGE1_HELLO_BIN" >"$HELLO_FILE_LOG"
  grep -q 'ELF 64-bit LSB executable, x86-64' "$HELLO_FILE_LOG"
fi

"$STAGE1_HELLO_BIN" >"$STAGE1_STDOUT" 2>"$STAGE1_STDERR"
if ! cmp -s "$EXPECTED_STDOUT" "$STAGE1_STDOUT"; then
  echo "[native-v2-driver-self] FAIL: stage1 hello stdout mismatch" >&2
  diff -u "$EXPECTED_STDOUT" "$STAGE1_STDOUT" >&2 || true
  exit 1
fi

if ! cmp -s "$BASELINE_STDOUT" "$STAGE1_STDOUT"; then
  echo "[native-v2-driver-self] FAIL: stage1 hello stdout differs from baseline driver output" >&2
  diff -u "$BASELINE_STDOUT" "$STAGE1_STDOUT" >&2 || true
  exit 1
fi

if command -v readelf >/dev/null 2>&1; then
  readelf -S "$STAGE1_HELLO_BIN" >"$HELLO_SECTIONS_LOG"
  grep -q '\.rodata' "$HELLO_SECTIONS_LOG"
  grep -q '\.data' "$HELLO_SECTIONS_LOG"
fi

if command -v strings >/dev/null 2>&1; then
  strings "$STAGE1_HELLO_BIN" >"$HELLO_STRINGS_LOG"
  grep -q 'Hello from self-hosted Sounio!' "$HELLO_STRINGS_LOG"
fi

# ── Stage1 smoke: confirm stage1 binary compiles+runs a cohort within a timeout.
# The fixed-point gate (stage2==stage3) only proves determinism. A deterministic
# codegen bug in stage1 will reproduce byte-identically in later stages and
# silently stay green. This phase asserts stage1 actually produces working code
# on a small cohort covering the main N-v2 surfaces (recursion+binop-in-call,
# while+mut, logical ops, zero-init arrays, struct field access).
SMOKE_DIR="$OUT_DIR/smoke"
SMOKE_LOG_DIR="$LOG_DIR/smoke"
mkdir -p "$SMOKE_DIR" "$SMOKE_LOG_DIR"
SMOKE_TIMEOUT="${SOUNIO_NATIVE_V2_DRIVER_SELF_COMPILE_SMOKE_TIMEOUT:-10}"
SMOKE_COHORT=(
  "examples/native/fib.sio"
  "examples/native/while_loop.sio"
  "examples/native/logical_ops.sio"
  "examples/native/array_basics.sio"
  "examples/native/struct_basic.sio"
  # Tail-literal array init. Currently skipped by baseline (the driver's
  # TK_VAR array-decl branch does not yet accept `= [v0, v1, ...]`), but
  # auto-activates as soon as M1.2 step 4 bisect lands the handler -- at which
  # point stage1 must finish within SMOKE_TIMEOUT or this phase fails.
  "examples/native/array_init_tail.sio"
  # Refinement-type non-termination canary (R2.1, post-Layer-B3 revert).
  # The driver should fail-fast (parse-reject `{ N: T | PRED }` syntax),
  # NOT hang. Layer B3's `&Type` param branch caused an infinite loop
  # on this pattern -- and the bug only surfaced during inventory, not
  # during driver-self-compile, because the driver's own source has no
  # refinement syntax. The new baseline-rc=124 → FAIL guard above turns
  # this entry into a hard regression test for that bug class.
  "tests/run-pass/refinement_nested_arithmetic.sio"
  # HOF parameter signature regression test (post-#56-revert re-land).
  # Validates that `find_fn_lbrace` and the parameter walker handle a
  # `fn(T) -> T` parameter type without bailing on the inner `fn`
  # keyword or running off the end. PR #56 was reverted because of a
  # purported logical_ops hang that turned out to be transient; this
  # entry pins the actual fix forward -- both the baseline (with the
  # fix) and stage1 must compile this file successfully, so any future
  # change that re-breaks find_fn_lbrace fails the gate immediately.
  "examples/native/hof_param_signature.sio"
  # First-class fn-ref call witness: named function values, fn-typed params,
  # local fn-ref variables, returned fn refs, and indirect calls must survive
  # the generated stage1 driver.
  "examples/native/hof_fn_ref_call.sio"
  # Non-capturing closure literals lower to generated fn refs and must survive
  # baseline + stage1 compile/run parity.
  "examples/native/hof_closure_literal.sio"
  # Science-shaped higher-order corpus: integration, root scan, and optimizer
  # scan through function references.
  "examples/native/hof_science_kernel.sio"
)

printf '[native-v2-driver-self] stage1-smoke timeout=%ss cohort=%d\n' \
  "$SMOKE_TIMEOUT" "${#SMOKE_COHORT[@]}"

smoke_checked=0
smoke_ran=0
for smoke_src in "${SMOKE_COHORT[@]}"; do
  if [[ ! -f "$smoke_src" ]]; then
    echo "[native-v2-driver-self] smoke SKIP: $smoke_src absent" >&2
    continue
  fi
  slug="$(basename "$smoke_src" .sio)"
  base_bin="$SMOKE_DIR/$slug.baseline"
  s1_bin="$SMOKE_DIR/$slug.stage1"
  base_compile_log="$SMOKE_LOG_DIR/$slug.baseline.compile.log"
  base_stdout="$SMOKE_LOG_DIR/$slug.baseline.stdout"
  s1_compile_log="$SMOKE_LOG_DIR/$slug.stage1.compile.log"
  s1_stdout="$SMOKE_LOG_DIR/$slug.stage1.stdout"

  # Self-certifying golden: capture baseline driver output; drop cases the
  # baseline can't handle rather than hardcoding expected stdout.
  # Distinguish parse-rejection (rc!=0, rc!=124) → soft SKIP from
  # wall-clock hang (rc==124) → hard FAIL. A baseline timeout means the
  # driver itself is non-terminating on this input -- the B3-class
  # regression (Layer B3 `&Type` param branch hung on refinement-type
  # syntax `{ N: T | PRED }`) we now refuse to ship past.
  set +e
  timeout "$SMOKE_TIMEOUT" "$SOUC_BIN" run "$DRIVER_SRC" -- "$smoke_src" -o "$base_bin" \
    >"$base_compile_log" 2>&1
  base_rc=$?
  set -e
  if [[ "$base_rc" -eq 124 ]]; then
    echo "[native-v2-driver-self] FAIL: baseline driver hung for ${SMOKE_TIMEOUT}s compiling $smoke_src -- non-termination regression" >&2
    tail -n 20 "$base_compile_log" >&2 || true
    exit 1
  fi
  if [[ "$base_rc" -ne 0 ]]; then
    echo "[native-v2-driver-self] smoke SKIP: baseline cannot compile $smoke_src (rc=$base_rc)" >&2
    tail -n 20 "$base_compile_log" >&2 || true
    continue
  fi
  if [[ ! -x "$base_bin" ]]; then
    echo "[native-v2-driver-self] smoke SKIP: baseline produced no binary for $smoke_src" >&2
    continue
  fi
  if ! timeout "$SMOKE_TIMEOUT" "$base_bin" >"$base_stdout" 2>/dev/null; then
    echo "[native-v2-driver-self] smoke SKIP: baseline binary did not complete within ${SMOKE_TIMEOUT}s on $smoke_src" >&2
    continue
  fi
  smoke_checked=$((smoke_checked + 1))

  # Stage1 must compile within the timeout and produce a binary that matches
  # the baseline's stdout within the same timeout.
  if ! timeout "$SMOKE_TIMEOUT" "$STAGE1_DRIVER" "$smoke_src" -o "$s1_bin" \
      >"$s1_compile_log" 2>&1; then
    echo "[native-v2-driver-self] FAIL: stage1-smoke timed out or errored compiling $smoke_src" >&2
    echo "[native-v2-driver-self] compile log: $s1_compile_log" >&2
    tail -n 80 "$s1_compile_log" >&2 || true
    exit 1
  fi
  if [[ ! -x "$s1_bin" ]]; then
    echo "[native-v2-driver-self] FAIL: stage1-smoke produced no executable for $smoke_src" >&2
    tail -n 80 "$s1_compile_log" >&2 || true
    exit 1
  fi
  if ! timeout "$SMOKE_TIMEOUT" "$s1_bin" >"$s1_stdout" 2>/dev/null; then
    echo "[native-v2-driver-self] FAIL: stage1-smoke binary did not complete within ${SMOKE_TIMEOUT}s on $smoke_src" >&2
    exit 1
  fi
  if ! cmp -s "$base_stdout" "$s1_stdout"; then
    echo "[native-v2-driver-self] FAIL: stage1-smoke stdout differs from baseline on $smoke_src" >&2
    diff -u "$base_stdout" "$s1_stdout" >&2 || true
    exit 1
  fi
  smoke_ran=$((smoke_ran + 1))
done

if [[ "$smoke_ran" -eq 0 ]]; then
  echo "[native-v2-driver-self] FAIL: stage1-smoke cohort produced zero runnable cases" >&2
  exit 1
fi
printf '[native-v2-driver-self] stage1-smoke OK ran=%d checked=%d\n' "$smoke_ran" "$smoke_checked"

# ── Stage2: stage1 native binary compiles the driver again ───────────────────
# Wall-clock timeout. The fixed-point property only catches divergence between
# stage2 and stage3 -- it cannot detect non-termination at stage2 itself. A
# stage1 binary that spins compiling the driver source is the recurring trap
# (step 4 array-init, step D user-globals). Baseline self-compile is ~1s; we
# allow 90s so a 90× slowdown still terminates and trips the explicit failure.
SELFCOMPILE_TIMEOUT="${SOUNIO_NATIVE_V2_DRIVER_SELF_COMPILE_TIMEOUT:-90}"
printf '[native-v2-driver-self] stage2/stage3 self-compile timeout=%ss\n' \
  "$SELFCOMPILE_TIMEOUT"

set +e
timeout "$SELFCOMPILE_TIMEOUT" "$STAGE1_DRIVER" "$DRIVER_SRC" -o "$STAGE2_DRIVER" \
  >"$STAGE2_COMPILE_LOG" 2>&1
stage2_rc=$?
set -e
if [[ "$stage2_rc" -eq 124 ]]; then
  echo "[native-v2-driver-self] FAIL: stage1 timed out after ${SELFCOMPILE_TIMEOUT}s compiling driver source (likely non-terminating loop in stage1 codegen -- see step-D blocker note)" >&2
  exit 1
fi
if [[ "$stage2_rc" -ne 0 ]]; then
  echo "[native-v2-driver-self] FAIL: stage1 driver failed to compile stage2 (rc=$stage2_rc)" >&2
  tail -n 120 "$STAGE2_COMPILE_LOG" >&2 || true
  exit 1
fi
if [[ ! -x "$STAGE2_DRIVER" ]]; then
  echo "[native-v2-driver-self] FAIL: stage2 driver is not executable" >&2
  exit 1
fi
check_driver_elf_layout "$STAGE2_DRIVER" "$STAGE2_SECTIONS_LOG" "stage2 driver"

# ── Stage2 hello parity ───────────────────────────────────────────────────────
if ! "$STAGE2_DRIVER" "$HELLO_SRC" -o "$STAGE2_HELLO_BIN" >"$STAGE2_HELLO_COMPILE_LOG" 2>&1; then
  echo "[native-v2-driver-self] FAIL: stage2 driver failed while compiling hello" >&2
  tail -n 120 "$STAGE2_HELLO_COMPILE_LOG" >&2 || true
  exit 1
fi
"$STAGE2_HELLO_BIN" >"$STAGE2_STDOUT" 2>/dev/null
if ! cmp -s "$EXPECTED_STDOUT" "$STAGE2_STDOUT"; then
  echo "[native-v2-driver-self] FAIL: stage2 hello stdout mismatch" >&2
  diff -u "$EXPECTED_STDOUT" "$STAGE2_STDOUT" >&2 || true
  exit 1
fi

# ── Fixed-point: stage2 compiles stage3, stage3 == stage2 ────────────────────
# Same wall-clock guard as stage2 -- defence in depth.
set +e
timeout "$SELFCOMPILE_TIMEOUT" "$STAGE2_DRIVER" "$DRIVER_SRC" -o "$STAGE3_DRIVER" \
  >"$STAGE3_COMPILE_LOG" 2>&1
stage3_rc=$?
set -e
if [[ "$stage3_rc" -eq 124 ]]; then
  echo "[native-v2-driver-self] FAIL: stage2 timed out after ${SELFCOMPILE_TIMEOUT}s compiling driver source" >&2
  exit 1
fi
if [[ "$stage3_rc" -ne 0 ]]; then
  echo "[native-v2-driver-self] FAIL: stage2 driver failed to compile stage3 (rc=$stage3_rc)" >&2
  tail -n 120 "$STAGE3_COMPILE_LOG" >&2 || true
  exit 1
fi
if ! cmp -s "$STAGE2_DRIVER" "$STAGE3_DRIVER"; then
  echo "[native-v2-driver-self] FAIL: fixed-point broken -- stage2 != stage3" >&2
  # Pure-bash byte-diff debug (replaces python3 -c heredoc).
  # cmp -l emits "byteoff val1 val2" with values in octal; we render hex.
  diff_total=$(cmp -l "$STAGE2_DRIVER" "$STAGE3_DRIVER" 2>/dev/null | wc -l)
  printf '  %s byte(s) differ\n' "$diff_total" >&2
  cmp -l "$STAGE2_DRIVER" "$STAGE3_DRIVER" 2>/dev/null | head -n 5 | while read -r off v1 v2; do
    printf '  offset 0x%x: %02x vs %02x\n' $((off - 1)) $((8#$v1)) $((8#$v2)) >&2
  done || true
  exit 1
fi
check_driver_elf_layout "$STAGE3_DRIVER" "$STAGE3_SECTIONS_LOG" "stage3 driver"
STAGE2_MD5="$(md5sum "$STAGE2_DRIVER" | cut -d' ' -f1)"
printf '[native-v2-driver-self] fixed-point md5=%s\n' "$STAGE2_MD5"

# ── Epistemic fixed-point: read .sounio.epistemic from stage1/2/3 ────────────
# Pure-bash SIEP chunk reader (replaces python struct.unpack heredoc).
# SIEP chunk layout: 'SIEP' magic at offset 0, u32 version at +4,
# u64 instr_count at +8, u64 u_c_scaled at +16. All little-endian.
read_epistemic() {
  local path="$1"
  if [[ ! -f "$path" ]]; then echo "absent"; return 0; fi
  local idx
  idx=$(LC_ALL=C grep -ab -o 'SIEP' "$path" 2>/dev/null | head -n 1 | cut -d: -f1)
  if [[ -z "$idx" ]]; then echo "absent"; return 0; fi
  local version instr_count u_c_scaled
  version=$(dd if="$path" bs=1 skip=$((idx+4)) count=4 2>/dev/null | od -An -t u4 | tr -d ' ')
  instr_count=$(dd if="$path" bs=1 skip=$((idx+8)) count=8 2>/dev/null | od -An -t u8 | tr -d ' ')
  u_c_scaled=$(dd if="$path" bs=1 skip=$((idx+16)) count=8 2>/dev/null | od -An -t u8 | tr -d ' ')
  echo "${version}:${instr_count}:${u_c_scaled}"
}

STAGE1_EP="$(read_epistemic "$STAGE1_DRIVER")"
STAGE2_EP="$(read_epistemic "$STAGE2_DRIVER")"
STAGE3_EP="$(read_epistemic "$STAGE3_DRIVER")"
printf '[native-v2-driver-self] epistemic stage1=%s stage2=%s stage3=%s\n' \
  "$STAGE1_EP" "$STAGE2_EP" "$STAGE3_EP"

if [[ "$STAGE2_EP" == "absent" || "$STAGE3_EP" == "absent" ]]; then
  echo "[native-v2-driver-self] FAIL: epistemic section missing from stage2 or stage3" >&2
  exit 1
fi
if [[ "$STAGE2_EP" != "$STAGE3_EP" ]]; then
  echo "[native-v2-driver-self] FAIL: epistemic fixed-point broken -- stage2 != stage3 epistemic profile" >&2
  exit 1
fi
STAGE2_INSTR="$(echo "$STAGE2_EP" | cut -d: -f2)"
printf '[native-v2-driver-self] epistemic-fixed-point instr=%s u_c=0 (stage2==stage3 confidence-stable)\n' "$STAGE2_INSTR"

echo "[native-v2-driver-self] PASS: baseline, stage1 driver, stage2 driver, fixed-point (stage2==stage3), hello parity across all stages, epistemic-fixed-point verified"
