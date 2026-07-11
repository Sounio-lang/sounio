#!/usr/bin/env bash
# Gate the public Madaros CLI and its current Stage1 compiler contract.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MADAROS="${MADAROS_BIN:-$ROOT_DIR/bin/madaros}"
GATE_STDLIB="${SOUNIO_MADAROS_FULL_GATE_STDLIB_PATH:-$ROOT_DIR/stdlib}"
is_elf() {
  [[ -n "${1:-}" && -x "$1" && "$(head -c 2 "$1" 2>/dev/null)" != "#!" ]]
}

receipt_ok() {
  local elf="$1"
  local receipt="${2:-$elf.gate-receipt}"
  [[ -f "$receipt" ]] || return 1
  local want
  want="$(sha256sum "$elf" 2>/dev/null | cut -d' ' -f1)"
  [[ -n "$want" ]] || return 1
  grep -Fq "$want" "$receipt" || return 1
  grep -Fxq "smt_skip=0" "$receipt" || return 1
  grep -Fxq "eisa_native_conformance=39/39" "$receipt" || return 1
  grep -Fxq "eisa_native_tamper=pass" "$receipt" || return 1
  grep -Fxq "eisa_native_anti_vacuity=pass" "$receipt" || return 1
  grep -Fxq "eisa_native_lowering=full_modular_ir_no_fallback" "$receipt" || return 1
  grep -Fxq "eisa_native_compact_opt_in=fail_closed_no_fallback" "$receipt" || return 1
  grep -Fxq "eisa_native_compact_overcapacity=130_functions_fail_closed_no_elf" "$receipt" || return 1
}

resolve_raw_madaros() {
  local cand
  for cand in "${MADAROS_RAW_BIN:-}" "${SOUNIO_MADAROS_BIN:-}"; do
    if is_elf "$cand"; then
      echo "$cand"
      return 0
    fi
  done

  local artifact="$ROOT_DIR/artifacts/self-hosted/madaros"
  if is_elf "$artifact"; then
    echo "$artifact"
    return 0
  fi

  for cand in "$ROOT_DIR/bin/madaros-relocgate" "$ROOT_DIR/bin/madaros-linux-x86_64"; do
    if is_elf "$cand"; then
      echo "$cand"
      return 0
    fi
  done
  return 1
}

RAW_MADAROS="$(resolve_raw_madaros)" || {
  echo "[madaros-full] FAIL: no Madaros raw ELF found" >&2
  exit 1
}
WORK="${SOUNIO_MADAROS_FULL_GATE_DIR:-$(mktemp -d /tmp/sounio-madaros-full.XXXXXX)}"
KEEP_WORK="${SOUNIO_MADAROS_FULL_GATE_KEEP:-0}"
SMT_GATE_RAN=0
SMT_GATE_PASS=0
EISA_NATIVE_GATE_RAN=0
EISA_NATIVE_GATE_PASS=0

if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

fail() {
  echo "[madaros-full] FAIL: $*" >&2
  exit 1
}

pass() {
  echo "[madaros-full] PASS: $*"
}

expect_exit() {
  local expected="$1"
  shift
  set +e
  "$@"
  local rc=$?
  set -e
  if [[ "$rc" != "$expected" ]]; then
    fail "expected exit $expected, got $rc: $*"
  fi
}

expect_log_contains() {
  local needle="$1"
  local file="$2"
  if ! grep -Fq "$needle" "$file"; then
    echo "[madaros-full] log tail for $file:" >&2
    tail -n 40 "$file" >&2 || true
    fail "missing log marker: $needle"
  fi
}

expect_elf_runs() {
  local expected="$1"
  local elf="$2"
  [[ -s "$elf" ]] || fail "missing ELF: $elf"
  file "$elf" | grep -Fq "ELF 64-bit" || fail "not an ELF64 executable: $elf"
  chmod +x "$elf"
  expect_exit "$expected" "$elf"
}

run_imported_smt_gate() {
  if [[ "${SOUNIO_MADAROS_FULL_GATE_SKIP_SMT:-0}" == "1" ]]; then
    echo "[madaros-full] SKIP: imported-SMT solver gate (SOUNIO_MADAROS_FULL_GATE_SKIP_SMT=1)"
    fail "imported-SMT solver gate skipped; this gate is non-green without SMT 6/6"
  fi

  mapfile -t smt_tests < <(find "$ROOT_DIR/tests/stdlib/theorem" -maxdepth 1 -name 'test_smt_*.sio' -print | sort)
  if [[ "${#smt_tests[@]}" -ne 6 ]]; then
    fail "expected exactly 6 imported-SMT fixtures, found ${#smt_tests[@]}"
  fi

  SMT_GATE_RAN=1
  local pass_count=0
  local test_path log rc
  for test_path in "${smt_tests[@]}"; do
    log="$WORK/smt_$(basename "$test_path").log"
    set +e
    env MADAROS_RAW_BIN="$RAW_MADAROS" timeout 120 "$MADAROS" run "$test_path" >"$log" 2>&1
    rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
      echo "[madaros-full] imported-SMT failure: $(basename "$test_path") rc=$rc" >&2
      tail -n 40 "$log" >&2 || true
      fail "imported-SMT solver gate failed"
    fi
    pass_count=$((pass_count + 1))
  done

  if [[ "$pass_count" -ne 6 ]]; then
    fail "imported-SMT solver gate incomplete: $pass_count/6"
  fi
  SMT_GATE_PASS=1
  pass "imported-SMT solver gate (6/6)"
}

run_eisa_native_gate() {
  EISA_NATIVE_GATE_RAN=1
  env MADAROS_RAW_BIN="$RAW_MADAROS" \
    EISA_MADAROS_NATIVE_WORK_DIR="$WORK/eisa-native" \
    EISA_MADAROS_NATIVE_RECEIPT_PATH="$WORK/eisa-native.receipt" \
    bash scripts/ci/eisa_madaros_native_conformance_gate.sh
  grep -Fxq 'receipts=39/39' "$WORK/eisa-native.receipt" \
    || fail "EISA native conformance receipt is incomplete"
  grep -Fxq 'tamper=pass' "$WORK/eisa-native.receipt" \
    || fail "EISA native tamper receipt is incomplete"
  grep -Fxq 'anti_vacuity=pass' "$WORK/eisa-native.receipt" \
    || fail "EISA native anti-vacuity receipt is incomplete"
  grep -Fxq 'native_lowering=full_modular_ir_no_fallback' "$WORK/eisa-native.receipt" \
    || fail "EISA native lowering receipt permits fallback"
  grep -Fxq 'compact_opt_in=fail_closed_no_fallback' "$WORK/eisa-native.receipt" \
    || fail "EISA compact opt-in negative receipt is incomplete"
  grep -Fxq 'compact_overcapacity_witness=130_functions_fail_closed_no_elf' "$WORK/eisa-native.receipt" \
    || fail "compact over-capacity witness receipt is incomplete"
  EISA_NATIVE_GATE_PASS=1
  pass "EISA METRON/VM to Madaros native conformance (39/39)"
}

write_gate_receipt() {
  if [[ "$SMT_GATE_RAN" != "1" || "$SMT_GATE_PASS" != "1" || \
        "$EISA_NATIVE_GATE_RAN" != "1" || "$EISA_NATIVE_GATE_PASS" != "1" ]]; then
    echo "[madaros-full] note: no gate receipt written (SMT or EISA native gate did not run/pass)"
    return
  fi

  local canonical="$ROOT_DIR/artifacts/self-hosted/madaros"
  local receipt="${SOUNIO_MADAROS_FULL_GATE_RECEIPT_PATH:-}"
  if [[ -z "$receipt" ]]; then
    if [[ "$RAW_MADAROS" != "$canonical" ]]; then
      echo "[madaros-full] note: no gate receipt written for non-canonical raw ELF: $RAW_MADAROS"
      return
    fi
    receipt="$canonical.gate-receipt"
  fi

  local sha now tmp
  sha="$(sha256sum "$RAW_MADAROS" | cut -d' ' -f1)"
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  tmp="$receipt.tmp.$$"
  mkdir -p "$(dirname "$receipt")"
  {
    echo "madaros_full_gate_receipt_v1"
    echo "artifact=$RAW_MADAROS"
    echo "sha256=$sha"
    echo "created_utc=$now"
    echo "gate=madaros_full_gate.sh"
    echo "smt_tests=6/6"
    echo "smt_skip=0"
    echo "eisa_native_conformance=39/39"
    echo "eisa_native_tamper=pass"
    echo "eisa_native_anti_vacuity=pass"
    echo "eisa_native_lowering=full_modular_ir_no_fallback"
    echo "eisa_native_compact_opt_in=fail_closed_no_fallback"
    echo "eisa_native_compact_overcapacity=130_functions_fail_closed_no_elf"
  } >"$tmp"
  mv "$tmp" "$receipt"
  receipt_ok "$RAW_MADAROS" "$receipt" || fail "gate receipt validation failed after write"
  pass "gate receipt: $receipt"
}

mkdir -p "$WORK"
printf '[madaros-full] madaros=%s\n' "$MADAROS"
printf '[madaros-full] raw_elf=%s\n' "$RAW_MADAROS"
printf '[madaros-full] stdlib=%s\n' "$GATE_STDLIB"
printf '[madaros-full] work=%s\n' "$WORK"

madaros() {
  env MADAROS_RAW_BIN="$RAW_MADAROS" SOUNIO_STDLIB_PATH="$GATE_STDLIB" "$MADAROS" "$@"
}

madaros --version | grep -Fq "Madares v0.80.0" || fail "version banner missing"
pass "version"

: > "$WORK/empty.sio"
printf 'fn main() -> i64 { 0 }\n' > "$WORK/nocallargs.sio"
printf 'fn f() -> i64 { 1 }\nfn main() -> i64 { f() }\n' > "$WORK/localcall.sio"
printf 'fn add(a: i64, b: i64) -> i64 { a + b }\nfn main() -> i64 { add(1, 2) }\n' > "$WORK/callargs.sio"
printf 'fn main() -> i64 { 0 }\n' > "$WORK/run0.sio"

for src in "$WORK/empty.sio" "$WORK/nocallargs.sio" "$WORK/localcall.sio" "$WORK/callargs.sio"; do
  madaros check "$src" >"$WORK/check.log" 2>&1
  expect_log_contains "check: OK" "$WORK/check.log"
done
pass "public check CLI"

[[ -x "$RAW_MADAROS" ]] || fail "missing raw Madaros ELF for raw check gate: $RAW_MADAROS"
env SOUNIO_STDLIB_PATH="$GATE_STDLIB" "$RAW_MADAROS" --check "$WORK/empty.sio" >"$WORK/raw_empty.log" 2>&1
expect_log_contains "check: OK" "$WORK/raw_empty.log"
expect_exit 1 env SOUNIO_STDLIB_PATH="$GATE_STDLIB" "$RAW_MADAROS" --check "$WORK/missing.sio" >"$WORK/raw_missing.log" 2>&1
expect_log_contains "could not read input file" "$WORK/raw_missing.log"
expect_exit 1 env SOUNIO_STDLIB_PATH="$GATE_STDLIB" "$RAW_MADAROS" --check /tmp >"$WORK/raw_tmp_dir.log" 2>&1
expect_log_contains "could not read input file" "$WORK/raw_tmp_dir.log"
pass "raw check CLI"

madaros check tests/multimodule/visibility_struct_pub_main.sio >"$WORK/visibility_struct_pub.log" 2>&1
expect_log_contains "check: OK" "$WORK/visibility_struct_pub.log"
madaros check tests/multimodule/visibility_enum_pub_main.sio >"$WORK/visibility_enum_pub.log" 2>&1
expect_log_contains "check: OK" "$WORK/visibility_enum_pub.log"
madaros check tests/multimodule/visibility_same_module_ok.sio >"$WORK/visibility_same_module.log" 2>&1
expect_log_contains "check: OK" "$WORK/visibility_same_module.log"

expect_exit 1 madaros check tests/multimodule/visibility_struct_private_main.sio >"$WORK/visibility_struct_private.log" 2>&1
expect_log_contains "error[E176" "$WORK/visibility_struct_private.log"
expect_exit 1 madaros check tests/multimodule/visibility_fn_private_main.sio >"$WORK/visibility_fn_private.log" 2>&1
expect_log_contains "error[E175" "$WORK/visibility_fn_private.log"
expect_exit 1 madaros check tests/multimodule/visibility_enum_private_main.sio >"$WORK/visibility_enum_private.log" 2>&1
expect_log_contains "error[E177" "$WORK/visibility_enum_private.log"
pass "multimodule visibility diagnostics"

expect_exit 2 madaros check "$WORK/missing.sio" >"$WORK/missing.log" 2>&1
expect_log_contains "requires <source.sio> or a project with sounio.toml" "$WORK/missing.log"
expect_exit 2 madaros check /tmp >"$WORK/wrapper_tmp_dir.log" 2>&1
expect_log_contains "requires <source.sio> or a project with sounio.toml" "$WORK/wrapper_tmp_dir.log"
pass "missing input diagnostic"

madaros build "$WORK/run0.sio" -o "$WORK/run0.elf" >"$WORK/build.log" 2>&1
expect_log_contains "Written to $WORK/run0.elf" "$WORK/build.log"
expect_elf_runs 0 "$WORK/run0.elf"
pass "source build to native ELF"

expect_exit 0 madaros run "$WORK/run0.sio" >"$WORK/run.log" 2>&1
pass "source run"

madaros --native-v2-emit-scalar 42 "$WORK/scalar42.elf" >"$WORK/scalar42.log" 2>&1
expect_elf_runs 42 "$WORK/scalar42.elf"
madaros --native-v2-emit-call "$WORK/call.elf" >"$WORK/call.log" 2>&1
expect_elf_runs 6 "$WORK/call.elf"
madaros --native-v2-emit-call5 "$WORK/call5.elf" >"$WORK/call5.log" 2>&1
expect_elf_runs 31 "$WORK/call5.elf"
madaros --native-v2-emit-call6 "$WORK/call6.elf" >"$WORK/call6.log" 2>&1
expect_elf_runs 63 "$WORK/call6.elf"
madaros --native-v2-emit-sret "$WORK/sret.elf" >"$WORK/sret.log" 2>&1
expect_elf_runs 14 "$WORK/sret.elf"
pass "native-v2 ABI/backend witnesses"

madaros pkg self-test >"$WORK/pkg_self_test.log" 2>&1
expect_log_contains "SPM self-tests: ALL PASSED" "$WORK/pkg_self_test.log"
pass "package manager self-test"

bash scripts/ci/madaros_read_env_absence_gate.sh
pass "source-fresh missing-env merged checker"

bash scripts/dev/madaros_v2_e1_enir_shadow_gate.sh
pass "Madaros v2 E1 compiler-owned ENIR shadow"

bash scripts/dev/madaros_v2_e2_enir_lowering_gate.sh
pass "Madaros v2 E2A v0 straight-line Source-to-ENIR lowering"

bash scripts/dev/madaros_v2_e2b_enir_cfg_gate.sh
pass "Madaros v2 E2B v1 finite-CFG Source-to-ENIR lowering"

bash scripts/dev/madaros_v2_e2c_enir_fuel_blockargs_gate.sh
pass "Madaros v2 E2C fuel and loop-carried block-argument lowering"

bash scripts/dev/madaros_v2_e2d_enir_rump_dd_gate.sh
pass "Madaros v2 E2D v1 Rump DD64 multi-observation lowering"

bash scripts/dev/madaros_v2_e2e_enir_qd128_gate.sh
pass "Madaros v2 E2E v2 qd128 arithmetic Source-to-ENIR lowering"

bash scripts/dev/madaros_v2_e2f_enir_rump_qd_gate.sh
pass "Madaros v2 E2F v2 Rump qd128 Source-to-ENIR lowering"

bash scripts/dev/madaros_v2_e2g_enir_fuel_control_frail_gate.sh
pass "Madaros v2 E2G v2 qd128 fuel, control, and frailty lowering"

run_eisa_native_gate
run_imported_smt_gate
write_gate_receipt

echo "[madaros-full] PASS: public CLI, source ELF path, ABI witnesses, visibility, and pkg self-test"
