#!/usr/bin/env bash
# Madaros gate — free functions literally named exp/cos must keep user bodies
# under multi-module native (not float-intrinsic hijack).
#
# Pre-fix (measured 2026-07-20 / #1287):
#   use lib::{exp}  → call returns true e^x (2718) instead of user body (2000)
#   use lib::{cos}  → call returns true cos  (540) instead of user body (3000)
#   use propagate::{exp} on Epistemic → SEGV rc=139
#   rename my_exp → user body OK
#
# Fix: native_v2_builtin_id_for_func_ref + compile_ir_function* only emit
# hand-coded float/string builtins for empty stubs (instr_count == 0).
#
# Also verifies bare print_f64 / sqrt intrinsics still work multi-mod.
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
unset SOUNIO_SOUC_ENGINE || true
SOUC=./bin/souc
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

engine_line="$($SOUC --version 2>&1 | head -1 || true)"
if echo "$engine_line" | grep -qi lean_single; then
  echo "FAIL: gate must run under default Madaros, not lean_single ($engine_line)"
  exit 1
fi
echo "engine: $engine_line"

fail=0
ROOT_TESTS=tests/compiler/exp_cos_import_collision

run_expect() {
  local name="$1"
  local src="$2"
  local expect_line="$3"
  local expect_rc="${4:-0}"
  if ! $SOUC compile "$src" -o "$OUT/$name.elf" >"$OUT/$name.compile" 2>&1; then
    echo "FAIL: $name compile"
    tail -40 "$OUT/$name.compile"
    fail=1
    return
  fi
  set +e
  "$OUT/$name.elf" >"$OUT/$name.out" 2>"$OUT/$name.err"
  local rc=$?
  set -e
  if [ "$rc" -ne "$expect_rc" ]; then
    echo "FAIL: $name rc=$rc (want $expect_rc)"
    cat "$OUT/$name.out" || true
    cat "$OUT/$name.err" || true
    fail=1
    return
  fi
  if ! grep -q "$expect_line" "$OUT/$name.out"; then
    echo "FAIL: $name missing sentinel $expect_line"
    cat "$OUT/$name.out" || true
    fail=1
    return
  fi
  echo "PASS: $name ($expect_line)"
}

echo "== multi-module user exp/cos keep bodies =="
run_expect multi_exp "$ROOT_TESTS/main_exp.sio" "USER_EXP_OK"
run_expect multi_cos "$ROOT_TESTS/main_cos.sio" "USER_COS_OK"
run_expect multi_rename "$ROOT_TESTS/main_rename.sio" "RENAME_OK"

echo "== same-file user exp/cos =="
run_expect samefile "$ROOT_TESTS/main_samefile.sio" "SAMEFILE_OK"

echo "== bare float intrinsics still work (print_f64 match; avoid flaky f64 compares) =="
# exp(1)≈2.718281 is the trustworthy bare intrinsic witness (exp(0) series is
# a pre-existing soft spot on main — not part of this collision fix).
cat >"$OUT/bare_exp.sio" <<'EOF'
fn main() -> i32 with IO {
    print_f64(exp(1.0))
    println("")
    return 0
}
EOF
cat >"$OUT/bare_sqrt.sio" <<'EOF'
fn main() -> i32 with IO {
    print_f64(sqrt(4.0))
    println("")
    return 0
}
EOF
cat >"$OUT/bare_print_f64.sio" <<'EOF'
fn main() -> i32 with IO {
    print_f64(3.5)
    println("PRINT_F64_OK")
    return 0
}
EOF
# Match printed magnitudes (print_f64 format: digit.fraction)
run_expect bare_exp "$OUT/bare_exp.sio" "2.718"
run_expect bare_sqrt "$OUT/bare_sqrt.sio" "2.000"
run_expect bare_print_f64 "$OUT/bare_print_f64.sio" "PRINT_F64_OK"

echo "== propagate::exp / ::cos multi-module (Epistemic; pre-fix SEGV) =="
run_expect prop_exp_cos "$ROOT_TESTS/main_propagate_exp.sio" "PROPAGATE_EXP_COS_OK"

echo
if [ $fail -eq 0 ]; then
  echo "MADAROS_EXP_COS_IMPORT_COLLISION_GATE_OK"
  exit 0
fi
echo "MADAROS_EXP_COS_IMPORT_COLLISION_GATE_FAIL"
exit 1
