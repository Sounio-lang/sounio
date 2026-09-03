#!/usr/bin/env bash
# Gate the public Madaros CLI and its current Stage1 compiler contract.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MADAROS="${MADAROS_BIN:-$ROOT_DIR/bin/madaros}"
RAW_MADAROS="${MADAROS_RAW_BIN:-}"
if [[ -z "$RAW_MADAROS" ]]; then
  if [[ -x "$ROOT_DIR/artifacts/self-hosted/madaros" ]]; then
    RAW_MADAROS="$ROOT_DIR/artifacts/self-hosted/madaros"
  else
    RAW_MADAROS="$ROOT_DIR/bin/madaros-linux-x86_64"
  fi
fi
WORK="${SOUNIO_MADAROS_FULL_GATE_DIR:-$(mktemp -d /tmp/sounio-madaros-full.XXXXXX)}"
KEEP_WORK="${SOUNIO_MADAROS_FULL_GATE_KEEP:-0}"

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
  # Check the ELF64 magic directly instead of shelling out to file(1).
  # SLURM compute nodes do not ship file(1), and this gate is now run there via
  # scripts/dev/souc-build-remote.sh. Bytes 0-3 are \x7fELF and byte 4 is 2 for
  # ELFCLASS64.
  local magic
  magic="$(od -An -tx1 -N5 "$elf" 2>/dev/null | tr -d ' \n')"
  [[ "$magic" == "7f454c4602" ]] || fail "not an ELF64 executable: $elf (magic=$magic)"
  chmod +x "$elf"
  expect_exit "$expected" "$elf"
}

mkdir -p "$WORK"
printf '[madaros-full] madaros=%s\n' "$MADAROS"
printf '[madaros-full] work=%s\n' "$WORK"

"$MADAROS" --version | grep -Fq "Madaros v0.80.0" || fail "version banner missing"
pass "version"

: > "$WORK/empty.sio"
printf 'fn main() -> i64 { 0 }\n' > "$WORK/nocallargs.sio"
printf 'fn f() -> i64 { 1 }\nfn main() -> i64 { f() }\n' > "$WORK/localcall.sio"
printf 'fn add(a: i64, b: i64) -> i64 { a + b }\nfn main() -> i64 { add(1, 2) }\n' > "$WORK/callargs.sio"
printf 'fn main() -> i64 { 0 }\n' > "$WORK/run0.sio"

for src in "$WORK/empty.sio" "$WORK/nocallargs.sio" "$WORK/localcall.sio" "$WORK/callargs.sio"; do
  "$MADAROS" check "$src" >"$WORK/check.log" 2>&1
  expect_log_contains "check: OK" "$WORK/check.log"
done
pass "public check CLI"

[[ -x "$RAW_MADAROS" ]] || fail "missing raw Madaros ELF for raw check gate: $RAW_MADAROS"
"$RAW_MADAROS" --check "$WORK/empty.sio" >"$WORK/raw_empty.log" 2>&1
expect_log_contains "check: OK" "$WORK/raw_empty.log"
expect_exit 1 "$RAW_MADAROS" --check "$WORK/missing.sio" >"$WORK/raw_missing.log" 2>&1
expect_log_contains "could not read input file" "$WORK/raw_missing.log"
expect_exit 1 "$RAW_MADAROS" --check /tmp >"$WORK/raw_tmp_dir.log" 2>&1
expect_log_contains "could not read input file" "$WORK/raw_tmp_dir.log"
pass "raw check CLI"

"$MADAROS" check tests/multimodule/visibility_struct_pub_main.sio >"$WORK/visibility_struct_pub.log" 2>&1
expect_log_contains "check: OK" "$WORK/visibility_struct_pub.log"
"$MADAROS" check tests/multimodule/visibility_enum_pub_main.sio >"$WORK/visibility_enum_pub.log" 2>&1
expect_log_contains "check: OK" "$WORK/visibility_enum_pub.log"
"$MADAROS" check tests/multimodule/visibility_same_module_ok.sio >"$WORK/visibility_same_module.log" 2>&1
expect_log_contains "check: OK" "$WORK/visibility_same_module.log"

expect_exit 1 "$MADAROS" check tests/multimodule/visibility_struct_private_main.sio >"$WORK/visibility_struct_private.log" 2>&1
expect_log_contains "error[E176" "$WORK/visibility_struct_private.log"
expect_exit 1 "$MADAROS" check tests/multimodule/visibility_fn_private_main.sio >"$WORK/visibility_fn_private.log" 2>&1
expect_log_contains "error[E175" "$WORK/visibility_fn_private.log"
expect_exit 1 "$MADAROS" check tests/multimodule/visibility_enum_private_main.sio >"$WORK/visibility_enum_private.log" 2>&1
expect_log_contains "error[E177" "$WORK/visibility_enum_private.log"
pass "multimodule visibility diagnostics"

expect_exit 2 "$MADAROS" check "$WORK/missing.sio" >"$WORK/missing.log" 2>&1
expect_log_contains "madaros check requires <source.sio> or a project with sounio.toml" "$WORK/missing.log"
expect_exit 2 env MADAROS_RAW_BIN="$RAW_MADAROS" "$MADAROS" check /tmp >"$WORK/wrapper_tmp_dir.log" 2>&1
expect_log_contains "madaros check requires <source.sio> or a project with sounio.toml" "$WORK/wrapper_tmp_dir.log"
pass "missing input diagnostic"

"$MADAROS" build "$WORK/run0.sio" -o "$WORK/run0.elf" >"$WORK/build.log" 2>&1
expect_log_contains "Written to $WORK/run0.elf" "$WORK/build.log"
expect_elf_runs 0 "$WORK/run0.elf"
pass "source build to native ELF"

expect_exit 0 "$MADAROS" run "$WORK/run0.sio" >"$WORK/run.log" 2>&1
pass "source run"

"$MADAROS" --native-v2-emit-scalar 42 "$WORK/scalar42.elf" >"$WORK/scalar42.log" 2>&1
expect_elf_runs 42 "$WORK/scalar42.elf"
"$MADAROS" --native-v2-emit-call "$WORK/call.elf" >"$WORK/call.log" 2>&1
expect_elf_runs 6 "$WORK/call.elf"
"$MADAROS" --native-v2-emit-call5 "$WORK/call5.elf" >"$WORK/call5.log" 2>&1
expect_elf_runs 31 "$WORK/call5.elf"
"$MADAROS" --native-v2-emit-call6 "$WORK/call6.elf" >"$WORK/call6.log" 2>&1
expect_elf_runs 63 "$WORK/call6.elf"
"$MADAROS" --native-v2-emit-sret "$WORK/sret.elf" >"$WORK/sret.log" 2>&1
expect_elf_runs 14 "$WORK/sret.elf"
pass "native-v2 ABI/backend witnesses"

"$MADAROS" pkg self-test >"$WORK/pkg_self_test.log" 2>&1
expect_log_contains "SPM self-tests: ALL PASSED" "$WORK/pkg_self_test.log"
pass "package manager self-test"

bash "$ROOT_DIR/scripts/ci/madaros_imported_deref_f64_array_gate.sh"
pass "current-source imported dereferenced f64-array lowering"

echo "[madaros-full] PASS: public CLI, source ELF path, ABI witnesses, visibility, pkg self-test, and current-source dereferenced f64-array lowering"
