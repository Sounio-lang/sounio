#!/usr/bin/env bash
# E2E gate for Sounio project mode: sounio.toml -> import graph -> ELF.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

fail() {
  echo "[project-spine] FAIL: $*" >&2
  exit 1
}

assert_contains() {
  local haystack="$1"
  local needle="$2"
  local label="$3"
  if [[ "$haystack" != *"$needle"* ]]; then
    echo "[project-spine] output for $label:" >&2
    printf '%s\n' "$haystack" >&2
    fail "expected '$needle' in $label"
  fi
}

echo "[project-spine] 1/8 public runner baseline"
bash scripts/ci/real_language_runner_gate.sh

echo "[project-spine] 2/8 checked example project"
./bin/souc check examples/projects/hello_pkg
HELLO_OUT="$(./bin/souc run examples/projects/hello_pkg 2>&1)"
assert_contains "$HELLO_OUT" "42" "souc run examples/projects/hello_pkg"
rm -f examples/projects/hello_pkg/target/hello_pkg
./bin/souc build examples/projects/hello_pkg
[[ -x examples/projects/hello_pkg/target/hello_pkg ]] || fail "project build did not create executable target"
BUILT_OUT="$(examples/projects/hello_pkg/target/hello_pkg 2>&1)"
assert_contains "$BUILT_OUT" "42" "built hello_pkg ELF"
rm -f examples/projects/hello_pkg/target/hello_pkg

echo "[project-spine] 3/8 souc init project loop"
TMP_DIR="$(mktemp -d /tmp/sounio-project-spine.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT
(
  cd "$TMP_DIR"
  "$ROOT_DIR/bin/souc" init generated_pkg
  cd generated_pkg
  "$ROOT_DIR/bin/souc" check
  GEN_RUN_OUT="$("$ROOT_DIR/bin/souc" run 2>&1)"
  assert_contains "$GEN_RUN_OUT" "42" "generated project run"
  "$ROOT_DIR/bin/souc" build
  [[ -x target/generated_pkg ]] || fail "generated project build did not create target/generated_pkg"
  GEN_ELF_OUT="$(target/generated_pkg 2>&1)"
  assert_contains "$GEN_ELF_OUT" "42" "generated project built ELF"
)

echo "[project-spine] 4/8 missing import negative"
NEG_LOG="$TMP_DIR/missing_import.log"
set +e
./bin/souc check examples/projects/bad_missing_import >"$NEG_LOG" 2>&1
NEG_RC=$?
set -e
if [[ "$NEG_RC" -eq 0 ]]; then
  cat "$NEG_LOG" >&2
  fail "missing import project unexpectedly passed"
fi
if ! grep -qE 'unreadable import|typecheck: failed|unknown identifier' "$NEG_LOG"; then
  cat "$NEG_LOG" >&2
  fail "missing import project did not emit an import/typecheck diagnostic"
fi

echo "[project-spine] 5/8 package import manifest gate"
bash scripts/ci/package_import_science_gate.sh

echo "[project-spine] 6/8 thin multimodule runtime"
THIN_ELF="$TMP_DIR/thin_single.elf"
./bin/souc compile tests/multimodule/thin_single_main.sio -o "$THIN_ELF"
set +e
"$THIN_ELF"
THIN_RC=$?
set -e
[[ "$THIN_RC" -eq 7 ]] || fail "thin multimodule ELF expected exit 7, got $THIN_RC"

echo "[project-spine] 7/8 Madaros identity and single-file native-v2"
bash scripts/gates/g6_madaros_identity.sh
MADAROS_SRC="$TMP_DIR/madaros_single.sio"
MADAROS_ELF="$TMP_DIR/madaros_single.elf"
printf 'fn main() -> i64 { 42 }\n' >"$MADAROS_SRC"
./bin/madaros compile "$MADAROS_SRC" -o "$MADAROS_ELF"
set +e
"$MADAROS_ELF"
MADAROS_RC=$?
set -e
[[ "$MADAROS_RC" -eq 42 ]] || fail "Madaros single-file ELF expected exit 42, got $MADAROS_RC"

echo "[project-spine] 8/8 imported modular IR lowering gate"
bash scripts/ci/native_v2_imported_body_lowering_gate.sh

echo "PROJECT_SPINE_PASS"
