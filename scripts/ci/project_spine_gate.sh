#!/usr/bin/env bash
# E2E gate for Sounio project mode: sounio.toml -> local imports -> ELF.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

TMP_DIR="$(mktemp -d /tmp/sounio-project-spine-XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

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

echo "[project-spine] 1/7 public runner baseline"
bash scripts/ci/real_language_runner_gate.sh

echo "[project-spine] 2/7 souc check/run/build on project dir"
./bin/souc check examples/projects/hello_pkg
HELLO_RUN_OUT="$(./bin/souc run examples/projects/hello_pkg 2>&1)"
assert_contains "$HELLO_RUN_OUT" "42" "souc run examples/projects/hello_pkg"
rm -f examples/projects/hello_pkg/target/hello_pkg
./bin/souc build examples/projects/hello_pkg
[[ -x examples/projects/hello_pkg/target/hello_pkg ]] || fail "souc build did not create target/hello_pkg"
HELLO_ELF_OUT="$(examples/projects/hello_pkg/target/hello_pkg 2>&1)"
assert_contains "$HELLO_ELF_OUT" "42" "built hello_pkg ELF"
rm -f examples/projects/hello_pkg/target/hello_pkg

echo "[project-spine] 3/7 souc init project loop"
(
  cd "$TMP_DIR"
  "$ROOT_DIR/bin/souc" init generated_pkg
  cd generated_pkg
  "$ROOT_DIR/bin/souc" check
  GENERATED_RUN_OUT="$("$ROOT_DIR/bin/souc" run 2>&1)"
  assert_contains "$GENERATED_RUN_OUT" "42" "generated project run"
  "$ROOT_DIR/bin/souc" build
  [[ -x target/generated_pkg ]] || fail "generated project build did not create target/generated_pkg"
  GENERATED_ELF_OUT="$(target/generated_pkg 2>&1)"
  assert_contains "$GENERATED_ELF_OUT" "42" "generated project built ELF"
)

echo "[project-spine] 4/7 missing entry negative"
NEG_LOG="$TMP_DIR/missing_entry.log"
set +e
./bin/souc check examples/projects/bad_missing_import >"$NEG_LOG" 2>&1
NEG_RC=$?
set -e
if [[ "$NEG_RC" -eq 0 ]]; then
  cat "$NEG_LOG" >&2
  fail "missing entry project unexpectedly passed"
fi
if ! grep -q "project entry not found:" "$NEG_LOG"; then
  cat "$NEG_LOG" >&2
  fail "missing entry project did not emit a project-entry diagnostic"
fi

echo "[project-spine] 5/7 Madaros identity"
bash scripts/gates/g6_madaros_identity.sh

echo "[project-spine] 6/7 Madaros check/run/build on project dir"
./bin/madaros check examples/projects/hello_pkg
MADAROS_RUN_OUT="$(./bin/madaros run examples/projects/hello_pkg 2>&1)"
assert_contains "$MADAROS_RUN_OUT" "42" "madaros run examples/projects/hello_pkg"
rm -f examples/projects/hello_pkg/target/hello_pkg
./bin/madaros build examples/projects/hello_pkg
[[ -x examples/projects/hello_pkg/target/hello_pkg ]] || fail "madaros build did not create target/hello_pkg"
MADAROS_ELF_OUT="$(examples/projects/hello_pkg/target/hello_pkg 2>&1)"
assert_contains "$MADAROS_ELF_OUT" "42" "madaros built hello_pkg ELF"
rm -f examples/projects/hello_pkg/target/hello_pkg

echo "[project-spine] 7/7 cwd project resolution"
(
  cd examples/projects/hello_pkg
  "$ROOT_DIR/bin/souc" check
  CWD_RUN_OUT="$("$ROOT_DIR/bin/souc" run 2>&1)"
  assert_contains "$CWD_RUN_OUT" "42" "souc run from project cwd"
)

echo "PROJECT_SPINE_PASS"
