#!/usr/bin/env bash
# E2E gate for the public Sounio developer loop.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

TMP_HOME="$(mktemp -d /tmp/sounio-real-runner-home-XXXXXX)"
TMP_ELF="$(mktemp /tmp/sounio-real-runner-XXXXXX.elf)"
TMP_RAW="$(mktemp /tmp/sounio-real-runner-raw-XXXXXX.elf)"
TMP_MADAROS_SRC="$(mktemp /tmp/sounio-madaros-XXXXXX.sio)"
TMP_MADAROS_ELF="$(mktemp /tmp/sounio-madaros-XXXXXX.elf)"
trap 'rm -rf "$TMP_HOME" "$TMP_ELF" "$TMP_RAW" "$TMP_MADAROS_SRC" "$TMP_MADAROS_ELF"' EXIT

fail() {
  echo "[real-runner] FAIL: $*" >&2
  exit 1
}

assert_contains() {
  local haystack="$1"
  local needle="$2"
  local label="$3"
  if [[ "$haystack" != *"$needle"* ]]; then
    echo "[real-runner] output for $label:" >&2
    printf '%s\n' "$haystack" >&2
    fail "expected '$needle' in $label"
  fi
}

echo "[real-runner] 1/7 souc --version"
VERSION_OUT="$(./bin/souc --version 2>&1)"
assert_contains "$VERSION_OUT" "Madar" "souc --version"
assert_contains "$VERSION_OUT" "Sounio self-hosted compiler" "souc --version"

echo "[real-runner] 2/7 souc info"
INFO_OUT="$(./bin/souc info 2>&1)"
assert_contains "$INFO_OUT" "wrapper:" "souc info"
assert_contains "$INFO_OUT" "raw_elf:" "souc info"
assert_contains "$INFO_OUT" "identity:" "souc info"

echo "[real-runner] 3/7 souc check"
./bin/souc check examples/hello.sio >/tmp/sounio-real-runner-check.log 2>&1 || {
  cat /tmp/sounio-real-runner-check.log >&2
  fail "souc check examples/hello.sio"
}

echo "[real-runner] 4/7 souc run stdout"
RUN_OUT="$(./bin/souc run examples/native/hello.sio 2>&1)"
assert_contains "$RUN_OUT" "Hello from self-hosted Sounio!" "souc run examples/native/hello.sio"
assert_contains "$RUN_OUT" "42" "souc run examples/native/hello.sio"

echo "[real-runner] 5/7 souc compile + execute ELF"
./bin/souc compile examples/native/hof_closure_literal.sio -o "$TMP_ELF" >/tmp/sounio-real-runner-compile.log 2>&1 || {
  cat /tmp/sounio-real-runner-compile.log >&2
  fail "souc compile examples/native/hof_closure_literal.sio"
}
[[ -x "$TMP_ELF" ]] || fail "compiled ELF is not executable: $TMP_ELF"
ELF_OUT="$("$TMP_ELF" 2>&1)"
assert_contains "$ELF_OUT" "42" "compiled ELF"

echo "[real-runner] 6/7 raw positional passthrough"
./bin/souc examples/hello.sio "$TMP_RAW" >/tmp/sounio-real-runner-raw.log 2>&1 || {
  cat /tmp/sounio-real-runner-raw.log >&2
  fail "souc raw positional passthrough"
}
[[ -s "$TMP_RAW" ]] || fail "raw passthrough produced no ELF"

echo "[real-runner] 7/7 REPL declaration + expression"
REPL_OUT="$(printf '%s\n' \
  'fn dbl(x: i64) -> i64 { x + x }' \
  'dbl(21)' \
  ':quit' \
  | env HOME="$TMP_HOME" timeout 120 ./bin/souc repl 2>&1)"
assert_contains "$REPL_OUT" "(defined)" "souc repl"
assert_contains "$REPL_OUT" "42" "souc repl"

echo "[real-runner] Madaros identity + native-v2 compile"
bash scripts/gates/g6_madaros_identity.sh

cat >"$TMP_MADAROS_SRC" <<'SRC'
fn main() -> i64 {
    42
}
SRC

./bin/madaros compile "$TMP_MADAROS_SRC" -o "$TMP_MADAROS_ELF" >/tmp/sounio-real-runner-madaros.log 2>&1 || {
  cat /tmp/sounio-real-runner-madaros.log >&2
  fail "madaros compile"
}
chmod +x "$TMP_MADAROS_ELF"
set +e
"$TMP_MADAROS_ELF"
MADAROS_RC=$?
set -e
[[ "$MADAROS_RC" -eq 42 ]] || fail "expected Madaros-compiled ELF exit 42, got $MADAROS_RC"

echo "REAL_LANGUAGE_RUNNER_PASS"
