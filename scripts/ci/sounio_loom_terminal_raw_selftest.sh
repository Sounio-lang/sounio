#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
LOOM="env SOUNIO_COORD_RUNTIME_MODE=local $ROOT_DIR/bin/sounio-loom"

fail() {
  printf 'sounio-loom-terminal-raw-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

command -v script >/dev/null 2>&1 || fail "util-linux script command is required"

probe_byte() {
  local label="$1" input="$2" expected="$3" output
  output="$(
    { sleep 0.2; printf '%b' "$input"; } |
      script --quiet --return --echo never \
        --command "stty rows 31 cols 97; $LOOM terminal-raw-probe --read-byte" /dev/null
  )"
  [[ "$output" == *"LOOM_TERMINAL_RAW state=pass"* ]] ||
    fail "$label did not report raw mode: $output"
  [[ "$output" == *"opost=preserved"* ]] ||
    fail "$label changed terminal output processing: $output"
  [[ "$output" == *"dimensions=97x31"* ]] ||
    fail "$label did not preserve terminal geometry: $output"
  [[ "$output" == *"byte=$expected"* ]] ||
    fail "$label changed in the PTY; expected=$expected output=$output"
}

"$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null

# These bytes are intercepted or translated unless the attach client uses
# POSIX raw terminal settings rather than only disabling echo and canonical mode.
probe_byte enter '\r' 0d
probe_byte interrupt '\003' 03
probe_byte flow-control '\023' 13

echo "sounio-loom-terminal-raw-selftest: PASS"
