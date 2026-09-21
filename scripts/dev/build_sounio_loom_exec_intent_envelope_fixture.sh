#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_EXEC_INTENT_ENVELOPE_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_EXEC_INTENT_ENVELOPE_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_EXEC_INTENT_ENVELOPE_MODULE:-$ROOT_DIR/stdlib/coordination/loom_exec_intent_envelope_authority.sio}"
ENTRYPOINT="${SOUNIO_LOOM_EXEC_INTENT_ENVELOPE_MAIN:-$ROOT_DIR/tools/loom/exec_intent_envelope_authority_main.sio}"
OUTPUT="${SOUNIO_LOOM_EXEC_INTENT_ENVELOPE_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-exec-intent-envelope}"

fail() {
  printf 'build-sounio-loom-exec-intent-envelope: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
[[ -f "$MODULE" && ! -L "$MODULE" ]] || fail 'Sounio authority module is absent or linked'
[[ -f "$ENTRYPOINT" && ! -L "$ENTRYPOINT" ]] || fail 'Sounio entrypoint is absent or linked'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-exec-intent-envelope.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/exec_intent_envelope.sio"
compiled="$work/sounio-loom-exec-intent-envelope"
sed -n '1,$p' "$MODULE" "$ENTRYPOINT" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the native Sounio executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

probe="$(printf '0\n' | "$OUTPUT")"
[[ "$probe" == 'SOUNIO_EXEC_INTENT_ENVELOPE_SELFTEST PASS cases=12' ]] ||
  fail "Sounio selftest diverged: $probe"
printf 'BUILT_EXEC_INTENT_ENVELOPE path=%s language=Sounio role=SEMANTIC_AUTHORITY action=9034 engine=%s cases=12\n' \
  "$OUTPUT" "$ENGINE"
