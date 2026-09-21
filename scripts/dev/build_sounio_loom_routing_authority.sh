#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_ROUTING_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_ROUTING_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_ROUTING_MODULE:-$ROOT_DIR/stdlib/coordination/loom_routing_authority.sio}"
ENTRYPOINT="${SOUNIO_LOOM_ROUTING_MAIN:-$ROOT_DIR/tools/loom/routing_authority_main.sio}"
OUTPUT="${SOUNIO_LOOM_ROUTING_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-routing-authority-runtime}"

fail() { printf 'build-sounio-loom-routing-authority: FAIL: %s\n' "$*" >&2; exit 1; }

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
[[ -f "$MODULE" ]] || fail "routing authority module is missing: $MODULE"
[[ -f "$ENTRYPOINT" ]] || fail "routing authority entrypoint is missing: $ENTRYPOINT"
mkdir -p "$(dirname "$OUTPUT")"
work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-routing-build.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/loom_routing_authority_runtime.sio"
compiled="$work/sounio-loom-routing-authority-runtime"

# Mechanical assembly only. Decisions and expected results originate in Sounio.
sed -n '1,$p' "$MODULE" "$ENTRYPOINT" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the native routing executable'
install -m 0755 "$compiled" "$OUTPUT"
probe="$(printf '0\n' | "$OUTPUT")"
[[ "$probe" == 'SOUNIO_ROUTING_AUTHORITY_SELFTEST PASS cases=29' ]] ||
  fail "Sounio-owned expected-result suite failed: $probe"
printf 'BUILT_ROUTING_AUTHORITY path=%s language=Sounio engine=%s cases=29\n' "$OUTPUT" "$ENGINE"
