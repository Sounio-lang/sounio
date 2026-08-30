#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_HOST_BOOT_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_HOST_BOOT_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_HOST_BOOT_MODULE:-$ROOT_DIR/stdlib/coordination/loom_host_boot_reconciler.sio}"
ENTRYPOINT="${SOUNIO_LOOM_HOST_BOOT_MAIN:-$ROOT_DIR/tools/loom/host_boot_reconciler_main.sio}"
OUTPUT="${SOUNIO_LOOM_HOST_BOOT_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-host-boot-reconciler}"

fail() {
  printf 'build-sounio-loom-host-boot-reconciler: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
[[ -f "$MODULE" && ! -L "$MODULE" ]] || fail 'Sounio authority module is absent or linked'
[[ -f "$ENTRYPOINT" && ! -L "$ENTRYPOINT" ]] || fail 'Sounio entrypoint is absent or linked'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-host-boot.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/host_boot_reconciler.sio"
compiled="$work/sounio-loom-host-boot-reconciler"
sed -n '1,$p' "$MODULE" "$ENTRYPOINT" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the native Sounio executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

probe="$(printf '0\n' | "$OUTPUT")"
[[ "$probe" == 'SOUNIO_HOST_BOOT_RECONCILER_SELFTEST PASS cases=14' ]] ||
  fail "Sounio selftest diverged: $probe"
printf 'BUILT_HOST_BOOT_RECONCILER path=%s language=Sounio role=SEMANTIC_AUTHORITY action=9041 engine=%s cases=14\n' \
  "$OUTPUT" "$ENGINE"
