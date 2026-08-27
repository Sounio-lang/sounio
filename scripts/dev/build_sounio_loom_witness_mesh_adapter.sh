#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_WITNESS_MESH_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_WITNESS_MESH_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_WITNESS_MESH_MODULE:-$ROOT_DIR/stdlib/coordination/loom_witness_mesh.sio}"
ENTRYPOINT="${SOUNIO_LOOM_WITNESS_MESH_MAIN:-$ROOT_DIR/tools/loom/witness_mesh_adapter_main.sio}"
OUTPUT="${SOUNIO_LOOM_WITNESS_MESH_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-witness-mesh-runtime}"
PREBUILT="${SOUNIO_LOOM_WITNESS_MESH_PREBUILT:-}"

fail() {
  printf 'build-sounio-loom-witness-mesh: FAIL: %s\n' "$*" >&2
  exit 1
}

mkdir -p "$(dirname "$OUTPUT")"

if [[ -n "$PREBUILT" ]]; then
  [[ -x "$PREBUILT" ]] || fail "prebuilt adapter is not executable: $PREBUILT"
  install -m 0755 "$PREBUILT" "$OUTPUT"
else
  [[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
  [[ -f "$MODULE" ]] || fail "witness-mesh module is missing: $MODULE"
  [[ -f "$ENTRYPOINT" ]] || fail "witness-mesh entrypoint is missing: $ENTRYPOINT"

  work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-witness-mesh-build.XXXXXX")"
  trap 'rm -rf "$work"' EXIT
  combined="$work/loom_witness_mesh_adapter.sio"
  compiled="$work/sounio-loom-witness-mesh-runtime"
  sed -n '1,$p' "$MODULE" "$ENTRYPOINT" > "$combined"
  SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
  [[ -f "$compiled" ]] || fail 'compiler omitted the native witness-mesh adapter executable'
  install -m 0755 "$compiled" "$OUTPUT"
fi

one='1 1 1 1 1 1 1 1'
two='2 2 2 2 2 2 2 2'
zero='0 0 0 0 0 0 0 0'
probe="$(printf '9013 2 1 1 0 101 201 301 101 201 0 401 401 401 0 0 1 1 1 0 0 3 3 3 0 %s %s %s %s %s %s %s %s\n' \
  "$one" "$one" "$one" "$zero" "$two" "$two" "$two" "$zero" | "$OUTPUT")"
[[ "$probe" == \
  'SOUNIO_WITNESS_MESH_ACCEPT schema=loom-native-witness-mesh-v0 transition=anchor state=quorum-verified' ]] || \
  fail "native witness-mesh adapter failed its anchor probe: $probe"

printf 'BUILT_WITNESS_MESH path=%s language=Sounio engine=%s\n' "$OUTPUT" "$ENGINE"
