#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_WITNESS_MESH_V1_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_WITNESS_MESH_V1_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_WITNESS_MESH_V1_MODULE:-$ROOT_DIR/stdlib/coordination/loom_witness_mesh_v1.sio}"
ENTRYPOINT="${SOUNIO_LOOM_WITNESS_MESH_V1_MAIN:-$ROOT_DIR/tools/loom/witness_mesh_v1_adapter_main.sio}"
OUTPUT="${SOUNIO_LOOM_WITNESS_MESH_V1_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-witness-mesh-v1-runtime}"
PREBUILT="${SOUNIO_LOOM_WITNESS_MESH_V1_PREBUILT:-}"

fail() {
  printf 'build-sounio-loom-witness-mesh-v1: FAIL: %s\n' "$*" >&2
  exit 1
}

mkdir -p "$(dirname "$OUTPUT")"

if [[ -n "$PREBUILT" ]]; then
  [[ -x "$PREBUILT" ]] || fail "prebuilt adapter is not executable: $PREBUILT"
  install -m 0755 "$PREBUILT" "$OUTPUT"
else
  [[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
  [[ -f "$MODULE" ]] || fail "witness-mesh-v1 module is missing: $MODULE"
  [[ -f "$ENTRYPOINT" ]] || fail "witness-mesh-v1 entrypoint is missing: $ENTRYPOINT"

  work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-witness-mesh-v1-build.XXXXXX")"
  trap 'rm -rf "$work"' EXIT
  combined="$work/loom_witness_mesh_v1_adapter.sio"
  compiled="$work/sounio-loom-witness-mesh-v1-runtime"
  sed -n '1,$p' "$MODULE" "$ENTRYPOINT" > "$combined"
  SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
  [[ -f "$compiled" ]] || fail 'compiler omitted the native witness-mesh-v1 adapter executable'
  install -m 0755 "$compiled" "$OUTPUT"
fi

one='1 1 1 1 1 1 1 1'
two='2 2 2 2 2 2 2 2'
zero='0 0 0 0 0 0 0 0'
probe="$(printf '9014 3 1 1 1 0 101 201 301 401 101 201 301 0 501 501 501 501 0 0 1 1 1 1 0 0 3 3 3 3 0 %s %s %s %s %s %s %s %s %s %s\n' \
  "$one" "$one" "$one" "$one" "$zero" \
  "$two" "$two" "$two" "$two" "$zero" | "$OUTPUT")"
[[ "$probe" == \
  'SOUNIO_WITNESS_MESH_V1_ACCEPT schema=loom-native-witness-mesh-v1 transition=anchor state=quorum-verified' ]] || \
  fail "native witness-mesh-v1 adapter failed its anchor probe: $probe"

printf 'BUILT_WITNESS_MESH_V1 path=%s language=Sounio engine=%s\n' "$OUTPUT" "$ENGINE"
