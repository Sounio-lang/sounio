#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_PROCESS_WITNESS_HANDSHAKE_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_PROCESS_WITNESS_HANDSHAKE_ENGINE:-lean_single}"
SOURCE="$ROOT_DIR/tools/loom/process_witness_handshake_payload_main.sio"
GARDEN="$ROOT_DIR/tools/loom/GARDEN_PROCESS_WITNESS_EXEC_HANDSHAKE_V1.md"
SEMANTIC_MANIFEST="$ROOT_DIR/tools/loom/kernel_exec_grant_cell_authority.freeze.v1"
HOST_GRANT_MANIFEST="$ROOT_DIR/tools/loom/host_exec_quorum_host.runtime.v1"
PARENT_PAYLOAD_MANIFEST="$ROOT_DIR/tools/loom/process_witness_payload.freeze.v1"
OUTPUT="${SOUNIO_LOOM_PROCESS_WITNESS_HANDSHAKE_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-process-witness-handshake}"

fail() {
  printf 'build-sounio-loom-process-witness-handshake-payload: FAIL: %s\n' "$*" >&2
  exit 1
}

file_hash() {
  local sum
  sum="$(sha256sum "$1")"
  printf '%s' "${sum%% *}"
}

record_field() {
  local path="$1" key="$2" count line
  count="$(grep -c "^${key}=" "$path" || true)"
  [[ "$count" == 1 ]] || fail "$key occurs $count times in ${path#$ROOT_DIR/}"
  line="$(grep -m1 "^${key}=" "$path")"
  printf '%s' "${line#*=}"
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
for path in "$SOURCE" "$GARDEN" "$SEMANTIC_MANIFEST" "$HOST_GRANT_MANIFEST" "$PARENT_PAYLOAD_MANIFEST"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "required input is absent or linked: ${path#$ROOT_DIR/}"
done
[[ "$(file_hash "$GARDEN")" == b2e73974a7b5e06269c808b46e63602e06d83eedafad6f450c576eccca80d9dc ]] ||
  fail 'ProcessWitness exec-handshake Garden drifted'
[[ "$(git -C "$ROOT_DIR" show 3f4a09b7ceb4d2f2ed5da5e80e25604661d0a3b3:tools/loom/GARDEN_PROCESS_WITNESS_EXEC_HANDSHAKE_V1.md | sha256sum | cut -d ' ' -f 1)" == \
  b2e73974a7b5e06269c808b46e63602e06d83eedafad6f450c576eccca80d9dc ]] ||
  fail 'exec-handshake Garden differs from its preregistration commit'
[[ "$(file_hash "$SEMANTIC_MANIFEST")" == 8687d889e08f69190daaf3cdbee02741cde3ce62f136ba63df1fa9c2ccb0d051 &&
   "$(record_field "$SEMANTIC_MANIFEST" stage)" == SEMANTICS_FROZEN &&
   "$(record_field "$SEMANTIC_MANIFEST" producing_language)" == Sounio &&
   "$(record_field "$SEMANTIC_MANIFEST" language_role)" == SEMANTIC_AUTHORITY &&
   "$(record_field "$SEMANTIC_MANIFEST" action)" == 9030 ]] ||
  fail 'Sounio action 9030 authority contract drifted'
[[ "$(file_hash "$HOST_GRANT_MANIFEST")" == 8c0851bb5e0f2f1982ec220d3e335bfd8c41e6b0500a763c02a3f1901c834ac5 &&
   "$(record_field "$HOST_GRANT_MANIFEST" material_grant)" == true &&
   "$(record_field "$HOST_GRANT_MANIFEST" material_execution)" == false ]] ||
  fail 'host material grant boundary drifted'
[[ "$(file_hash "$PARENT_PAYLOAD_MANIFEST")" == 59d9acbdf097edc6d025cbaca8dfe9b6241e83d32edc32a5bd13b94b84e85ed8 &&
   "$(record_field "$PARENT_PAYLOAD_MANIFEST" producing_language)" == Sounio &&
   "$(record_field "$PARENT_PAYLOAD_MANIFEST" stage)" == SOUNIO_PAYLOAD_FROZEN ]] ||
  fail 'parent calibration payload drifted'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-process-witness-handshake.XXXXXX")"
trap 'rm -rf "$work"' EXIT
compiled="$work/sounio-loom-process-witness-handshake"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$SOURCE" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the Sounio handshake executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

printf 'BUILT_LOOM_PROCESS_WITNESS_HANDSHAKE path=%s producer=Sounio role=SEMANTIC_PAYLOAD action=9030 engine=%s parent_payload_sha256=59d9acbdf097edc6d025cbaca8dfe9b6241e83d32edc32a5bd13b94b84e85ed8 material_grant=true material_execution=false\n' \
  "$OUTPUT" "$ENGINE"
