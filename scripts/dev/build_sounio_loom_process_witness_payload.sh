#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_PROCESS_WITNESS_PAYLOAD_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_PROCESS_WITNESS_PAYLOAD_ENGINE:-lean_single}"
SOURCE="$ROOT_DIR/tools/loom/process_witness_payload_main.sio"
GARDEN="$ROOT_DIR/tools/loom/GARDEN_PROCESS_WITNESS_CELL_V1.md"
SEMANTIC_MANIFEST="$ROOT_DIR/tools/loom/kernel_exec_grant_cell_authority.freeze.v2"
HOST_GRANT_MANIFEST="$ROOT_DIR/tools/loom/host_exec_quorum_host.runtime.v1"
OUTPUT="${SOUNIO_LOOM_PROCESS_WITNESS_PAYLOAD_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-process-witness-payload}"

fail() {
  printf 'build-sounio-loom-process-witness-payload: FAIL: %s\n' "$*" >&2
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
for path in "$SOURCE" "$GARDEN" "$SEMANTIC_MANIFEST" "$HOST_GRANT_MANIFEST"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "required input is absent or linked: ${path#$ROOT_DIR/}"
done
[[ "$(file_hash "$GARDEN")" == 9538355b174c7d0fddb13ae5d554201ae408667b3328eb753895be377ac8c6b7 ]] ||
  fail 'ProcessWitnessCell Garden drifted'
[[ "$(git -C "$ROOT_DIR" show cb1dacd2244e04571786d67a18870d29759d5559:tools/loom/GARDEN_PROCESS_WITNESS_CELL_V1.md | sha256sum | cut -d ' ' -f 1)" == \
  9538355b174c7d0fddb13ae5d554201ae408667b3328eb753895be377ac8c6b7 ]] ||
  fail 'ProcessWitnessCell Garden does not match its preregistration commit'
[[ "$(file_hash "$SEMANTIC_MANIFEST")" == a763255a922b95c5eb8855c85250f2f3e9db5f54889ea910250434b1df00e0f4 ]] ||
  fail 'frozen action 9030 manifest drifted'
[[ "$(record_field "$SEMANTIC_MANIFEST" stage)" == SEMANTICS_FROZEN &&
   "$(record_field "$SEMANTIC_MANIFEST" producing_language)" == Sounio &&
   "$(record_field "$SEMANTIC_MANIFEST" language_role)" == SEMANTIC_AUTHORITY &&
   "$(record_field "$SEMANTIC_MANIFEST" action)" == 9030 ]] ||
  fail 'Sounio action 9030 authority contract drifted'
[[ "$(file_hash "$HOST_GRANT_MANIFEST")" == 8c0851bb5e0f2f1982ec220d3e335bfd8c41e6b0500a763c02a3f1901c834ac5 ]] ||
  fail 'frozen host material grant manifest drifted'
[[ "$(record_field "$HOST_GRANT_MANIFEST" stage)" == MATERIAL_GRANT_FROZEN &&
   "$(record_field "$HOST_GRANT_MANIFEST" semantic_authority)" == Sounio &&
   "$(record_field "$HOST_GRANT_MANIFEST" material_grant)" == true &&
   "$(record_field "$HOST_GRANT_MANIFEST" material_execution)" == false ]] ||
  fail 'host material grant boundary drifted'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-process-witness-payload.XXXXXX")"
trap 'rm -rf "$work"' EXIT
compiled="$work/sounio-loom-process-witness-payload"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$SOURCE" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the Sounio payload executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

printf 'BUILT_LOOM_PROCESS_WITNESS_PAYLOAD path=%s producer=Sounio role=SEMANTIC_PAYLOAD action=9030 engine=%s semantic_manifest_sha256=a763255a922b95c5eb8855c85250f2f3e9db5f54889ea910250434b1df00e0f4 host_grant_manifest_sha256=8c0851bb5e0f2f1982ec220d3e335bfd8c41e6b0500a763c02a3f1901c834ac5 material_grant=true material_execution=false\n' \
  "$OUTPUT" "$ENGINE"
