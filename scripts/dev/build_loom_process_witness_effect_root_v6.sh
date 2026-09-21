#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
OUTPUT_ROOT="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_ROOT_V6_OUTPUT:-$ROOT_DIR/tools/loom/_build/effect-root-v6}"
CELL_BUILDER="$ROOT_DIR/scripts/dev/build_loom_process_witness_effect_policy_v6.sh"
PAYLOAD_BUILDER="$ROOT_DIR/scripts/dev/build_sounio_loom_process_witness_handshake_payload.sh"
POLICY_MANIFEST="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v6.freeze.v1"
PAYLOAD_MANIFEST="$ROOT_DIR/tools/loom/process_witness_handshake_payload.freeze.v1"

fail() {
  printf 'build-loom-process-witness-effect-root-v6: FAIL reason=%s\n' "$*" >&2
  exit 1
}

[[ "$OUTPUT_ROOT" == /* ]] || fail 'output root must be absolute'
for tool in install find sha256sum readelf; do
  command -v "$tool" >/dev/null 2>&1 || fail "required build tool is absent: $tool"
done
for path in "$CELL_BUILDER" "$PAYLOAD_BUILDER"; do
  [[ -x "$path" && -f "$path" && ! -L "$path" ]] ||
    fail "builder is absent, linked, or non-executable: $path"
done
[[ "$(sha256sum "$POLICY_MANIFEST" | cut -d ' ' -f 1)" == \
  6ec33f3554236e7ccf73f5b5c16a15ba8006705b83d9d62265a2cd8f94437d66 ]] ||
  fail 'frozen V6 policy manifest drifted'
[[ "$(sha256sum "$PAYLOAD_MANIFEST" | cut -d ' ' -f 1)" == \
  624ccd7297778803eff8d9972a33d5e55fb022f9e7e37f444f0aee13c22fb4da ]] ||
  fail 'frozen Sounio payload manifest drifted'
if [[ -e "$OUTPUT_ROOT" ]] &&
   [[ -n "$(find "$OUTPUT_ROOT" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  fail 'output root is not empty'
fi

install -d -m 0755 "$OUTPUT_ROOT" "$OUTPUT_ROOT/loom" "$OUTPUT_ROOT/dev" \
  "$OUTPUT_ROOT/proc" "$OUTPUT_ROOT/tmp" "$OUTPUT_ROOT/run" \
  "$OUTPUT_ROOT/run/systemd" "$OUTPUT_ROOT/run/systemd/incoming" \
  "$OUTPUT_ROOT/sys" "$OUTPUT_ROOT/var" "$OUTPUT_ROOT/var/tmp"
SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V6_OUTPUT="$OUTPUT_ROOT/loom/effect-cell" \
  "$CELL_BUILDER" >/dev/null
SOUNIO_LOOM_PROCESS_WITNESS_HANDSHAKE_OUTPUT="$OUTPUT_ROOT/loom/payload" \
  "$PAYLOAD_BUILDER" >/dev/null
install -m 0444 "$PAYLOAD_MANIFEST" "$OUTPUT_ROOT/loom/payload.freeze.v1"
install -m 0444 "$POLICY_MANIFEST" "$OUTPUT_ROOT/loom/effect-policy-v6.freeze.v1"
chmod 0555 "$OUTPUT_ROOT/loom/effect-cell" "$OUTPUT_ROOT/loom/payload"
chmod 0555 "$OUTPUT_ROOT" "$OUTPUT_ROOT/loom" "$OUTPUT_ROOT/dev" \
  "$OUTPUT_ROOT/proc" "$OUTPUT_ROOT/tmp" "$OUTPUT_ROOT/run" \
  "$OUTPUT_ROOT/run/systemd" "$OUTPUT_ROOT/run/systemd/incoming" \
  "$OUTPUT_ROOT/sys" "$OUTPUT_ROOT/var" "$OUTPUT_ROOT/var/tmp"

for binary in "$OUTPUT_ROOT/loom/effect-cell" "$OUTPUT_ROOT/loom/payload"; do
  if readelf -l "$binary" | grep -q 'INTERP'; then
    fail "root binary retained a dynamic interpreter: $binary"
  fi
done

CELL_SHA256="$(sha256sum "$OUTPUT_ROOT/loom/effect-cell" | cut -d ' ' -f 1)"
PAYLOAD_SHA256="$(sha256sum "$OUTPUT_ROOT/loom/payload" | cut -d ' ' -f 1)"
[[ "$PAYLOAD_SHA256" == \
  7249748c322ede756c779904cb2d87f561ba2e17d0691314a09feaf16ca2ed4d ]] ||
  fail 'source-fresh static Sounio payload drifted'
TREE_RECORD="loom/effect-cell:0555:$CELL_SHA256
loom/payload:0555:$PAYLOAD_SHA256
loom/payload.freeze.v1:0444:624ccd7297778803eff8d9972a33d5e55fb022f9e7e37f444f0aee13c22fb4da
loom/effect-policy-v6.freeze.v1:0444:6ec33f3554236e7ccf73f5b5c16a15ba8006705b83d9d62265a2cd8f94437d66
dev/null:host-character-device:1:3
proc:empty:0555
tmp:empty-read-only:0555
run/systemd/incoming:empty-systemd-mountpoint:0555
sys:empty-systemd-mountpoint:0555
var/tmp:empty-read-only-systemd-mountpoint:0555"
TREE_SHA256="$(printf '%s\n' "$TREE_RECORD" | sha256sum | cut -d ' ' -f 1)"

printf 'BUILT_LOOM_PROCESS_WITNESS_EFFECT_ROOT_V6 path=%s producer=C++20+Sounio role=MATERIAL_PARITY semantic_authority=Sounio action=9025 capsule=true cell_sha256=%s payload_sha256=%s policy_manifest_sha256=6ec33f3554236e7ccf73f5b5c16a15ba8006705b83d9d62265a2cd8f94437d66 payload_manifest_sha256=624ccd7297778803eff8d9972a33d5e55fb022f9e7e37f444f0aee13c22fb4da tree_sha256=%s static_cell=true static_payload=true systemd_mount=/run/systemd/incoming systemd_sys_mount=/sys systemd_var_tmp=/var/tmp dev_null=host_materialization_required host_root_ownership=false root_read_only=false material_coverage=false complete_effects=false material_execution=false launch_open=false\n' \
  "$OUTPUT_ROOT" "$CELL_SHA256" "$PAYLOAD_SHA256" "$TREE_SHA256"
