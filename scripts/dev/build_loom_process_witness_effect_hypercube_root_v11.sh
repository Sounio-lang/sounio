#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
OUTPUT_ROOT="${SOUNIO_LOOM_EFFECT_HYPERCUBE_ROOT_V11_OUTPUT:-$ROOT_DIR/tools/loom/_build/effect-hypercube-root-v11}"
CELL_BUILDER="$ROOT_DIR/scripts/dev/build_loom_process_witness_effect_hypercube_v11.sh"
POLICY_MANIFEST="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v11.freeze.v1"

fail() {
  printf 'build-loom-process-witness-effect-hypercube-root-v11: FAIL reason=%s\n' "$*" >&2
  exit 1
}

[[ "$OUTPUT_ROOT" == /* ]] || fail 'output root must be absolute'
for tool in install find sha256sum readelf; do
  command -v "$tool" >/dev/null 2>&1 || fail "required build tool is absent: $tool"
done
[[ -x "$CELL_BUILDER" && -f "$CELL_BUILDER" && ! -L "$CELL_BUILDER" ]] ||
  fail 'V11 material cell builder is absent, linked, or non-executable'
[[ "$(sha256sum "$POLICY_MANIFEST" | cut -d ' ' -f 1)" == \
  adbc7151da91bd12928cf059a4fce01de59b38096bb7bebe55be0402fab9972c ]] ||
  fail 'frozen V11 policy manifest drifted'
if [[ -e "$OUTPUT_ROOT" ]] &&
   [[ -n "$(find "$OUTPUT_ROOT" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  fail 'output root is not empty'
fi

install -d -m 0755 "$OUTPUT_ROOT" "$OUTPUT_ROOT/loom" "$OUTPUT_ROOT/dev" \
  "$OUTPUT_ROOT/proc" "$OUTPUT_ROOT/tmp" "$OUTPUT_ROOT/run" \
  "$OUTPUT_ROOT/run/systemd" "$OUTPUT_ROOT/run/systemd/incoming" \
  "$OUTPUT_ROOT/sys" "$OUTPUT_ROOT/var" "$OUTPUT_ROOT/var/tmp"
SOUNIO_LOOM_EFFECT_HYPERCUBE_V11_OUTPUT="$OUTPUT_ROOT/loom/effect-cell" \
  "$CELL_BUILDER" >/dev/null
install -m 0444 "$POLICY_MANIFEST" \
  "$OUTPUT_ROOT/loom/effect-policy-v11.freeze.v1"
chmod 0555 "$OUTPUT_ROOT/loom/effect-cell"
chmod 0555 "$OUTPUT_ROOT" "$OUTPUT_ROOT/loom" "$OUTPUT_ROOT/dev" \
  "$OUTPUT_ROOT/proc" "$OUTPUT_ROOT/tmp" "$OUTPUT_ROOT/run" \
  "$OUTPUT_ROOT/run/systemd" "$OUTPUT_ROOT/run/systemd/incoming" \
  "$OUTPUT_ROOT/sys" "$OUTPUT_ROOT/var" "$OUTPUT_ROOT/var/tmp"

if readelf -l "$OUTPUT_ROOT/loom/effect-cell" | grep -q 'INTERP'; then
  fail 'root material cell retained a dynamic interpreter'
fi
CELL_SHA256="$(sha256sum "$OUTPUT_ROOT/loom/effect-cell" | cut -d ' ' -f 1)"
TREE_RECORD="loom/effect-cell:0555:$CELL_SHA256
loom/effect-policy-v11.freeze.v1:0444:adbc7151da91bd12928cf059a4fce01de59b38096bb7bebe55be0402fab9972c
dev/null:host-character-device:1:3
proc:empty:0555
tmp:empty-mountpoint:0555
run/systemd/incoming:empty-systemd-mountpoint:0555
sys:empty-systemd-mountpoint:0555
var/tmp:empty-mountpoint:0555"
TREE_SHA256="$(printf '%s\n' "$TREE_RECORD" | sha256sum | cut -d ' ' -f 1)"

printf 'BUILT_LOOM_PROCESS_WITNESS_EFFECT_HYPERCUBE_ROOT_V11 path=%s producer=C++20 role=MATERIAL_PARITY transitory=true semantic_authority=Sounio semantic_decision=false action=9025 capsule=true cell_sha256=%s policy_manifest_sha256=adbc7151da91bd12928cf059a4fce01de59b38096bb7bebe55be0402fab9972c tree_sha256=%s static_cell=true self_exec_target=true families=12 probes=13 vertices=40 dev_null=host_materialization_required host_root_ownership=false root_read_only=false material_hypercube=false material_coverage=false complete_effects=false material_execution=false claim_ready=false\n' \
  "$OUTPUT_ROOT" "$CELL_SHA256" "$TREE_SHA256"
