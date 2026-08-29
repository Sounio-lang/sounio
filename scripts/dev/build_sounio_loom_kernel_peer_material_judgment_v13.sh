#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_KERNEL_PEER_MATERIAL_JUDGMENT_V13_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_KERNEL_PEER_MATERIAL_JUDGMENT_V13_ENGINE:-lean_single}"
SOURCE="$ROOT_DIR/tools/loom/kernel_peer_material_judgment_v13_main.sio"
SEMANTIC_MANIFEST="$ROOT_DIR/tools/loom/kernel_peer_authority_plan_v13.freeze.v1"
MATERIAL_MANIFEST="$ROOT_DIR/tools/loom/kernel_peer_controls_v13.freeze.v1"
ACTION_9025_MANIFEST="$ROOT_DIR/tools/loom/effect_closure_authority.freeze.v1"
OUTPUT="${SOUNIO_LOOM_KERNEL_PEER_MATERIAL_JUDGMENT_V13_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-kernel-peer-material-judgment-v13}"

fail() {
  printf 'build-sounio-loom-kernel-peer-material-judgment-v13: FAIL: %s\n' "$*" >&2
  exit 1
}
expect_hash() {
  local path="$1" expected="$2"
  [[ -f "$path" && ! -L "$path" ]] || fail "frozen input is absent or linked: $path"
  [[ "$(sha256sum "$path" | cut -d ' ' -f 1)" == "$expected" ]] ||
    fail "frozen input hash drifted: $path"
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
expect_hash "$SOURCE" 3383ad078cbdc3d029a1c96a6a3bd20928b206f6a634d2a2d4afced0c693accb
expect_hash "$SEMANTIC_MANIFEST" b3557d850ce0dc13c900f8dbb10c33f824ac25e908cb4a48dd2ef913267194c2
expect_hash "$MATERIAL_MANIFEST" 7ffdff3f9dd48753502e9151a117fdcac8ea5149ef4772aaea5594269c54b301
expect_hash "$ACTION_9025_MANIFEST" c1f0cf93f8427acdf794246a11c3551e265a09be12a3cd000bad25b707e8ca91
grep -Fxq 'stage=SEMANTICS_FROZEN_V13' "$SEMANTIC_MANIFEST" ||
  fail 'V13 peer semantics are not frozen'
grep -Fxq 'stage=MATERIAL_CONTROL_MATRIX_FROZEN_V13' "$MATERIAL_MANIFEST" ||
  fail 'V13 material controls are not frozen'
grep -Fxq 'controls_executed=true' "$MATERIAL_MANIFEST" ||
  fail 'V13 material controls were not executed'
grep -Fxq 'material_peer_matrix=true' "$MATERIAL_MANIFEST" ||
  fail 'V13 material peer matrix is incomplete'
grep -Fxq 'same_uid_peer_isolation=false' "$MATERIAL_MANIFEST" ||
  fail 'material producer attempted semantic self-promotion'
grep -Fxq 'action_9025_decision=DENY451' "$MATERIAL_MANIFEST" ||
  fail 'material producer did not preserve the prematerial denial'
grep -Fxq 'next_stage=SOUNIO_JUDGMENT_V13' "$MATERIAL_MANIFEST" ||
  fail 'material certificate does not authorize Sounio judgment'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-kernel-peer-material-judgment-v13.XXXXXX")"
trap 'rm -rf "$work"' EXIT
compiled="$work/sounio-loom-kernel-peer-material-judgment-v13"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$SOURCE" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the Sounio V13 material judgment executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

fixtures="$(printf '0\n' | "$OUTPUT")"
metadata="$(printf '%s\n' "$fixtures" | sed -n '1p')"
[[ "$metadata" == 'LOOM_KERNEL_PEER_MATERIAL_JUDGMENT_V13 producer=Sounio role=SEMANTIC_AUTHORITY semantic_authority=Sounio action=9025 semantic_manifest_sha256=b3557d850ce0dc13c900f8dbb10c33f824ac25e908cb4a48dd2ef913267194c2 material_manifest_sha256=7ffdff3f9dd48753502e9151a117fdcac8ea5149ef4772aaea5594269c54b301 cases=14' ]] ||
  fail "V13 material judgment metadata diverged: $metadata"
[[ "$(printf '%s\n' "$fixtures" | grep -c '^CASE ')" == 14 ]] ||
  fail 'Sounio V13 material judgment fixture count drifted'
[[ "$(printf '%s\n' "$fixtures" | grep -c '^EXPECTED_ACTION_9025 ')" == 1 ]] ||
  fail 'Sounio V13 material judgment omitted the action boundary'

printf 'BUILT_KERNEL_PEER_MATERIAL_JUDGMENT_V13 path=%s language=Sounio role=SEMANTIC_AUTHORITY semantic_authority=Sounio action=9025 engine=%s cases=14 observations=50 controls=30 sabotage_twins=5 material_peer_matrix=true prematerial_decision=DENY451\n' \
  "$OUTPUT" "$ENGINE"
