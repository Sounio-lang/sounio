#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_ENGINE:-lean_single}"
MEMBRANE_MODULE="$ROOT_DIR/stdlib/coordination/loom_subprocess_membrane_authority.sio"
MEMBRANE_ENTRYPOINT="$ROOT_DIR/tools/loom/subprocess_membrane_main.sio"
RESIDENT_MODULE="$ROOT_DIR/stdlib/coordination/loom_resident_authority.sio"
RESIDENT_ENTRYPOINT="$ROOT_DIR/tools/loom/resident_authority_main.sio"
CLOSURE_MODULE="$ROOT_DIR/stdlib/coordination/loom_effect_closure_authority.sio"
CLOSURE_ENTRYPOINT="$ROOT_DIR/tools/loom/effect_closure_authority_main.sio"
INVOCATION_MODULE="$ROOT_DIR/stdlib/coordination/loom_kernel_invocation_cell_authority.sio"
INVOCATION_ENTRYPOINT="$ROOT_DIR/tools/loom/kernel_invocation_cell_authority_main.sio"
GRANT_MODULE="$ROOT_DIR/stdlib/coordination/loom_kernel_exec_grant_cell_authority.sio"
GRANT_ENTRYPOINT="$ROOT_DIR/tools/loom/kernel_exec_grant_cell_authority_main.sio"
ACTIVATION_MODULE="$ROOT_DIR/stdlib/coordination/loom_kernel_peer_activation_capsule_authority.sio"
ACTIVATION_ENTRYPOINT="$ROOT_DIR/tools/loom/kernel_peer_activation_capsule_authority_main.sio"
DISPATCH_MAIN="$ROOT_DIR/tools/loom/resident_membrane_v5_main.sio"
MEMBRANE_MANIFEST="$ROOT_DIR/tools/loom/subprocess_membrane.freeze.v1"
RESIDENT_MANIFEST="$ROOT_DIR/tools/loom/resident_authority.freeze.v1"
CLOSURE_MANIFEST="$ROOT_DIR/tools/loom/effect_closure_authority.freeze.v1"
INVOCATION_MANIFEST="$ROOT_DIR/tools/loom/kernel_invocation_cell_authority.freeze.v1"
GRANT_MANIFEST="$ROOT_DIR/tools/loom/kernel_exec_grant_cell_authority.freeze.v1"
ACTIVATION_MANIFEST="$ROOT_DIR/tools/loom/kernel_peer_activation_capsule_authority.freeze.v1"
RESIDENT_V4_MANIFEST="$ROOT_DIR/tools/loom/resident_membrane.runtime.v4"
OUTPUT="${SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_OUTPUT:-$ROOT_DIR/tools/loom/.runtime/sounio-loom-resident-membrane-runtime-v5}"

fail() {
  printf 'build-sounio-loom-resident-membrane-v5: FAIL: %s\n' "$*" >&2
  exit 1
}

manifest_value() {
  local manifest="$1" key="$2" count line
  count="$(grep -c "^${key}=" "$manifest" || true)"
  [[ "$count" == 1 ]] || fail "manifest field $key occurs $count times in $manifest"
  line="$(grep -m1 "^${key}=" "$manifest")"
  printf '%s' "${line#*=}"
}

file_hash() {
  local sum
  sum="$(sha256sum "$1")"
  printf '%s' "${sum%% *}"
}

verify_parent() {
  local manifest="$1" schema="$2" action="$3"
  [[ -f "$manifest" ]] || fail "frozen parent manifest is missing: $manifest"
  [[ "$(manifest_value "$manifest" schema)" == "$schema" ]] || fail "wrong parent schema: $manifest"
  [[ "$(manifest_value "$manifest" stage)" == SEMANTICS_FROZEN ]] || fail "parent is not frozen: $manifest"
  [[ "$(manifest_value "$manifest" producing_language)" == Sounio ]] || fail "parent producer is not Sounio"
  [[ "$(manifest_value "$manifest" language_role)" == SEMANTIC_AUTHORITY ]] || fail "parent role is not semantic authority"
  [[ "$(manifest_value "$manifest" action)" == "$action" ]] || fail "wrong parent action: $manifest"
  local source_path entrypoint_path
  source_path="$ROOT_DIR/$(manifest_value "$manifest" source_path)"
  entrypoint_path="$ROOT_DIR/$(manifest_value "$manifest" entrypoint_path)"
  [[ "$(file_hash "$source_path")" == "$(manifest_value "$manifest" source_sha256)" ]] ||
    fail "frozen parent source drifted: $source_path"
  [[ "$(file_hash "$entrypoint_path")" == "$(manifest_value "$manifest" entrypoint_sha256)" ]] ||
    fail "frozen parent entrypoint drifted: $entrypoint_path"
}

adapt_entrypoint() {
  local entrypoint="$1" replacement="$2" output="$3"
  grep -Fqx 'fn main() -> i64 with IO, Mut, Div, Panic {' "$entrypoint" ||
    fail "main signature changed: $entrypoint"
  [[ "$(grep -Fxc '    let raw = read_line()' "$entrypoint")" == 1 ]] ||
    fail "input boundary changed: $entrypoint"
  sed \
    -e "s/^fn main() -> i64 with IO, Mut, Div, Panic {$/fn $replacement() -> i64 with IO, Mut, Div, Panic {/" \
    -e 's/^    let raw = read_line()$/    let raw = resident_membrane_v5_read_line()/' \
    "$entrypoint" > "$output"
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
for source_path in "$MEMBRANE_MODULE" "$MEMBRANE_ENTRYPOINT" "$RESIDENT_MODULE" \
  "$RESIDENT_ENTRYPOINT" "$CLOSURE_MODULE" "$CLOSURE_ENTRYPOINT" \
  "$INVOCATION_MODULE" "$INVOCATION_ENTRYPOINT" "$GRANT_MODULE" \
  "$GRANT_ENTRYPOINT" "$ACTIVATION_MODULE" "$ACTIVATION_ENTRYPOINT" \
  "$DISPATCH_MAIN"; do
  [[ -f "$source_path" ]] || fail "resident v5 source is missing: $source_path"
done
verify_parent "$MEMBRANE_MANIFEST" loom-subprocess-membrane-freeze-v1 9023
verify_parent "$RESIDENT_MANIFEST" loom-resident-authority-freeze-v1 9024
verify_parent "$CLOSURE_MANIFEST" loom-effect-closure-authority-freeze-v1 9025
verify_parent "$INVOCATION_MANIFEST" loom-kernel-invocation-cell-authority-freeze-v1 9029
verify_parent "$GRANT_MANIFEST" loom-kernel-exec-grant-cell-authority-freeze-v1 9030
verify_parent "$ACTIVATION_MANIFEST" loom-kernel-peer-activation-capsule-authority-freeze-v1 9031
[[ "$(manifest_value "$GRANT_MANIFEST" parent_9029_manifest_sha256)" == "$(file_hash "$INVOCATION_MANIFEST")" ]] ||
  fail 'action 9030 does not bind the current action 9029 manifest'
[[ "$(manifest_value "$ACTIVATION_MANIFEST" parent_9030_manifest_sha256)" == "$(file_hash "$GRANT_MANIFEST")" ]] ||
  fail 'action 9031 does not bind the current action 9030 manifest'
[[ "$(manifest_value "$ACTIVATION_MANIFEST" parent_9025_manifest_sha256)" == \
   "$(file_hash "$ROOT_DIR/$(manifest_value "$ACTIVATION_MANIFEST" parent_9025_manifest_path)")" ]] ||
  fail 'action 9031 does not bind the frozen V13 action 9025 material judgment'
[[ "$(manifest_value "$RESIDENT_V4_MANIFEST" parent_9030_sha256)" == "$(file_hash "$GRANT_MANIFEST")" ]] ||
  fail 'resident v4 does not bind the current action 9030 manifest'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-resident-membrane-v5-build.XXXXXX")"
trap 'rm -rf "$work"' EXIT
membrane_adapter="$work/subprocess_membrane_resident_v5_adapter.sio"
resident_adapter="$work/resident_authority_resident_v5_adapter.sio"
closure_adapter="$work/effect_closure_resident_v5_adapter.sio"
invocation_adapter="$work/kernel_invocation_cell_resident_v5_adapter.sio"
grant_adapter="$work/kernel_exec_grant_cell_resident_v5_adapter.sio"
activation_adapter="$work/kernel_peer_activation_capsule_resident_v5_adapter.sio"
combined="$work/loom_resident_membrane_v5_runtime.sio"
compiled="$work/sounio-loom-resident-membrane-runtime-v5"
activation_reference="$work/sounio-loom-kernel-peer-activation-capsule-authority-runtime"

adapt_entrypoint "$MEMBRANE_ENTRYPOINT" subprocess_membrane_v5_decide_one "$membrane_adapter"
adapt_entrypoint "$RESIDENT_ENTRYPOINT" resident_authority_v5_decide_one "$resident_adapter"
adapt_entrypoint "$CLOSURE_ENTRYPOINT" effect_closure_v5_decide_one "$closure_adapter"
adapt_entrypoint "$INVOCATION_ENTRYPOINT" kernel_invocation_cell_v5_decide_one "$invocation_adapter"
adapt_entrypoint "$GRANT_ENTRYPOINT" kernel_exec_grant_cell_v5_decide_one "$grant_adapter"
adapt_entrypoint "$ACTIVATION_ENTRYPOINT" kernel_peer_activation_capsule_v5_decide_one "$activation_adapter"

# Source assembly and adapter derivation are mechanical. Expected decisions
# remain in the six frozen Sounio functions and are checked byte-for-byte.
sed -n '1,$p' "$MEMBRANE_MODULE" "$membrane_adapter" "$RESIDENT_MODULE" \
  "$resident_adapter" "$CLOSURE_MODULE" "$closure_adapter" \
  "$INVOCATION_MODULE" "$invocation_adapter" "$GRANT_MODULE" \
  "$grant_adapter" "$ACTIVATION_MODULE" "$activation_adapter" \
  "$DISPATCH_MAIN" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the resident Sounio v5 executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

SOUNIO_LOOM_KERNEL_PEER_ACTIVATION_CAPSULE_SOUC="$SOUC" \
  SOUNIO_LOOM_KERNEL_PEER_ACTIVATION_CAPSULE_ENGINE="$ENGINE" \
  SOUNIO_LOOM_KERNEL_PEER_ACTIVATION_CAPSULE_OUTPUT="$activation_reference" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_peer_activation_capsule_authority.sh" >/dev/null
fixtures="$(printf '1\n' | "$activation_reference")"
probe_frame="$(printf '%s\n' "$fixtures" | sed -n 's/^CASE label=seal EXPECT code=[0-9][0-9]* FRAME //p')"
[[ -n "$probe_frame" ]] || fail 'Sounio action 9031 omitted the seal fixture'
expected="$(printf '%s\n' "$probe_frame" | "$activation_reference")"
probe="$(printf '%s\n' '6' "$probe_frame" '0' | "$OUTPUT")"
[[ "$probe" == "$expected" ]] ||
  fail "resident Sounio v5 diverged from standalone action 9031: resident=$probe standalone=$expected"

printf 'BUILT_RESIDENT_MEMBRANE_V5 path=%s language=Sounio engine=%s actions=9023,9024,9025,9029,9030,9031\n' \
  "$OUTPUT" "$ENGINE"
