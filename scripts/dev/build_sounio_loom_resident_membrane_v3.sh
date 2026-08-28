#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_RESIDENT_MEMBRANE_V3_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_RESIDENT_MEMBRANE_V3_ENGINE:-lean_single}"
MEMBRANE_MODULE="$ROOT_DIR/stdlib/coordination/loom_subprocess_membrane_authority.sio"
MEMBRANE_ENTRYPOINT="$ROOT_DIR/tools/loom/subprocess_membrane_main.sio"
RESIDENT_MODULE="$ROOT_DIR/stdlib/coordination/loom_resident_authority.sio"
RESIDENT_ENTRYPOINT="$ROOT_DIR/tools/loom/resident_authority_main.sio"
CLOSURE_MODULE="$ROOT_DIR/stdlib/coordination/loom_effect_closure_authority.sio"
CLOSURE_ENTRYPOINT="$ROOT_DIR/tools/loom/effect_closure_authority_main.sio"
CELL_MODULE="$ROOT_DIR/stdlib/coordination/loom_kernel_invocation_cell_authority.sio"
CELL_ENTRYPOINT="$ROOT_DIR/tools/loom/kernel_invocation_cell_authority_main.sio"
DISPATCH_MAIN="$ROOT_DIR/tools/loom/resident_membrane_v3_main.sio"
MEMBRANE_MANIFEST="$ROOT_DIR/tools/loom/subprocess_membrane.freeze.v1"
RESIDENT_MANIFEST="$ROOT_DIR/tools/loom/resident_authority.freeze.v1"
CLOSURE_MANIFEST="$ROOT_DIR/tools/loom/effect_closure_authority.freeze.v1"
CELL_MANIFEST="$ROOT_DIR/tools/loom/kernel_invocation_cell_authority.freeze.v1"
RESIDENT_V2_MANIFEST="$ROOT_DIR/tools/loom/resident_membrane.runtime.v2"
OUTPUT="${SOUNIO_LOOM_RESIDENT_MEMBRANE_V3_OUTPUT:-$ROOT_DIR/tools/loom/.runtime/sounio-loom-resident-membrane-runtime-v3}"

fail() {
  printf 'build-sounio-loom-resident-membrane-v3: FAIL: %s\n' "$*" >&2
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
    -e 's/^    let raw = read_line()$/    let raw = resident_membrane_v3_read_line()/' \
    "$entrypoint" > "$output"
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
for source_path in "$MEMBRANE_MODULE" "$MEMBRANE_ENTRYPOINT" "$RESIDENT_MODULE" \
  "$RESIDENT_ENTRYPOINT" "$CLOSURE_MODULE" "$CLOSURE_ENTRYPOINT" \
  "$CELL_MODULE" "$CELL_ENTRYPOINT" "$DISPATCH_MAIN"; do
  [[ -f "$source_path" ]] || fail "resident v3 source is missing: $source_path"
done
verify_parent "$MEMBRANE_MANIFEST" loom-subprocess-membrane-freeze-v1 9023
verify_parent "$RESIDENT_MANIFEST" loom-resident-authority-freeze-v1 9024
verify_parent "$CLOSURE_MANIFEST" loom-effect-closure-authority-freeze-v1 9025
verify_parent "$CELL_MANIFEST" loom-kernel-invocation-cell-authority-freeze-v1 9029
[[ "$(manifest_value "$CELL_MANIFEST" parent_9023_manifest_sha256)" == "$(file_hash "$MEMBRANE_MANIFEST")" ]] ||
  fail 'action 9029 does not bind the current action 9023 manifest'
[[ "$(manifest_value "$CELL_MANIFEST" parent_9025_manifest_sha256)" == "$(file_hash "$CLOSURE_MANIFEST")" ]] ||
  fail 'action 9029 does not bind the current action 9025 manifest'
[[ "$(manifest_value "$RESIDENT_V2_MANIFEST" parent_9025_sha256)" == "$(file_hash "$CLOSURE_MANIFEST")" ]] ||
  fail 'resident v2 does not bind the current action 9025 manifest'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-resident-membrane-v3-build.XXXXXX")"
trap 'rm -rf "$work"' EXIT
membrane_adapter="$work/subprocess_membrane_resident_v3_adapter.sio"
resident_adapter="$work/resident_authority_resident_v3_adapter.sio"
closure_adapter="$work/effect_closure_resident_v3_adapter.sio"
cell_adapter="$work/kernel_invocation_cell_resident_v3_adapter.sio"
combined="$work/loom_resident_membrane_v3_runtime.sio"
compiled="$work/sounio-loom-resident-membrane-runtime-v3"

adapt_entrypoint "$MEMBRANE_ENTRYPOINT" subprocess_membrane_v3_decide_one "$membrane_adapter"
adapt_entrypoint "$RESIDENT_ENTRYPOINT" resident_authority_v3_decide_one "$resident_adapter"
adapt_entrypoint "$CLOSURE_ENTRYPOINT" effect_closure_v3_decide_one "$closure_adapter"
adapt_entrypoint "$CELL_ENTRYPOINT" kernel_invocation_cell_v3_decide_one "$cell_adapter"

# Source assembly and adapter derivation are mechanical. Expected decisions
# remain in the four frozen Sounio functions and are checked byte-for-byte.
sed -n '1,$p' "$MEMBRANE_MODULE" "$membrane_adapter" "$RESIDENT_MODULE" \
  "$resident_adapter" "$CLOSURE_MODULE" "$closure_adapter" \
  "$CELL_MODULE" "$cell_adapter" "$DISPATCH_MAIN" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the resident Sounio v3 executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

parent_9028='1991017987 113822720 1367310835 4264184359 1117900107 2622180275 1259621157 4224578159'
parent_9025='3253784467 4165106381 4153681002 298013982 643434942 312724736 195896759 132696721'
parent_9023='2365323 2301161672 762924345 38070334 1558458629 1166539901 3590963442 1546541903'
one='1 1 1 1 1 1 1 1'
bindings="$parent_9028 $parent_9025 $parent_9023 $one $one $one $one $one $one $one $one $one"
prepare_join='1 1 1 1 1 1'
capsule='1 1 5 6 7 1 1 0 0 1 1 1'
membrane='1 8 9 10 11 1 1 1 1'
scope='1 1 1 1 1 1'
coverage='1 100 1 50 1 1 1 1'
lifecycle='1 1 1 12 13 1 0 0 0'
outcome='0 0 0 0 0 0 0 0 0 0'
authority='1 1 1 1 1 1'
evidence='1 1 10 10'
probe_frame="9029 3 1 $prepare_join $capsule $membrane $scope $coverage $lifecycle $outcome $authority $evidence $bindings"
probe="$(printf '%s\n' '4' "$probe_frame" '0' | "$OUTPUT")"
[[ "$probe" == 'SOUNIO_KERNEL_INVOCATION_CELL_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN' ]] ||
  fail "resident Sounio v3 probe failed: $probe"

printf 'BUILT_RESIDENT_MEMBRANE_V3 path=%s language=Sounio engine=%s actions=9023,9024,9025,9029\n' \
  "$OUTPUT" "$ENGINE"
