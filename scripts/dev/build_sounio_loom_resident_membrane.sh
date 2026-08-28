#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_RESIDENT_MEMBRANE_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_RESIDENT_MEMBRANE_ENGINE:-lean_single}"
MEMBRANE_MODULE="$ROOT_DIR/stdlib/coordination/loom_subprocess_membrane_authority.sio"
MEMBRANE_ENTRYPOINT="$ROOT_DIR/tools/loom/subprocess_membrane_main.sio"
RESIDENT_MODULE="$ROOT_DIR/stdlib/coordination/loom_resident_authority.sio"
RESIDENT_ENTRYPOINT="$ROOT_DIR/tools/loom/resident_authority_main.sio"
DISPATCH_MAIN="$ROOT_DIR/tools/loom/resident_membrane_main.sio"
MEMBRANE_MANIFEST="$ROOT_DIR/tools/loom/subprocess_membrane.freeze.v1"
RESIDENT_MANIFEST="$ROOT_DIR/tools/loom/resident_authority.freeze.v1"
OUTPUT="${SOUNIO_LOOM_RESIDENT_MEMBRANE_OUTPUT:-$ROOT_DIR/tools/loom/.runtime/sounio-loom-resident-membrane-runtime}"

fail() {
  printf 'build-sounio-loom-resident-membrane: FAIL: %s\n' "$*" >&2
  exit 1
}

manifest_value() {
  local manifest="$1" key="$2" count line
  count="$(grep -c "^${key}=" "$manifest" || true)"
  [[ "$count" == 1 ]] || fail "manifest field $key occurs $count times"
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

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
for path in "$MEMBRANE_MODULE" "$MEMBRANE_ENTRYPOINT" "$RESIDENT_MODULE" \
  "$RESIDENT_ENTRYPOINT" "$DISPATCH_MAIN"; do
  [[ -f "$path" ]] || fail "resident source is missing: $path"
done
verify_parent "$MEMBRANE_MANIFEST" loom-subprocess-membrane-freeze-v1 9023
verify_parent "$RESIDENT_MANIFEST" loom-resident-authority-freeze-v1 9024

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-resident-membrane-build.XXXXXX")"
trap 'rm -rf "$work"' EXIT
membrane_adapter="$work/subprocess_membrane_resident_adapter.sio"
resident_adapter="$work/resident_authority_resident_adapter.sio"
combined="$work/loom_resident_membrane_runtime.sio"
compiled="$work/sounio-loom-resident-membrane-runtime"

grep -Fqx 'fn main() -> i64 with IO, Mut, Div, Panic {' "$MEMBRANE_ENTRYPOINT" ||
  fail 'subprocess-membrane main signature changed'
[[ "$(grep -Fxc '    let raw = read_line()' "$MEMBRANE_ENTRYPOINT")" == 1 ]] ||
  fail 'subprocess-membrane input boundary changed'
sed \
  -e 's/^fn main() -> i64 with IO, Mut, Div, Panic {$/fn subprocess_membrane_decide_one() -> i64 with IO, Mut, Div, Panic {/' \
  -e 's/^    let raw = read_line()$/    let raw = resident_membrane_read_line()/' \
  "$MEMBRANE_ENTRYPOINT" > "$membrane_adapter"

grep -Fqx 'fn main() -> i64 with IO, Mut, Div, Panic {' "$RESIDENT_ENTRYPOINT" ||
  fail 'resident-authority main signature changed'
[[ "$(grep -Fxc '    let raw = read_line()' "$RESIDENT_ENTRYPOINT")" == 1 ]] ||
  fail 'resident-authority input boundary changed'
sed \
  -e 's/^fn main() -> i64 with IO, Mut, Div, Panic {$/fn resident_authority_decide_one() -> i64 with IO, Mut, Div, Panic {/' \
  -e 's/^    let raw = read_line()$/    let raw = resident_membrane_read_line()/' \
  "$RESIDENT_ENTRYPOINT" > "$resident_adapter"

# Source assembly and adapter derivation are mechanical. Both decisions remain
# the frozen Sounio functions and are checked byte-for-byte by the parity gate.
sed -n '1,$p' "$MEMBRANE_MODULE" "$membrane_adapter" "$RESIDENT_MODULE" \
  "$resident_adapter" "$DISPATCH_MAIN" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the resident Sounio executable'
install -m 0755 "$compiled" "$OUTPUT"

one='1 1 1 1 1 1 1 1'
zero='0 0 0 0 0 0 0 0'
probe_frame="9024 3 1 1 1 0 0 0 0 1 1 1 0 $one $one $zero $zero $one"
probe="$(printf '%s\n' '1' "$probe_frame" '0' | "$OUTPUT")"
[[ "$probe" == 'SOUNIO_RESIDENT_AUTHORITY_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN' ]] ||
  fail "resident Sounio probe failed: $probe"

printf 'BUILT_RESIDENT_MEMBRANE path=%s language=Sounio engine=%s actions=9023,9024\n' \
  "$OUTPUT" "$ENGINE"
