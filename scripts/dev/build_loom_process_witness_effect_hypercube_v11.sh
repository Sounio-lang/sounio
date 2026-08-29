#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
CXX="${SOUNIO_LOOM_EFFECT_HYPERCUBE_V11_CXX:-c++}"
SOURCE="$ROOT_DIR/tools/loom/src/loom_process_witness_effect_hypercube_v11.cpp"
POLICY_MANIFEST="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v11.freeze.v1"
OUTPUT="${SOUNIO_LOOM_EFFECT_HYPERCUBE_V11_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/loom-process-witness-effect-hypercube-v11}"

fail() {
  printf 'build-loom-process-witness-effect-hypercube-v11: FAIL reason=%s\n' "$*" >&2
  exit 1
}

for tool in "$CXX" sha256sum readelf; do
  command -v "$tool" >/dev/null 2>&1 || fail "required build tool is missing: $tool"
done
[[ -f "$SOURCE" && ! -L "$SOURCE" ]] || fail 'V11 material source is absent or linked'
[[ "$(sha256sum "$SOURCE" | cut -d ' ' -f 1)" == \
  424d8cd2d5b8b32880cfce7b9ab2825c66932404f1fb6e34f9f78692c6526d5a ]] ||
  fail 'V11 material source drifted'
[[ -f "$POLICY_MANIFEST" && ! -L "$POLICY_MANIFEST" ]] ||
  fail 'frozen Sounio V11 policy manifest is absent or linked'
[[ "$(sha256sum "$POLICY_MANIFEST" | cut -d ' ' -f 1)" == \
  adbc7151da91bd12928cf059a4fce01de59b38096bb7bebe55be0402fab9972c ]] ||
  fail 'frozen Sounio V11 policy manifest drifted'
for line in \
  'stage=SEMANTICS_FROZEN' \
  'semantic_authority=Sounio' \
  'family_count=12' \
  'probe_count=13' \
  'mechanism_dimension_count=18' \
  'vertex_count=40' \
  'vertex_hash_binding=invariant_sha256+delta_sha256+witness_sha256' \
  'expected_results_source=Sounio' \
  'native_v11_bytes_created=false' \
  'material_hypercube=false' \
  'material_coverage=false'; do
  grep -Fxq "$line" "$POLICY_MANIFEST" || fail "Sounio V11 manifest omitted: $line"
done

mkdir -p "$(dirname "$OUTPUT")"
stage="$(mktemp "${TMPDIR:-/tmp}/loom-effect-hypercube-v11.XXXXXX")"
trap 'rm -f "$stage"' EXIT
"$CXX" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic -static \
  -fno-record-gcc-switches \
  -frandom-seed=loom-process-witness-effect-hypercube-v11 \
  -ffile-prefix-map="$ROOT_DIR"=. -Wl,--build-id=none \
  "$SOURCE" -lcrypto -o "$stage"
if readelf -l "$stage" | grep -q 'INTERP'; then
  fail 'native V11 laboratory retained a dynamic interpreter'
fi
install -m 0755 "$stage" "$OUTPUT"

printf 'BUILT_LOOM_PROCESS_WITNESS_EFFECT_HYPERCUBE_V11 path=%s language=C++20 role=MATERIAL_PARITY transitory=true semantic_authority=Sounio semantic_decision=false action=9025 policy_manifest_sha256=adbc7151da91bd12928cf059a4fce01de59b38096bb7bebe55be0402fab9972c static=true families=12 probes=13 mechanism_dimensions=18 vertices=40 vertex_hash_binding=invariant_sha256+delta_sha256+witness_sha256 material_hypercube=false material_coverage=false complete_effects=false material_execution=false claim_ready=false\n' \
  "$OUTPUT"
