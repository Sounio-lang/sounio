#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_KERNEL_PEER_PLAN_V12_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_KERNEL_PEER_PLAN_V12_ENGINE:-lean_single}"
SOURCE="$ROOT_DIR/tools/loom/kernel_peer_authority_plan_v12_main.sio"
GARDEN="$ROOT_DIR/tools/loom/GARDEN_KERNEL_PEER_AUTHORITY_V12.md"
V11_MANIFEST="$ROOT_DIR/tools/loom/process_witness_effect_material_judgment_v11.freeze.v1"
V11_EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-process-witness-effect-material-judgment-v11-20260829.txt"
ACTION_9025_MANIFEST="$ROOT_DIR/tools/loom/effect_closure_authority.freeze.v1"
OUTPUT="${SOUNIO_LOOM_KERNEL_PEER_PLAN_V12_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-kernel-peer-authority-plan-v12}"

fail() {
  printf 'build-sounio-loom-kernel-peer-authority-plan-v12: FAIL: %s\n' "$*" >&2
  exit 1
}

expect_hash() {
  local path="$1" expected="$2"
  [[ -f "$path" && ! -L "$path" ]] || fail "frozen input is absent or linked: $path"
  [[ "$(sha256sum "$path" | cut -d ' ' -f 1)" == "$expected" ]] ||
    fail "frozen input hash drifted: $path"
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
expect_hash "$GARDEN" \
  f426b8b1c0b6dd0233345225ff56bce2c65fb95d8e227076fdb5611c9808bec2
expect_hash "$SOURCE" \
  42f71e0c77b5997bc35ab5df73e50108ee41b444d59b2ed736e015b14864b2d9
expect_hash "$V11_MANIFEST" \
  f227cca70aa30351517403e13f60143c683bb86d445320661d68c08317c81b89
expect_hash "$V11_EVIDENCE" \
  4aa5704fe529ee93c88992a630976395b49a28ed13189af9d7a07aeb7ecc4c64
expect_hash "$ACTION_9025_MANIFEST" \
  c1f0cf93f8427acdf794246a11c3551e265a09be12a3cd000bad25b707e8ca91
grep -Fxq 'action_9025_decision=DENY451' "$V11_MANIFEST" ||
  fail 'V11 did not freeze the peer-isolation denial'
grep -Fxq 'same_uid_peer_isolation=false' "$V11_MANIFEST" ||
  fail 'V11 peer-isolation fact drifted'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-kernel-peer-plan-v12.XXXXXX")"
trap 'rm -rf "$work"' EXIT
compiled="$work/sounio-loom-kernel-peer-authority-plan-v12"
bundle="$work/semantic-bundle"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$SOURCE" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the Sounio V12 executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

# Bound both wall time and output size. A bootstrap regression once repeated a
# long fixture frame; such bytes are not admissible semantic evidence.
(ulimit -f 2048; timeout 10s "$OUTPUT" >"$bundle") ||
  fail 'Sounio V12 semantic bundle timed out or crossed its output bound'
[[ "$(wc -c <"$bundle")" -le 65536 ]] || fail 'Sounio V12 semantic bundle is oversized'
[[ "$(sha256sum "$bundle" | cut -d ' ' -f 1)" == \
  94d22ea974168f41200684c60da5e673a4afebb7e34f0b4cd8f228d0303e7b97 ]] ||
  fail 'Sounio V12 semantic bundle drifted'
[[ "$(grep -c '^VERTEX ' "$bundle")" == 50 ]] || fail 'V12 vertex count drifted'
[[ "$(grep -c '^OPERATION ' "$bundle")" == 10 ]] || fail 'V12 operation count drifted'
[[ "$(grep -c '^RECEIVER_PROPERTY ' "$bundle")" == 7 ]] ||
  fail 'V12 receiver-property count drifted'
[[ "$(grep -c '^ACTION_CASE ' "$bundle")" == 3 ]] || fail 'V12 action-case count drifted'

metadata="$(sed -n '1p' "$bundle")"
[[ "$metadata" == 'LOOM_KERNEL_PEER_AUTHORITY_PLAN_V12 producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 schema=12 principal_vertices=5 operations=10 observations=50 decisive_pairs=10 receiver_properties=7 action_cases=3' ]] ||
  fail "V12 policy metadata diverged: $metadata"

printf 'BUILT_KERNEL_PEER_AUTHORITY_PLAN_V12 path=%s language=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 engine=%s principal_vertices=5 operations=10 observations=50 decisive_pairs=10 receiver_properties=7 output_bound_bytes=65536\n' \
  "$OUTPUT" "$ENGINE"
