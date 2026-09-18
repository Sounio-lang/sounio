#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_KERNEL_PEER_PLAN_V13_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_KERNEL_PEER_PLAN_V13_ENGINE:-lean_single}"
SOURCE="$ROOT_DIR/tools/loom/kernel_peer_authority_plan_v13_main.sio"
GARDEN="$ROOT_DIR/tools/loom/GARDEN_KERNEL_PEER_AUTHORITY_V13.md"
V12_SEMANTIC_MANIFEST="$ROOT_DIR/tools/loom/kernel_peer_authority_plan_v12.freeze.v1"
V12_MATRIX_MANIFEST="$ROOT_DIR/tools/loom/kernel_peer_matrix_v12.freeze.v1"
V12_FALSIFICATION_MANIFEST="$ROOT_DIR/tools/loom/kernel_peer_dumpable_prlimit_falsification_v12.freeze.v1"
ACTION_9025_MANIFEST="$ROOT_DIR/tools/loom/effect_closure_authority.freeze.v1"
OUTPUT="${SOUNIO_LOOM_KERNEL_PEER_PLAN_V13_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-kernel-peer-authority-plan-v13}"

fail() {
  printf 'build-sounio-loom-kernel-peer-authority-plan-v13: FAIL: %s\n' "$*" >&2
  exit 1
}
expect_hash() {
  local path="$1" expected="$2"
  [[ -f "$path" && ! -L "$path" ]] || fail "frozen input is absent or linked: $path"
  [[ "$(sha256sum "$path" | cut -d ' ' -f 1)" == "$expected" ]] ||
    fail "frozen input hash drifted: $path"
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
expect_hash "$GARDEN" 8c5f456b3979517ab42a62050bf07c6c0e66db9c79b5b55b9d244a0d715289e9
expect_hash "$SOURCE" 3545f75dca264b4378ab4cf633a686ffcde5152cb02ac18b74ab00192baed7f0
expect_hash "$V12_SEMANTIC_MANIFEST" daaf9b056bde02078163849fdcc0f0b3acb96a357741c70e3d23c688c2bd0b30
expect_hash "$V12_MATRIX_MANIFEST" 1692782657cbe6fe7a548b6f11d4d542d24fe05569686d536a4c69af0775cd75
expect_hash "$V12_FALSIFICATION_MANIFEST" d4b3cdc1dfc6c139538cffecddca60fe34498908b38a2476a7beba8e7e60db7e
expect_hash "$ACTION_9025_MANIFEST" c1f0cf93f8427acdf794246a11c3551e265a09be12a3cd000bad25b707e8ca91
grep -Fxq 'v12_hypothesis_falsified=true' "$V12_FALSIFICATION_MANIFEST" ||
  fail 'V12 material falsifier is not affirmative'
grep -Fxq 'material_observed=EFFECT_COMPLETED' "$V12_FALSIFICATION_MANIFEST" ||
  fail 'V12 material falsifier omitted the typed completion'
grep -Fxq 'next_stage=SOUNIO_V13_GARDEN' "$V12_FALSIFICATION_MANIFEST" ||
  fail 'V12 falsification does not authorize V13 Garden'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-kernel-peer-plan-v13.XXXXXX")"
trap 'rm -rf "$work"' EXIT
compiled="$work/sounio-loom-kernel-peer-authority-plan-v13"
bundle="$work/semantic-bundle"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$SOURCE" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the Sounio V13 executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

(ulimit -f 2048; timeout 10s "$OUTPUT" >"$bundle") ||
  fail 'Sounio V13 semantic bundle timed out or crossed its output bound'
[[ "$(wc -c <"$bundle")" -le 65536 ]] || fail 'Sounio V13 semantic bundle is oversized'
[[ "$(sha256sum "$bundle" | cut -d ' ' -f 1)" == 44a3052926f0958ee970fe21c772276102d6ff9069907f5f80f8b5aa5063ae87 ]] ||
  fail 'Sounio V13 semantic bundle drifted'
[[ "$(grep -c '^VERTEX ' "$bundle")" == 50 ]] || fail 'V13 vertex count drifted'
[[ "$(grep -c '^OPERATION ' "$bundle")" == 10 ]] || fail 'V13 operation count drifted'
[[ "$(grep -c '^RECEIVER_PROPERTY ' "$bundle")" == 7 ]] ||
  fail 'V13 receiver-property count drifted'
[[ "$(grep -c '^ACTION_CASE ' "$bundle")" == 3 ]] || fail 'V13 action-case count drifted'
[[ "$(grep -c '^SABOTAGE_TWIN ' "$bundle")" == 5 ]] || fail 'V13 sabotage-twin count drifted'

metadata="$(sed -n '1p' "$bundle")"
[[ "$metadata" == 'LOOM_KERNEL_PEER_AUTHORITY_PLAN_V13 producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 schema=13 principal_vertices=5 operations=10 observations=50 decisive_pairs=10 receiver_properties=7 action_cases=3 sabotage_twins=5' ]] ||
  fail "V13 policy metadata diverged: $metadata"

printf 'BUILT_KERNEL_PEER_AUTHORITY_PLAN_V13 path=%s language=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 engine=%s principal_vertices=5 operations=10 observations=50 decisive_pairs=10 receiver_properties=7 sabotage_twins=5 refused=25 completed=15 unavailable=10 dumpable_partial=5+5 v12_hypothesis_falsified=true output_bound_bytes=65536\n' \
  "$OUTPUT" "$ENGINE"
