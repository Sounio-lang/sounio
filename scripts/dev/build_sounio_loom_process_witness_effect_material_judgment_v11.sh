#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_EFFECT_MATERIAL_JUDGMENT_V11_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_EFFECT_MATERIAL_JUDGMENT_V11_ENGINE:-lean_single}"
SOURCE="$ROOT_DIR/tools/loom/process_witness_effect_material_judgment_v11_main.sio"
POLICY_MANIFEST="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v11.freeze.v1"
MATERIAL_EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-process-witness-effect-hypercube-v11-host-20260829.txt"
HOST_PRINCIPAL_EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-host-principal-cell-v1-20260828.txt"
GRANT_STACK="$ROOT_DIR/tools/loom/kernel_exec_grant_cell.stack.v1"
ACTION_9025_MANIFEST="$ROOT_DIR/tools/loom/effect_closure_authority.freeze.v1"
OUTPUT="${SOUNIO_LOOM_EFFECT_MATERIAL_JUDGMENT_V11_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-process-witness-effect-material-judgment-v11}"

fail() {
  printf 'build-sounio-loom-process-witness-effect-material-judgment-v11: FAIL: %s\n' "$*" >&2
  exit 1
}

expect_hash() {
  local path="$1" expected="$2"
  [[ -f "$path" && ! -L "$path" ]] || fail "frozen input is absent or linked: $path"
  [[ "$(sha256sum "$path" | cut -d ' ' -f 1)" == "$expected" ]] ||
    fail "frozen input hash drifted: $path"
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
expect_hash "$SOURCE" \
  48f49e4da369bc2704523692db99966d8647c64f5449b993236288d411aa2017
expect_hash "$POLICY_MANIFEST" \
  adbc7151da91bd12928cf059a4fce01de59b38096bb7bebe55be0402fab9972c
expect_hash "$MATERIAL_EVIDENCE" \
  57bc9730b0b5662a548af8271bdca6ed1651c5684c7999182e6c3d6e6ad53738
expect_hash "$HOST_PRINCIPAL_EVIDENCE" \
  01c63677ab36668c17fe4454f9792c4595350d0d091ba21407a0e5061c36c7f7
expect_hash "$GRANT_STACK" \
  1d7b8a3b1dfba1d1f9e60b5392cdf7e57a8d085cd872659feea5e333e43759b1
expect_hash "$ACTION_9025_MANIFEST" \
  c1f0cf93f8427acdf794246a11c3551e265a09be12a3cd000bad25b707e8ca91
grep -Fxq 'stage=SEMANTICS_FROZEN' "$POLICY_MANIFEST" ||
  fail 'V11 policy semantics are not frozen'
grep -Fxq 'material_hypercube=true' "$MATERIAL_EVIDENCE" ||
  fail 'V11 material hypercube has not passed'
grep -Fxq 'same_uid_peer_isolation=false' "$HOST_PRINCIPAL_EVIDENCE" ||
  fail 'host principal evidence no longer carries the frozen negative fact'
grep -Fxq 'same_uid_peer_isolation=false' "$GRANT_STACK" ||
  fail 'grant stack no longer carries the frozen negative fact'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-effect-material-judgment-v11.XXXXXX")"
trap 'rm -rf "$work"' EXIT
compiled="$work/sounio-loom-process-witness-effect-material-judgment-v11"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$SOURCE" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the Sounio material judgment executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

fixtures="$(printf '0\n' | "$OUTPUT")"
metadata="$(printf '%s\n' "$fixtures" | sed -n '1p')"
[[ "$metadata" == 'LOOM_PROCESS_WITNESS_EFFECT_MATERIAL_JUDGMENT_V11 producer=Sounio role=SEMANTIC_AUTHORITY action=9025 material_evidence_sha256=57bc9730b0b5662a548af8271bdca6ed1651c5684c7999182e6c3d6e6ad53738 certificate_bundle_sha256=1c92fcd7c97a5df4e8316b722f769f6777ea5979edcd09c207e88f9930f8d3dd cases=9' ]] ||
  fail "material judgment metadata diverged: $metadata"
[[ "$(printf '%s\n' "$fixtures" | grep -c '^CASE ')" == 9 ]] ||
  fail 'Sounio material judgment fixture count drifted'
[[ "$(printf '%s\n' "$fixtures" | grep -c '^EXPECTED_ACTION_9025 ')" == 1 ]] ||
  fail 'Sounio material judgment omitted the expected action-9025 boundary'

printf 'BUILT_PROCESS_WITNESS_EFFECT_MATERIAL_JUDGMENT_V11 path=%s language=Sounio role=SEMANTIC_AUTHORITY action=9025 engine=%s cases=9 material_hypercube=true same_uid_peer_isolation=false\n' \
  "$OUTPUT" "$ENGINE"
