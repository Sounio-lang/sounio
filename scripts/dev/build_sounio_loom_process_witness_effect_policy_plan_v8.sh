#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V8_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V8_ENGINE:-lean_single}"
SOURCE="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v8_main.sio"
GARDEN="$ROOT_DIR/tools/loom/GARDEN_PROCESS_WITNESS_EFFECT_POLICY_V8.md"
V7_MANIFEST="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v7.freeze.v1"
FAILURE_EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-process-witness-effect-root-v7-host-attempt-v1-20260828.txt"
OUTPUT="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V8_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-process-witness-effect-policy-plan-v8}"

fail() {
  printf 'build-sounio-loom-process-witness-effect-policy-plan-v8: FAIL: %s\n' "$*" >&2
  exit 1
}

expect_regular_hash() {
  local path="$1" expected="$2"
  [[ -f "$path" && ! -L "$path" ]] || fail "frozen input is absent or linked: $path"
  [[ "$(sha256sum "$path" | cut -d ' ' -f 1)" == "$expected" ]] ||
    fail "frozen input hash drifted: $path"
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
expect_regular_hash "$GARDEN" \
  1a5cce39dd307af5f5a21ce8d5ace90994ad0d73fe911e121c3b7650ed37d533
expect_regular_hash "$V7_MANIFEST" \
  cc7ca5a17babb43e145678879607b2804bdbfc66665f994b73f8649c86e420d9
expect_regular_hash "$FAILURE_EVIDENCE" \
  a4e9bab136988e6034a775e347ecc6642d7624dedfbf35b3e52d5b14236929bb
expect_regular_hash "$SOURCE" \
  e512d0b465b170b9f9d50022f8eac5d021228fae6ba944d64b948c902831dc30
grep -Fxq 'stage=SEMANTICS_FROZEN' "$V7_MANIFEST" || fail 'V7 policy is not frozen'
grep -Fxq 'falsifying_status=70/SOFTWARE' "$FAILURE_EVIDENCE" ||
  fail 'V7 post-exec refusal is absent'
grep -Fxq 'falsifying_path=/loom/effect-cell' "$FAILURE_EVIDENCE" ||
  fail 'V7 bounded object path drifted'
grep -Fxq 'falsifying_configured_max_bytes=131072' "$FAILURE_EVIDENCE" ||
  fail 'V7 generic file bound drifted'
grep -Fxq 'v7_file_bound_assignment_sufficient=false' "$FAILURE_EVIDENCE" ||
  fail 'V7 file-bound insufficiency is absent'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-effect-policy-v8.XXXXXX")"
trap 'rm -rf "$work"' EXIT
compiled="$work/sounio-loom-process-witness-effect-policy-plan-v8"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$SOURCE" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the Sounio V8 executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

metadata="$($OUTPUT | sed -n '1p')"
[[ "$metadata" == \
  'LOOM_PROCESS_WITNESS_EFFECT_POLICY_PLAN_V8 producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 schema=8 object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE families=12 treatments=12 sabotages=12 bootstrap_cases=4 allowed_syscalls=4' ]] ||
  fail "V8 policy metadata diverged: $metadata"

printf 'BUILT_PROCESS_WITNESS_EFFECT_POLICY_PLAN_V8 path=%s language=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 engine=%s object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE families=12 treatments=12 sabotages=12 bootstrap_cases=4 allowed_syscalls=4\n' \
  "$OUTPUT" "$ENGINE"
