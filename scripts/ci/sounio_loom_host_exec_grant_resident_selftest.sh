#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-host-exec-grant-resident.XXXXXX")"
BROKER_ONE="$TEST_ROOT/broker-one"
BROKER_TWO="$TEST_ROOT/broker-two"
RESIDENT_ONE="$TEST_ROOT/resident-one"
RESIDENT_TWO="$TEST_ROOT/resident-two"
CURRENT_FRAME="$TEST_ROOT/current.frame"
PYTHON_FRAME="$TEST_ROOT/python.frame"
EXEC_GRANT_MANIFEST="$ROOT_DIR/tools/loom/kernel_exec_grant_cell_authority.freeze.v1"
RESIDENT_MANIFEST="$ROOT_DIR/tools/loom/resident_membrane.runtime.v4"
GARDEN="$ROOT_DIR/tools/loom/GARDEN_HOST_EXEC_GRANT_RESIDENT_ATTACHMENT_V1.md"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-host-exec-grant-resident-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

run_refusal() {
  local label="$1"
  shift
  local output status
  set +e
  output="$($@ 2>&1)"
  status=$?
  set -e
  [[ $status -ne 0 ]] || fail "$label unexpectedly succeeded"
  printf '%s' "$output"
}

for path in "$EXEC_GRANT_MANIFEST" "$RESIDENT_MANIFEST" "$GARDEN"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "required input is absent or linked: $path"
done
[[ "$(sha256sum "$EXEC_GRANT_MANIFEST" | cut -d ' ' -f 1)" == \
  8687d889e08f69190daaf3cdbee02741cde3ce62f136ba63df1fa9c2ccb0d051 ]] ||
  fail 'frozen action 9030 manifest drifted'
[[ "$(sha256sum "$RESIDENT_MANIFEST" | cut -d ' ' -f 1)" == \
  f61c93a3aefdbab792ed757faddf778017d34e0fa6bed97c565b56fe3147d473 ]] ||
  fail 'frozen resident v4 manifest drifted'

SOUNIO_LOOM_KERNEL_PRINCIPAL_BROKER_OUTPUT="$BROKER_ONE" \
  bash "$ROOT_DIR/scripts/dev/build_loom_kernel_principal_broker.sh" >/dev/null
SOUNIO_LOOM_KERNEL_PRINCIPAL_BROKER_OUTPUT="$BROKER_TWO" \
  bash "$ROOT_DIR/scripts/dev/build_loom_kernel_principal_broker.sh" >/dev/null
cmp "$BROKER_ONE" "$BROKER_TWO" || fail 'two broker builds differ'

SOUNIO_LOOM_RESIDENT_MEMBRANE_V4_OUTPUT="$RESIDENT_ONE" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_resident_membrane_v4.sh" >/dev/null
SOUNIO_LOOM_RESIDENT_MEMBRANE_V4_OUTPUT="$RESIDENT_TWO" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_resident_membrane_v4.sh" >/dev/null
cmp "$RESIDENT_ONE" "$RESIDENT_TWO" || fail 'two resident v4 builds differ'
[[ "$(sha256sum "$RESIDENT_ONE" | cut -d ' ' -f 1)" == \
  "$(sed -n 's/^runtime_sha256=//p' "$RESIDENT_MANIFEST")" ]] ||
  fail 'rebuilt resident v4 runtime is not the frozen runtime'

parent_9029='1636926980 3205986131 3323207532 3505413428 706242987 2411760920 1929815169 3727939342'
parent_9021='3497534264 556131944 3943529214 1565657389 3821375173 3204015455 2733765994 2625951936'
parent_9022='4125506095 3601417934 2711931735 20635855 2708941890 3284947684 758124027 2068177262'
one='1 1 1 1 1 1 1 1'
bindings="$parent_9029 $parent_9021 $parent_9022 $one $one $one $one $one $one"
transition='1 0 1 2 3 4 5 6 7 100 50'
identity='1 1 1 1 1 1 1 1'
peer='1 1 1 1 1 1 1 1 1'
shape='1 1 1 1 1 1 1 1'
consumption='1 1 1 1 1 1 1'
revocation='1 1 1 1 1 1 1'
extinction='0 0 0 0 1'
outcome='0 0 0 0 0 0 0 0'
authority='1 1 1 1 1 1 1'
evidence='1 1 11 11'
current="9030 3 1 $transition 1 0 0 0 0 0 0 0 $identity $peer $shape $consumption $revocation $extinction $outcome $authority $evidence $bindings"
python="9030 3 1 $transition 1 1 0 0 0 1 0 0 $identity $peer $shape $consumption $revocation $extinction $outcome 0 0 0 1 1 1 1 $evidence $bindings"
printf '%s\n' "$current" > "$CURRENT_FRAME"
printf '%s\n' "$python" > "$PYTHON_FRAME"

steady="$($BROKER_ONE --selftest-exec-grant-resident \
  --exec-grant-manifest "$EXEC_GRANT_MANIFEST" \
  --resident-manifest "$RESIDENT_MANIFEST" --resident-runtime "$RESIDENT_ONE" \
  --frame "$CURRENT_FRAME" --second-frame "$PYTHON_FRAME")"
[[ "$steady" == LOOM_KERNEL_PRINCIPAL_BROKER_RESIDENT_SELFTEST\ PASS* ]] ||
  fail "resident steady-state gate failed: $steady"
[[ "$steady" == *'sequences=1,2 current=DENY491 python=DENY499'* ]] ||
  fail 'resident decisions or monotonic sequence diverged'
[[ "$steady" == *'process_identity=stable generation_poisoned=false'* ]] ||
  fail 'resident process identity was not stable'
[[ "$steady" == *'launch_open=false material_grant=false material_execution=false exec_attached=false' ]] ||
  fail 'resident selftest opened execution'

faults="$($BROKER_ONE --selftest-exec-grant-resident-faults \
  --exec-grant-manifest "$EXEC_GRANT_MANIFEST" \
  --resident-manifest "$RESIDENT_MANIFEST" --resident-runtime "$RESIDENT_ONE" \
  --frame "$CURRENT_FRAME")"
[[ "$faults" == \
  'LOOM_KERNEL_PRINCIPAL_BROKER_RESIDENT_FAULT_SELFTEST PASS death=poisoned timeout=poisoned malformed=poisoned restart_within_generation=false replay_after_poison=refused launch_open=false material_grant=false material_execution=false exec_attached=false' ]] ||
  fail "resident destructive gate failed: $faults"

TAMPERED_EXEC_GRANT="$TEST_ROOT/exec-grant-tampered.v1"
TAMPERED_RESIDENT_MANIFEST="$TEST_ROOT/resident-tampered.v4"
TAMPERED_RESIDENT_RUNTIME="$TEST_ROOT/resident-tampered"
cp "$EXEC_GRANT_MANIFEST" "$TAMPERED_EXEC_GRANT"
cp "$RESIDENT_MANIFEST" "$TAMPERED_RESIDENT_MANIFEST"
cp "$RESIDENT_ONE" "$TAMPERED_RESIDENT_RUNTIME"
printf '\n' >> "$TAMPERED_EXEC_GRANT"
printf '\n' >> "$TAMPERED_RESIDENT_MANIFEST"
printf 'X' >> "$TAMPERED_RESIDENT_RUNTIME"

exec_manifest_refusal="$(run_refusal exec-manifest-tamper "$BROKER_ONE" \
  --selftest-exec-grant-resident --exec-grant-manifest "$TAMPERED_EXEC_GRANT" \
  --resident-manifest "$RESIDENT_MANIFEST" --resident-runtime "$RESIDENT_ONE" \
  --frame "$CURRENT_FRAME" --second-frame "$PYTHON_FRAME")"
[[ "$exec_manifest_refusal" == *'frozen action 9030 manifest hash mismatch'* ]] ||
  fail 'tampered action 9030 manifest was not refused before resident start'

resident_manifest_refusal="$(run_refusal resident-manifest-tamper "$BROKER_ONE" \
  --selftest-exec-grant-resident --exec-grant-manifest "$EXEC_GRANT_MANIFEST" \
  --resident-manifest "$TAMPERED_RESIDENT_MANIFEST" --resident-runtime "$RESIDENT_ONE" \
  --frame "$CURRENT_FRAME" --second-frame "$PYTHON_FRAME")"
[[ "$resident_manifest_refusal" == *'frozen resident v4 manifest hash mismatch'* ]] ||
  fail 'tampered resident manifest was not refused before resident start'

resident_runtime_refusal="$(run_refusal resident-runtime-tamper "$BROKER_ONE" \
  --selftest-exec-grant-resident --exec-grant-manifest "$EXEC_GRANT_MANIFEST" \
  --resident-manifest "$RESIDENT_MANIFEST" --resident-runtime "$TAMPERED_RESIDENT_RUNTIME" \
  --frame "$CURRENT_FRAME" --second-frame "$PYTHON_FRAME")"
[[ "$resident_runtime_refusal" == *'Sounio resident v4 runtime hash mismatch'* ]] ||
  fail 'tampered resident runtime was not refused before resident start'

dependencies="$({ ldd "$BROKER_ONE" || true; ldd "$RESIDENT_ONE" 2>&1 || true; })"
if printf '%s\n' "$dependencies" | grep -Eqi 'python|rust'; then
  fail 'resident attachment has a prohibited Python or Rust runtime dependency'
fi

grep -Fq 'START -> REQUEST -> route 5/action 9030 -> RESPONSE' "$GARDEN" ||
  fail 'Garden omitted the resident action 9024 envelope'
grep -Fq 'A semantic `DENY491` must not write or release' "$GARDEN" ||
  fail 'Garden omitted DENY491 non-release'

printf 'sounio-loom-host-exec-grant-resident-selftest: PASS semantic_authority=Sounio material_producer=C++20+resident-Sounio material_role=MATERIAL_PARITY actions=9024+9030 exec_grant_manifest_sha256=%s resident_manifest_sha256=%s broker_sha256=%s resident_runtime_sha256=%s rebuilds=2 process_identity=stable sequence=monotonic current=DENY491 python=DENY499 exec_manifest_tamper=refused resident_manifest_tamper=refused resident_runtime_tamper=refused death=poisoned timeout=poisoned malformed=poisoned replay_after_poison=refused runtime_dependencies=clean resident_action_9030_attached=local-gate decision_transport_material=true host_socket_attached=false launch_open=false material_grant=false material_execution=false exec_attached=false parity_open=false claim_ready=false\n' \
  "$(sha256sum "$EXEC_GRANT_MANIFEST" | cut -d ' ' -f 1)" \
  "$(sha256sum "$RESIDENT_MANIFEST" | cut -d ' ' -f 1)" \
  "$(sha256sum "$BROKER_ONE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$RESIDENT_ONE" | cut -d ' ' -f 1)"
