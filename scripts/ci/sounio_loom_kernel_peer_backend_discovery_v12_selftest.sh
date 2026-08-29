#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-kernel-peer-backend-discovery-v12.XXXXXX")"
BINARY_ONE="$TEST_ROOT/backend-discovery-one"
BINARY_TWO="$TEST_ROOT/backend-discovery-two"
SOURCE="$ROOT_DIR/tools/loom/src/loom_kernel_peer_backend_discovery_v12.cpp"
PROFILE="$ROOT_DIR/tools/loom/apparmor/loom-kernel-peer-backend-discovery-v12.profile"
SEMANTIC_MANIFEST="$ROOT_DIR/tools/loom/kernel_peer_authority_plan_v12.freeze.v1"
HOST_GATE="$ROOT_DIR/scripts/ci/sounio_loom_kernel_peer_backend_discovery_v12_host_gate.sh"
HOST_PROBE="$ROOT_DIR/scripts/dev/run_loom_kernel_peer_backend_discovery_v12_host_probe.sh"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-kernel-peer-backend-discovery-v12-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

for path in "$SOURCE" "$PROFILE" "$SEMANTIC_MANIFEST" "$HOST_GATE" "$HOST_PROBE"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "required input is absent or linked: $path"
done
[[ "$(sha256sum "$SEMANTIC_MANIFEST" | cut -d ' ' -f 1)" == daaf9b056bde02078163849fdcc0f0b3acb96a357741c70e3d23c688c2bd0b30 ]] ||
  fail 'V12 semantic freeze drifted'
bash "$ROOT_DIR/scripts/ci/sounio_loom_kernel_peer_authority_plan_v12_freeze_selftest.sh" >/dev/null

SOUNIO_LOOM_KERNEL_PEER_BACKEND_DISCOVERY_V12_OUTPUT="$BINARY_ONE" \
  bash "$ROOT_DIR/scripts/dev/build_loom_kernel_peer_backend_discovery_v12.sh" >/dev/null
SOUNIO_LOOM_KERNEL_PEER_BACKEND_DISCOVERY_V12_OUTPUT="$BINARY_TWO" \
  bash "$ROOT_DIR/scripts/dev/build_loom_kernel_peer_backend_discovery_v12.sh" >/dev/null
cmp "$BINARY_ONE" "$BINARY_TWO" || fail 'two C++20 backend-discovery builds differ'
[[ "$(stat -c '%a' "$BINARY_ONE")" == 755 ]] || fail 'backend-discovery binary mode is not 0755'
[[ ! -u "$BINARY_ONE" && ! -g "$BINARY_ONE" ]] || fail 'backend-discovery binary acquired set-id privilege'

if ldd "$BINARY_ONE" 2>&1 | grep -Eqi 'python|rust'; then
  fail 'backend-discovery binary has a forbidden runtime dependency'
fi
if strings "$BINARY_ONE" | grep -Eq 'SOUNIO_(KERNEL_PEER|EFFECT_CLOSURE)_(ALLOW|DENY)'; then
  fail 'C++20 backend discovery copied Sounio semantic result strings'
fi

selftest="$($BINARY_ONE --selftest)"
[[ "$selftest" == 'LOOM_KERNEL_PEER_BACKEND_DISCOVERY_V12_SELFTEST PASS language=C++20 role=MATERIAL_DISCOVERY transitory=true semantic_authority=Sounio action=9025 operations=kill_SIGTERM+prlimit64 semantic_results_encoded=false python_executed=false rust_executed=false' ]] ||
  fail "native selftest failed: $selftest"

grep -Fxq 'profile sounio-loom-kernel-peer-backend-discovery-v12 flags=(attach_disconnected,mediate_deleted) {' "$PROFILE" ||
  fail 'AppArmor profile identity drifted'
grep -Fq 'deny signal (receive) set=(term) peer=unconfined,' "$PROFILE" ||
  fail 'AppArmor profile omits the live signal control'
grep -Fq 'deny ptrace (tracedby) peer=unconfined,' "$PROFILE" ||
  fail 'AppArmor profile omits the ptrace receiver rule'
if grep -Eq 'prlimit.*(allow|deny)|task_prlimit' "$PROFILE"; then
  fail 'AppArmor profile pretends to mediate prlimit64'
fi

printf 'sounio-loom-kernel-peer-backend-discovery-v12-selftest: PASS semantic_authority=Sounio action=9025 material_producer=C++20 material_role=MATERIAL_DISCOVERY transitory=true semantic_manifest_sha256=daaf9b056bde02078163849fdcc0f0b3acb96a357741c70e3d23c688c2bd0b30 source_sha256=%s profile_sha256=%s binary_sha256=%s host_gate_sha256=%s host_probe_sha256=%s rebuilds=2 operations=kill_SIGTERM+prlimit64 semantic_results_encoded=false python_executed=false rust_executed=false backend_discovery=unmeasured native_discovery_bytes_created=true material_peer_matrix=false same_uid_peer_isolation=false action_9025=DENY451 claim_ready=false\n' \
  "$(sha256sum "$SOURCE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$PROFILE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$BINARY_ONE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$HOST_GATE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$HOST_PROBE" | cut -d ' ' -f 1)"
