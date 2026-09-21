#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-effect-root-v10.XXXXXX")"
ROOT_ONE="$TEST_ROOT/root-one"
ROOT_TWO="$TEST_ROOT/root-two"

cleanup() {
  chmod -R u+w "$TEST_ROOT" >/dev/null 2>&1 || true
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-process-witness-effect-root-v10-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

for output in "$ROOT_ONE" "$ROOT_TWO"; do
  SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_ROOT_V10_OUTPUT="$output" \
    bash "$ROOT_DIR/scripts/dev/build_loom_process_witness_effect_root_v10.sh" \
      >/dev/null
done
diff -r "$ROOT_ONE" "$ROOT_TWO" >/dev/null ||
  fail 'two source-fresh V10 root capsules differ'

expected_paths='./dev
./loom
./loom/effect-cell
./loom/effect-policy-v10.freeze.v1
./loom/payload
./loom/payload.freeze.v1
./proc
./run
./run/systemd
./run/systemd/incoming
./sys
./tmp
./var
./var/tmp'
actual_paths="$(cd "$ROOT_ONE" && find . -mindepth 1 -printf '%p\n' | sort)"
[[ "$actual_paths" == "$expected_paths" ]] || fail 'V10 capsule path set drifted'
for directory in "$ROOT_ONE" "$ROOT_ONE/loom" "$ROOT_ONE/dev" \
                 "$ROOT_ONE/proc" "$ROOT_ONE/tmp" "$ROOT_ONE/run" \
                 "$ROOT_ONE/run/systemd" "$ROOT_ONE/run/systemd/incoming" \
                 "$ROOT_ONE/sys" "$ROOT_ONE/var" "$ROOT_ONE/var/tmp"; do
  [[ "$(stat -c '%F:%a' "$directory")" == 'directory:555' ]] ||
    fail "V10 root directory mode drifted: $directory"
done
for binary in "$ROOT_ONE/loom/effect-cell" "$ROOT_ONE/loom/payload"; do
  [[ "$(stat -c '%F:%a:%h' "$binary")" == 'regular file:555:1' ]] ||
    fail "V10 root binary metadata drifted: $binary"
  if readelf -l "$binary" | grep -q 'INTERP'; then
    fail "V10 root binary retained a dynamic interpreter: $binary"
  fi
done
[[ "$(sha256sum "$ROOT_ONE/loom/payload" | cut -d ' ' -f 1)" == \
  7249748c322ede756c779904cb2d87f561ba2e17d0691314a09feaf16ca2ed4d ]] ||
  fail 'V10 capsule Sounio payload drifted'
[[ "$(sha256sum "$ROOT_ONE/loom/payload.freeze.v1" | cut -d ' ' -f 1)" == \
  624ccd7297778803eff8d9972a33d5e55fb022f9e7e37f444f0aee13c22fb4da ]] ||
  fail 'V10 capsule payload manifest drifted'
[[ "$(sha256sum "$ROOT_ONE/loom/effect-policy-v10.freeze.v1" | cut -d ' ' -f 1)" == \
  9e7f42fd4bd18fd2b5f996b279a67f46a50546a20ef6949e4dc069c16b3d0dda ]] ||
  fail 'V10 capsule policy manifest drifted'

native="$($ROOT_ONE/loom/effect-cell --selftest \
  --policy-manifest "$ROOT_ONE/loom/effect-policy-v10.freeze.v1")"
[[ "$native" == LOOM_PROCESS_WITNESS_EFFECT_POLICY_V10_SELFTEST\ PASS* &&
   "$native" == *'material_sabotages=0'* &&
   "$native" == *'material_coverage=false complete_effects=false material_execution=false'* ]] ||
  fail 'V10 capsule native policy gate diverged'

dependencies="$(ldd "$ROOT_ONE/loom/effect-cell" 2>&1 || true)
$(ldd "$ROOT_ONE/loom/payload" 2>&1 || true)"
if ! printf '%s\n' "$dependencies" | grep -Eq 'not a dynamic executable|statically linked'; then
  fail 'V10 capsule did not prove static binaries'
fi
if printf '%s\n' "$dependencies" | grep -Eqi 'python|rust'; then
  fail 'V10 capsule has a prohibited runtime dependency'
fi

printf 'sounio-loom-process-witness-effect-root-v10-selftest: PASS semantic_authority=Sounio producer=C++20+Sounio role=MATERIAL_PARITY action=9025 capsule_paths=14 cell_sha256=%s payload_sha256=7249748c322ede756c779904cb2d87f561ba2e17d0691314a09feaf16ca2ed4d static_cell=true static_payload=true deterministic=true typed_file_bounds=effect-cell-16MiB+payload-1MiB+manifests-64KiB effective_mount_truth=DynamicUser+disconnected+strict+read-only identity_typed_mounts=CAPSULE_EMPTY_BIND filesystem_authority=ROOT_HOST_MOUNTINFO systemd_mount=/run/systemd/incoming principal_readable=false principal_enumeration=forbidden empty_observer=ROOT_HOST systemd_sys_mount=/sys systemd_var_tmp=/var/tmp dev_null=host_materialization_required host_root_ownership=false root_read_only=false root_gate_required=true bootstrap_sabotage=false material_sabotages=0 material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(sha256sum "$ROOT_ONE/loom/effect-cell" | cut -d ' ' -f 1)"
