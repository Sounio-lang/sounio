#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-kernel-peer-microhost-v12.XXXXXX")"
ARCHIVE_ONE="$TEST_ROOT/microhost-one.cpio.gz"
ARCHIVE_TWO="$TEST_ROOT/microhost-two.cpio.gz"
EXTRACTED="$TEST_ROOT/extracted"
RAW_ARCHIVE="$TEST_ROOT/microhost.cpio"
PACKER_ONE="$TEST_ROOT/packer-one"
PACKER_TWO="$TEST_ROOT/packer-two"
SOURCE="$ROOT_DIR/tools/loom/src/loom_kernel_peer_microhost_init_v12.cpp"
PACKER_SOURCE="$ROOT_DIR/tools/loom/src/loom_newc_packer.cpp"
CONTRACT="$ROOT_DIR/tools/loom/KVM_BPF_LSM_MICROHOST_V12.md"
HOST_GATE="$ROOT_DIR/scripts/ci/sounio_loom_kernel_peer_microhost_v12_host_gate.sh"
HOST_PROBE="$ROOT_DIR/scripts/dev/run_loom_kernel_peer_microhost_v12_host_probe.sh"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-kernel-peer-microhost-v12-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

for path in "$SOURCE" "$PACKER_SOURCE" "$CONTRACT" "$HOST_GATE" "$HOST_PROBE"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "required input is absent or linked: $path"
done
bash "$ROOT_DIR/scripts/ci/sounio_loom_kernel_peer_backend_discovery_v12_freeze_selftest.sh" >/dev/null

SOUNIO_LOOM_KERNEL_PEER_MICROHOST_V12_OUTPUT="$ARCHIVE_ONE" \
  SOUNIO_LOOM_NEWC_PACKER_OUTPUT="$PACKER_ONE" \
  bash "$ROOT_DIR/scripts/dev/build_loom_kernel_peer_microhost_v12.sh" >/dev/null
SOUNIO_LOOM_KERNEL_PEER_MICROHOST_V12_OUTPUT="$ARCHIVE_TWO" \
  SOUNIO_LOOM_NEWC_PACKER_OUTPUT="$PACKER_TWO" \
  bash "$ROOT_DIR/scripts/dev/build_loom_kernel_peer_microhost_v12.sh" >/dev/null
cmp "$ARCHIVE_ONE" "$ARCHIVE_TWO" || fail 'two microhost initramfs builds differ'
cmp "$PACKER_ONE" "$PACKER_TWO" || fail 'two newc packer builds differ'
[[ "$($PACKER_ONE --selftest)" == 'LOOM_NEWC_PACKER_SELFTEST PASS language=C++20 role=MATERIAL_PACKER deterministic=true symlinks=refused traversal=refused python_executed=false rust_executed=false' ]] ||
  fail 'newc packer selftest failed'

mkdir "$EXTRACTED"
gzip -dc "$ARCHIVE_ONE" > "$RAW_ARCHIVE"
"$PACKER_ONE" --extract "$RAW_ARCHIVE" "$EXTRACTED"
[[ -x "$EXTRACTED/init" && ! -L "$EXTRACTED/init" ]] || fail 'microhost init is absent or linked'
[[ "$(stat -c '%a' "$EXTRACTED/init")" == 755 ]] || fail 'microhost init mode is not 0755'
if ldd "$EXTRACTED/init" 2>&1 | grep -vq 'not a dynamic executable'; then
  fail 'microhost init is not static'
fi
if ldd "$PACKER_ONE" 2>&1 | grep -vq 'not a dynamic executable'; then
  fail 'newc packer is not static'
fi
if find "$EXTRACTED" -type f -print | grep -Eqi 'python|rust'; then
  fail 'microhost archive contains a forbidden runtime'
fi
selftest="$($EXTRACTED/init --selftest)"
[[ "$selftest" == 'LOOM_KERNEL_PEER_MICROHOST_INIT_V12_SELFTEST PASS language=C++20 role=MATERIAL_BOOTSTRAP transitory=true semantic_authority=Sounio action=9025 disk=none network=none python_executed=false rust_executed=false material_peer_matrix=false same_uid_peer_isolation=false claim_ready=false' ]] ||
  fail "microhost init selftest failed: $selftest"
grep -Fq 'guest disk          = none' "$CONTRACT" || fail 'contract omits disk absence'
grep -Fq 'guest network       = none' "$CONTRACT" || fail 'contract omits network absence'
grep -Fq 'Activating BPF LSM is not evidence' "$CONTRACT" || fail 'contract promotes BPF activation'

printf 'sounio-loom-kernel-peer-microhost-v12-selftest: PASS semantic_authority=Sounio action=9025 material_producer=C++20 material_role=MATERIAL_BOOTSTRAP transitory=true source_sha256=%s packer_source_sha256=%s contract_sha256=%s init_sha256=%s packer_sha256=%s archive_sha256=%s host_gate_sha256=%s host_probe_sha256=%s rebuilds=2 archive_reproducible=true init_static=true packer_static=true guest_disk=none guest_network=none python_executed=false rust_executed=false bpf_lsm=unmeasured material_peer_matrix=false same_uid_peer_isolation=false action_9025=DENY451 claim_ready=false\n' \
  "$(sha256sum "$SOURCE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$PACKER_SOURCE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$CONTRACT" | cut -d ' ' -f 1)" \
  "$(sha256sum "$EXTRACTED/init" | cut -d ' ' -f 1)" \
  "$(sha256sum "$PACKER_ONE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$ARCHIVE_ONE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$HOST_GATE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$HOST_PROBE" | cut -d ' ' -f 1)"
