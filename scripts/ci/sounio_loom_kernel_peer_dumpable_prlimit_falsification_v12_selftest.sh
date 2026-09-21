#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BUILDER="$ROOT_DIR/scripts/dev/build_loom_kernel_peer_dumpable_prlimit_falsification_v12.sh"
CONTRACT="$ROOT_DIR/tools/loom/KERNEL_PEER_DUMPABLE_PRLIMIT_FALSIFICATION_V12.md"
INIT_SOURCE="$ROOT_DIR/tools/loom/src/loom_kernel_peer_dumpable_prlimit_falsification_init_v12.cpp"
BASE_SOURCE="$ROOT_DIR/tools/loom/src/loom_kernel_peer_matrix_init_v12.cpp"
SEMANTIC_MANIFEST="$ROOT_DIR/tools/loom/kernel_peer_authority_plan_v12.freeze.v1"
MATRIX_MANIFEST="$ROOT_DIR/tools/loom/kernel_peer_matrix_v12.freeze.v1"

fail() {
  printf 'sounio-loom-kernel-peer-dumpable-prlimit-falsification-v12-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}
for path in "$BUILDER" "$CONTRACT" "$INIT_SOURCE" "$BASE_SOURCE" \
  "$SEMANTIC_MANIFEST" "$MATRIX_MANIFEST"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "required input is absent or linked: $path"
done
[[ "$(sha256sum "$SEMANTIC_MANIFEST" | cut -d ' ' -f 1)" == daaf9b056bde02078163849fdcc0f0b3acb96a357741c70e3d23c688c2bd0b30 ]] ||
  fail 'V12 Sounio semantic freeze drifted'
[[ "$(sha256sum "$MATRIX_MANIFEST" | cut -d ' ' -f 1)" == 1692782657cbe6fe7a548b6f11d4d542d24fe05569686d536a4c69af0775cd75 ]] ||
  fail 'V12 material matrix freeze drifted'
[[ "$(sha256sum "$BASE_SOURCE" | cut -d ' ' -f 1)" == 54a447bd18a7d0319edda89fb01c593e5e28448c3994c4c0002c7b74795b4ab2 ]] ||
  fail 'frozen peer-matrix base source drifted'
for forbidden in python python3 python2 rustc cargo; do
  ! grep -E "(^|[^[:alnum:]_])${forbidden}([^[:alnum:]_]|$)" "$BUILDER" \
    "$INIT_SOURCE" "$CONTRACT" >/dev/null || fail "forbidden runtime named: $forbidden"
done

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-kernel-peer-dumpable-prlimit-v12-selftest.XXXXXX")"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT
ARCHIVE_A="$WORK/a.cpio.gz"
ARCHIVE_B="$WORK/b.cpio.gz"
PACKER_A="$WORK/packer-a"
PACKER_B="$WORK/packer-b"
SOUNIO_LOOM_KERNEL_PEER_DUMPABLE_PRLIMIT_V12_OUTPUT="$ARCHIVE_A" \
SOUNIO_LOOM_KERNEL_PEER_DUMPABLE_PRLIMIT_V12_PACKER_OUTPUT="$PACKER_A" \
  bash "$BUILDER" >/dev/null
SOUNIO_LOOM_KERNEL_PEER_DUMPABLE_PRLIMIT_V12_OUTPUT="$ARCHIVE_B" \
SOUNIO_LOOM_KERNEL_PEER_DUMPABLE_PRLIMIT_V12_PACKER_OUTPUT="$PACKER_B" \
  bash "$BUILDER" >/dev/null
[[ "$(sha256sum "$ARCHIVE_A" | cut -d ' ' -f 1)" == "$(sha256sum "$ARCHIVE_B" | cut -d ' ' -f 1)" ]] ||
  fail 'deterministic initramfs twins diverged'
[[ "$(sha256sum "$PACKER_A" | cut -d ' ' -f 1)" == "$(sha256sum "$PACKER_B" | cut -d ' ' -f 1)" ]] ||
  fail 'deterministic packer twins diverged'
gzip -cd "$ARCHIVE_A" > "$WORK/base.cpio"
"$PACKER_A" --extract "$WORK/base.cpio" "$WORK/tree"
init_result="$("$WORK/tree/init" --selftest)"
for fact in 'LOOM_KERNEL_PEER_DUMPABLE_PRLIMIT_FALSIFICATION_INIT_V12_SELFTEST PASS' operation=9 syscall=prlimit64 frozen_expected=REFUSED_BEFORE_EFFECT observations=1 language=C++20 role=MATERIAL_BOOTSTRAP transitory=true semantic_authority=Sounio python_executed=false rust_executed=false controls_executed=false material_peer_matrix=false same_uid_peer_isolation=false action_9025_decision=DENY451 claim_ready=false; do
  [[ " $init_result " == *" $fact "* ]] || fail "init selftest omitted $fact"
done

printf 'sounio-loom-kernel-peer-dumpable-prlimit-falsification-v12-selftest: PASS semantic_authority=Sounio action=9025 operation=9 syscall=prlimit64 frozen_expected=REFUSED_BEFORE_EFFECT observations=1 language=C++20 role=MATERIAL_BOOTSTRAP transitory=true source_sha256=%s base_source_sha256=%s contract_sha256=%s archive_sha256=%s init_sha256=%s deterministic_twins=true python_executed=false rust_executed=false v12_hypothesis_falsified=unmeasured controls_executed=false material_peer_matrix=false same_uid_peer_isolation=false action_9025_decision=DENY451 claim_ready=false\n' \
  "$(sha256sum "$INIT_SOURCE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$BASE_SOURCE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$CONTRACT" | cut -d ' ' -f 1)" \
  "$(sha256sum "$ARCHIVE_A" | cut -d ' ' -f 1)" \
  "$(sha256sum "$WORK/tree/init" | cut -d ' ' -f 1)"
