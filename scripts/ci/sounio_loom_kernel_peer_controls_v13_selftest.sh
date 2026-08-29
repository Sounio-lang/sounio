#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BUILDER="$ROOT_DIR/scripts/dev/build_loom_kernel_peer_controls_v13.sh"
CONTRACT="$ROOT_DIR/tools/loom/KERNEL_PEER_CONTROLS_V13.md"
SOURCE="$ROOT_DIR/tools/loom/src/loom_kernel_peer_controls_init_v13.cpp"
BASE_SOURCE="$ROOT_DIR/tools/loom/src/loom_kernel_peer_matrix_init_v12.cpp"
SEMANTIC_MANIFEST="$ROOT_DIR/tools/loom/kernel_peer_authority_plan_v13.freeze.v1"

fail() {
  printf 'sounio-loom-kernel-peer-controls-v13-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}
for path in "$BUILDER" "$CONTRACT" "$SOURCE" "$BASE_SOURCE" "$SEMANTIC_MANIFEST"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "required input is absent or linked: $path"
done
[[ "$(sha256sum "$SEMANTIC_MANIFEST" | cut -d ' ' -f 1)" == b3557d850ce0dc13c900f8dbb10c33f824ac25e908cb4a48dd2ef913267194c2 ]] ||
  fail 'V13 semantic freeze drifted'
[[ "$(sha256sum "$BASE_SOURCE" | cut -d ' ' -f 1)" == 54a447bd18a7d0319edda89fb01c593e5e28448c3994c4c0002c7b74795b4ab2 ]] ||
  fail 'frozen decisive-pair base source drifted'

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-kernel-peer-controls-v13-selftest.XXXXXX")"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT
ARCHIVE_A="$WORK/a.cpio.gz"
ARCHIVE_B="$WORK/b.cpio.gz"
PACKER_A="$WORK/packer-a"
PACKER_B="$WORK/packer-b"
BPF_A="$WORK/policy-a.bpf.o"
BPF_B="$WORK/policy-b.bpf.o"
SOUNIO_LOOM_KERNEL_PEER_CONTROLS_V13_OUTPUT="$ARCHIVE_A" \
SOUNIO_LOOM_KERNEL_PEER_CONTROLS_V13_PACKER_OUTPUT="$PACKER_A" \
SOUNIO_LOOM_KERNEL_PEER_CONTROLS_V13_BPF_OUTPUT="$BPF_A" \
  bash "$BUILDER" >/dev/null
SOUNIO_LOOM_KERNEL_PEER_CONTROLS_V13_OUTPUT="$ARCHIVE_B" \
SOUNIO_LOOM_KERNEL_PEER_CONTROLS_V13_PACKER_OUTPUT="$PACKER_B" \
SOUNIO_LOOM_KERNEL_PEER_CONTROLS_V13_BPF_OUTPUT="$BPF_B" \
  bash "$BUILDER" >/dev/null
for pair in "$ARCHIVE_A:$ARCHIVE_B" "$PACKER_A:$PACKER_B" "$BPF_A:$BPF_B"; do
  left="${pair%%:*}"; right="${pair#*:}"
  [[ "$(sha256sum "$left" | cut -d ' ' -f 1)" == "$(sha256sum "$right" | cut -d ' ' -f 1)" ]] ||
    fail "deterministic twin diverged: $left"
done
gzip -cd "$ARCHIVE_A" > "$WORK/base.cpio"
"$PACKER_A" --extract "$WORK/base.cpio" "$WORK/tree"
init_result="$("$WORK/tree/init" --selftest)"
for fact in 'LOOM_KERNEL_PEER_CONTROLS_INIT_V13_SELFTEST PASS' semantic_authority=Sounio action=9025 observations=50 decisive_pairs=10 controls=30 sabotage_twins=5 refused=25 completed=15 unavailable=10 dumpable_partial=5+5 v12_hypothesis_falsified=true language=C+BPF+C++20 role=MATERIAL_BOOTSTRAP transitory=true python_executed=false rust_executed=false controls_executed=false material_peer_matrix=false same_uid_peer_isolation=false action_9025_decision=DENY451 claim_ready=false; do
  [[ " $init_result " == *" $fact "* ]] || fail "init selftest omitted $fact"
done

printf 'sounio-loom-kernel-peer-controls-v13-selftest: PASS semantic_authority=Sounio action=9025 observations=50 decisive_pairs=10 controls=30 sabotage_twins=5 refused=25 completed=15 unavailable=10 dumpable_partial=5+5 v12_hypothesis_falsified=true language=C+BPF+C++20 role=MATERIAL_BOOTSTRAP transitory=true source_sha256=%s base_source_sha256=%s contract_sha256=%s archive_sha256=%s init_sha256=%s bpf_object_sha256=%s packer_sha256=%s deterministic_twins=true guest_root_traversable=true python_executed=false rust_executed=false controls_executed=false material_peer_matrix=false same_uid_peer_isolation=false action_9025_decision=DENY451 claim_ready=false\n' \
  "$(sha256sum "$SOURCE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$BASE_SOURCE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$CONTRACT" | cut -d ' ' -f 1)" \
  "$(sha256sum "$ARCHIVE_A" | cut -d ' ' -f 1)" \
  "$(sha256sum "$WORK/tree/init" | cut -d ' ' -f 1)" \
  "$(sha256sum "$BPF_A" | cut -d ' ' -f 1)" \
  "$(sha256sum "$PACKER_A" | cut -d ' ' -f 1)"
