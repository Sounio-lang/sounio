#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GIT_COMMON_DIR="$(git -C "$ROOT_DIR" rev-parse --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"

APPLE_GARDEN="$ROOT_DIR/docs/internal/garden/seeds/2026-08-27-pireus-xor-lowering-apple-silicon.md"
DGX_GARDEN="$ROOT_DIR/docs/internal/garden/seeds/2026-08-27-pireus-xor-lowering-dgx.md"
LOWERING_SOURCE="$ROOT_DIR/stdlib/hardware/pireus/xor_lowering_legality.sio"
LOWERING_SEMANTICS="$ROOT_DIR/docs/research/pireus_xor_lowering_legality_semantics.md"
LOWERING_RECEIPT="$ROOT_DIR/docs/research/receipts/pireus_xor_lowering_legality_20260827.md"
AARCH_SOURCE="$ROOT_DIR/stdlib/hardware/pireus/aarchmrs_import.sio"
AARCH_SEMANTICS="$ROOT_DIR/docs/research/pireus_aarchmrs_tbl_import_semantics.md"
AARCH_RECEIPT="$ROOT_DIR/docs/research/receipts/pireus_aarchmrs_tbl_import_20260827.md"
PTX_SOURCE="$ROOT_DIR/stdlib/hardware/pireus/ptx_import.sio"
PTX_SEMANTICS="$ROOT_DIR/docs/research/pireus_ptx_prmt_import_semantics.md"
PTX_RECEIPT="$ROOT_DIR/docs/research/receipts/pireus_ptx_prmt_import_20260827.md"

APPLE_MODULE="$ROOT_DIR/stdlib/hardware/pireus/apple_a64_tbl_lowering.sio"
APPLE_EXAMPLE="$ROOT_DIR/examples/pireus_apple_a64_tbl_lowering.sio"
APPLE_SEMANTICS="$ROOT_DIR/docs/research/pireus_apple_a64_tbl_lowering_semantics.md"
APPLE_RECEIPT="$ROOT_DIR/docs/research/receipts/pireus_apple_a64_tbl_lowering_20260827.md"
DGX_MODULE="$ROOT_DIR/stdlib/hardware/pireus/dgx_ptx_shfl_lowering.sio"
DGX_EXAMPLE="$ROOT_DIR/examples/pireus_dgx_ptx_shfl_lowering.sio"
DGX_SEMANTICS="$ROOT_DIR/docs/research/pireus_dgx_ptx_shfl_lowering_semantics.md"
DGX_RECEIPT="$ROOT_DIR/docs/research/receipts/pireus_dgx_ptx_shfl_lowering_20260827.md"

A64_DIR="${PIREUS_A64_XML_DIR:-/tmp/pireus-a64-isa-2025-12/ISA_A64_xml_A_profile-2025-12}"
PTX_PREFIX="${PIREUS_PTX_CHUNK_PREFIX:-/tmp/pireus-ptx-13.2.0/chunks-v1/part-}"

fail() {
  printf 'pireus-canonical-target-lowering-sounio: FAIL: %s\n' "$*" >&2
  exit 1
}

sha_file() {
  sha256sum "$1" | cut -d' ' -f1
}

sha_text() {
  printf '%s\n' "$1" | sha256sum | cut -d' ' -f1
}

require_hash() {
  local path="$1"
  local expected="$2"
  [[ -f "$path" ]] || fail "missing artifact: $path"
  [[ "$(sha_file "$path")" == "$expected" ]] || fail "hash drift: $path"
}

sha_limbs() {
  local hex="$1"
  local i part
  local limbs=()
  for ((i = 0; i < 8; i++)); do
    part="${hex:$((i * 8)):8}"
    limbs+=("$((16#${part}))")
  done
  printf '%s' "${limbs[*]}"
}

preaction_frame() {
  local source_sha="$1"
  local command_sha="$2"
  local zero='0 0 0 0 0 0 0 0'
  printf '9020 1 2 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "$source_sha")" "$zero" "$zero" \
    "$(sha_limbs 2ce5194cdc517de8d7a0063e09d4f4e7b6b5701a23fa1031a3ec9e8f56486b6e)" \
    "$(sha_limbs b6326139297e9ba59d82e208a2404b01e3d57445357a5e803c51c845dd388db0)" \
    "$(sha_limbs "$command_sha")" "$zero" "$zero"
}

freeze_frame() {
  local source_sha="$1"
  local semantics_sha="$2"
  local parent_sha="$3"
  local command_sha="$4"
  local result_sha="$5"
  local zero='0 0 0 0 0 0 0 0'
  printf '9020 2 3 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "$source_sha")" "$(sha_limbs "$semantics_sha")" \
    "$(sha_limbs "$parent_sha")" \
    "$(sha_limbs 2ce5194cdc517de8d7a0063e09d4f4e7b6b5701a23fa1031a3ec9e8f56486b6e)" \
    "$(sha_limbs b6326139297e9ba59d82e208a2404b01e3d57445357a5e803c51c845dd388db0)" \
    "$(sha_limbs "$command_sha")" "$(sha_limbs "$result_sha")" "$zero"
}

authorize() {
  local expected_frame_sha="$1"
  local expected_decision="$2"
  local frame="$3"
  local decision
  [[ "$(sha_text "$frame")" == "$expected_frame_sha" ]] ||
    fail "Loom frame drift: expected $expected_frame_sha"
  decision="$(printf '%s\n' "$frame" | "$GUARDIAN")"
  [[ "$decision" == "$expected_decision" ]] ||
    fail "Loom decision mismatch: $decision"
}

[[ -x "$GUARDIAN" ]] || fail "native Sounio Loom guardian unavailable: $GUARDIAN"
require_hash "$ROOT_DIR/bin/souc" ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008
require_hash "$ROOT_DIR/bin/souc-lean-single-x86_64" 6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2

require_hash "$APPLE_GARDEN" 85c03626b396563ee69460dacad6faa9a7cb8719ad661aca26692e0e099df5a0
require_hash "$DGX_GARDEN" c084d7a6ebe728931371b60af0c41d3ca1dad7198fc8bacdaf8ce9c491f884a2
require_hash "$LOWERING_SOURCE" 7087649a5cfdb41a884aa9a2e1b0b64bbe2d25da3ca3cd1d54d5b70429854edb
require_hash "$LOWERING_SEMANTICS" 9e92f256c25a774979a1b02cb10d5d39a1ae702ccec3a273044f91930f03a970
require_hash "$LOWERING_RECEIPT" daef832ee6370b656e93ae84c76ba6d17c98aaf5ad1dd86674dee27ba0f84346
require_hash "$AARCH_SOURCE" ce0693e51f5204f89c67b7917fd129dc1976f069675323ec73d4e2c42913078b
require_hash "$AARCH_SEMANTICS" ed66cc2e2fe27ce06842c1ef2091e2f482b8bcb2d4b84e4e649361ca957b7b14
require_hash "$AARCH_RECEIPT" cd64c91c330c9a81e554408a10de4bccbdf9984395ec049c48dc99148aa11934
require_hash "$PTX_SOURCE" ca2760d539c4602c85841ac8475a9ffd8a2f760313a8169faf99a32956063bba
require_hash "$PTX_SEMANTICS" 1454e6a212f320fbf4194b3cbb220a30abed56fbf5e8041ce076b7dee5cae697
require_hash "$PTX_RECEIPT" e68f6edacfa85c48cd3cb51ab4929975a187174b0b1ab980a2c0f0868f5f38fa

require_hash "$A64_DIR/tbl_advsimd.xml" 48ef32ed67b9824ba39eb58518faec196472c3a574cf1bbe1f3a494811a6cbbe
require_hash "$A64_DIR/tbx_advsimd.xml" fa21f8c0784ec327ca9089552d22b55e0eb4b9dd6e0a2eeb078eeed0e203ca79
require_hash "$A64_DIR/notice.xml" 7f6e2780187dc8eb12b53d97eb435be19597b1af256a84fb44d4b5bd41846747
require_hash "${PTX_PREFIX}000.part" 6590d9e3ba60e55e3f0d2cb7f1d83cd3d5735abb7526517709533cfa3093ee91
require_hash "${PTX_PREFIX}001.part" 3e080ba7e8e556e29aed0c69ef818de39cb48b67ee6b442bd018dcb6ffa9bd8d
require_hash "${PTX_PREFIX}002.part" 9b120aa8ca72eabc4db120a19486292c5b4715f8f29a31c9b284e7651195ae91
require_hash "${PTX_PREFIX}003.part" 0ae51edf20e03d37e77f826350b3d63e726fa2e55cebc6a60e00ce00e733292d

require_hash "$APPLE_MODULE" 79c2e859ffe81f3add1ebb36608a5995672c10a5c1645ec4500a03fcd9bcd031
require_hash "$APPLE_EXAMPLE" 0fa666dd3c07e3d261b11a49e08c1cba3e1822f2bfbcff2bc73a71d610711c5b
require_hash "$APPLE_SEMANTICS" 377aed20ffd302aeb3ff71f6609643f17d2a9983129e319d5545b81c589dc3e6
require_hash "$APPLE_RECEIPT" ffac9c58e1ea853767395c5cfd339af130d3285bf3a29ee2b2b800b4f8fc2810
require_hash "$DGX_MODULE" 4be23864a14274d7996dd890473a5b3356a88441a589e509080c9978ba1cf404
require_hash "$DGX_EXAMPLE" 976866431a13fd7ea833ecd3f6fa81983573a389e32bb2e1e4a779d99ac73dd8
require_hash "$DGX_SEMANTICS" a163f5924428de0f8f2a33a54ea864d82bfab753cf80dc04b8c9698c4a225336
require_hash "$DGX_RECEIPT" b9a85c51fb4af5767395440d0b3669f258c86d510fd89b5fb7ebc3368399a807

APPLE_SOURCE_SHA="$(
  cd "$ROOT_DIR"
  sha256sum stdlib/hardware/pireus/apple_a64_tbl_lowering.sio \
    examples/pireus_apple_a64_tbl_lowering.sio | sha256sum | cut -d' ' -f1
)"
DGX_SOURCE_SHA="$(
  cd "$ROOT_DIR"
  sha256sum stdlib/hardware/pireus/dgx_ptx_shfl_lowering.sio \
    examples/pireus_dgx_ptx_shfl_lowering.sio | sha256sum | cut -d' ' -f1
)"
[[ "$APPLE_SOURCE_SHA" == 03c0a315a579e568b14876dc06a116565599598f4f6cf7ac4a1bb6221a3d1e09 ]] || fail 'Apple source manifest drift'
[[ "$DGX_SOURCE_SHA" == a5e3d1c25f0c3745ad8d0e78f96ab0c2bd5ea0cd68a71ba11453ea06f9c1d733 ]] || fail 'DGX source manifest drift'

APPLE_COMMAND='SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pireus_apple_a64_tbl_lowering.sio docs/internal/garden/seeds/2026-08-27-pireus-xor-lowering-apple-silicon.md stdlib/hardware/pireus/xor_lowering_legality.sio docs/research/pireus_xor_lowering_legality_semantics.md docs/research/receipts/pireus_xor_lowering_legality_20260827.md stdlib/hardware/pireus/aarchmrs_import.sio docs/research/pireus_aarchmrs_tbl_import_semantics.md docs/research/receipts/pireus_aarchmrs_tbl_import_20260827.md /tmp/pireus-a64-isa-2025-12/ISA_A64_xml_A_profile-2025-12/tbl_advsimd.xml /tmp/pireus-a64-isa-2025-12/ISA_A64_xml_A_profile-2025-12/tbx_advsimd.xml /tmp/pireus-a64-isa-2025-12/ISA_A64_xml_A_profile-2025-12/notice.xml'
DGX_COMMAND='SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pireus_dgx_ptx_shfl_lowering.sio docs/internal/garden/seeds/2026-08-27-pireus-xor-lowering-dgx.md stdlib/hardware/pireus/xor_lowering_legality.sio docs/research/pireus_xor_lowering_legality_semantics.md docs/research/receipts/pireus_xor_lowering_legality_20260827.md stdlib/hardware/pireus/ptx_import.sio docs/research/pireus_ptx_prmt_import_semantics.md docs/research/receipts/pireus_ptx_prmt_import_20260827.md /tmp/pireus-ptx-13.2.0/chunks-v1/part-'
APPLE_COMMAND_SHA="$(sha_text "$APPLE_COMMAND")"
DGX_COMMAND_SHA="$(sha_text "$DGX_COMMAND")"
[[ "$APPLE_COMMAND_SHA" == 2bab9bfea153113dccf161dff5b7f8ef80a146bd62bb0d60a25611d8432a756a ]] || fail 'Apple command record drift'
[[ "$DGX_COMMAND_SHA" == 3ef15a2030f83773ee948598b37727743fb73b639b97e556388090e8defb40b0 ]] || fail 'DGX command record drift'

authorize a457d7859b3336701d538e07d876ef15d97fedcafd3daf02f2fe22f85262176f \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE' \
  "$(preaction_frame "$APPLE_SOURCE_SHA" "$APPLE_COMMAND_SHA")"
authorize f6bd989409f73686d6b9909fe72802c5476163f1cd5076b020e8c8fc68968ad8 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE' \
  "$(preaction_frame "$DGX_SOURCE_SHA" "$DGX_COMMAND_SHA")"

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pireus-canonical-lowering.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

APPLE_ARGS=(
  "$APPLE_GARDEN" "$LOWERING_SOURCE" "$LOWERING_SEMANTICS" "$LOWERING_RECEIPT"
  "$AARCH_SOURCE" "$AARCH_SEMANTICS" "$AARCH_RECEIPT"
  "$A64_DIR/tbl_advsimd.xml" "$A64_DIR/tbx_advsimd.xml" "$A64_DIR/notice.xml"
)
DGX_ARGS=(
  "$DGX_GARDEN" "$LOWERING_SOURCE" "$LOWERING_SEMANTICS" "$LOWERING_RECEIPT"
  "$PTX_SOURCE" "$PTX_SEMANTICS" "$PTX_RECEIPT" "$PTX_PREFIX"
)

SOUNIO_SOUC_ENGINE=lean_single "$ROOT_DIR/bin/souc" run "$APPLE_EXAMPLE" "${APPLE_ARGS[@]}" >"$WORK_DIR/apple-1.txt"
SOUNIO_SOUC_ENGINE=lean_single "$ROOT_DIR/bin/souc" run "$APPLE_EXAMPLE" "${APPLE_ARGS[@]}" >"$WORK_DIR/apple-2.txt"
cmp -s "$WORK_DIR/apple-1.txt" "$WORK_DIR/apple-2.txt" || fail 'Apple authority streams differ'
[[ "$(sha_file "$WORK_DIR/apple-1.txt")" == d1de1ec160d0cf7c69a7f8e3f50d5ae027457f8c23648fb685c5216d19f10f81 ]] || fail 'Apple result drift'
grep -Fqx ' in_domain_sources=256' "$WORK_DIR/apple-1.txt" || fail 'Apple in-domain coverage missing'
grep -Fqx ' out_of_domain_sources=0' "$WORK_DIR/apple-1.txt" || fail 'Apple out-of-domain refusal missing'
grep -Fqx ' bijective_displacements=16' "$WORK_DIR/apple-1.txt" || fail 'Apple bijection coverage missing'
grep -Fqx ' dimension_matches_bits=1' "$WORK_DIR/apple-1.txt" || fail 'Apple dimension closure missing'
grep -Fqx ' payload_addresses_preserved=1' "$WORK_DIR/apple-1.txt" || fail 'Apple symbolic address preservation missing'
grep -Fqx ' total=15' "$WORK_DIR/apple-1.txt" || fail 'Apple negatives incomplete'
grep -Fqx ' failures=0' "$WORK_DIR/apple-1.txt" || fail 'Apple authority result failed'

SOUNIO_SOUC_ENGINE=lean_single "$ROOT_DIR/bin/souc" run "$DGX_EXAMPLE" "${DGX_ARGS[@]}" >"$WORK_DIR/dgx-1.txt"
SOUNIO_SOUC_ENGINE=lean_single "$ROOT_DIR/bin/souc" run "$DGX_EXAMPLE" "${DGX_ARGS[@]}" >"$WORK_DIR/dgx-2.txt"
cmp -s "$WORK_DIR/dgx-1.txt" "$WORK_DIR/dgx-2.txt" || fail 'DGX authority streams differ'
[[ "$(sha_file "$WORK_DIR/dgx-1.txt")" == 495c52ccf2370c4e668ab1e9bc4d7dbc02c0d97a8cd27a0dbdfe5aa130d8e54e ]] || fail 'DGX result drift'
grep -Fqx ' identity_shfl_sync=2' "$WORK_DIR/dgx-1.txt" || fail 'DGX identity count missing'
grep -Fqx ' nontrivial_shfl_sync=30' "$WORK_DIR/dgx-1.txt" || fail 'DGX nontrivial count missing'
grep -Fqx ' payload_addresses_preserved=1' "$WORK_DIR/dgx-1.txt" || fail 'DGX symbolic address preservation missing'
grep -Fqx ' total=25' "$WORK_DIR/dgx-1.txt" || fail 'DGX negatives incomplete'
grep -Fqx ' failures=0' "$WORK_DIR/dgx-1.txt" || fail 'DGX authority result failed'

authorize cec8d7c4b299b3e88d911b699fc09bcb3d1483b1d7c13e651c1de2ecb098d1cd \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN' \
  "$(freeze_frame "$APPLE_SOURCE_SHA" 377aed20ffd302aeb3ff71f6609643f17d2a9983129e319d5545b81c589dc3e6 a479ff676865104174ed6f34972724680f5024db7093ac5e4a0a64d9afb16f6f "$APPLE_COMMAND_SHA" d1de1ec160d0cf7c69a7f8e3f50d5ae027457f8c23648fb685c5216d19f10f81)"
authorize 3aa426e97e146f180d2fe0a5961153e5a7323bb893b94388a66c23bfa4ea2ab0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN' \
  "$(freeze_frame "$DGX_SOURCE_SHA" a163f5924428de0f8f2a33a54ea864d82bfab753cf80dc04b8c9698c4a225336 1de516eb2e819cac86385f86123e26db8e31e059404f59d29e786d2073a3bc71 "$DGX_COMMAND_SHA" 495c52ccf2370c4e668ab1e9bc4d7dbc02c0d97a8cd27a0dbdfe5aa130d8e54e)"

ZERO='0 0 0 0 0 0 0 0'
PYTHON_FRAME="9020 3 4 7 7 1 0 0 0 0 0 0 0 0 0 0 0 0 $(sha_limbs "$DGX_SOURCE_SHA") $(sha_limbs a163f5924428de0f8f2a33a54ea864d82bfab753cf80dc04b8c9698c4a225336) $(sha_limbs a163f5924428de0f8f2a33a54ea864d82bfab753cf80dc04b8c9698c4a225336) $(sha_limbs 2ce5194cdc517de8d7a0063e09d4f4e7b6b5701a23fa1031a3ec9e8f56486b6e) $(sha_limbs b6326139297e9ba59d82e208a2404b01e3d57445357a5e803c51c845dd388db0) $(sha_limbs "$DGX_COMMAND_SHA") $(sha_limbs 495c52ccf2370c4e668ab1e9bc4d7dbc02c0d97a8cd27a0dbdfe5aa130d8e54e) $ZERO"
set +e
PYTHON_DECISION="$(printf '%s\n' "$PYTHON_FRAME" | "$GUARDIAN")"
PYTHON_RC=$?
set -e
[[ "$PYTHON_RC" -eq 110 ]] || fail "Python authority frame returned $PYTHON_RC"
[[ "$PYTHON_DECISION" == 'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN' ]] || fail "Python authority frame was not E110: $PYTHON_DECISION"

if [[ "${PIREUS_REQUIRE_CANONICAL_MADAROS:-0}" == 1 ]]; then
  "$ROOT_DIR/bin/souc" check "$APPLE_EXAMPLE" || fail 'canonical Madaros Apple check remains blocked'
  "$ROOT_DIR/bin/souc" check "$DGX_EXAMPLE" || fail 'canonical Madaros DGX check remains blocked'
fi

printf 'PIREUS_CANONICAL_TARGET_LOWERING_SOUNIO_PASS=true apple_result=%s dgx_result=%s apple_negatives=15/15 dgx_negatives=25/25 python_oracle=E110 interpreter_launch_count=0 parity_open=false claim_ready=false canonical_madaros=%s\n' \
  d1de1ec160d0cf7c69a7f8e3f50d5ae027457f8c23648fb685c5216d19f10f81 \
  495c52ccf2370c4e668ab1e9bc4d7dbc02c0d97a8cd27a0dbdfe5aa130d8e54e \
  "${PIREUS_REQUIRE_CANONICAL_MADAROS:-BLOCKED_RECORDED}"
