#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"

GARDEN_REL='tools/pireus/GARDEN_PIREUS_OPERATOR_GENOME_V3.md'
CONTRACT_REL='tools/pireus/PIREUS_OPERATOR_GENOME_CONTRACT_V3.md'
BASE_REL='stdlib/algebra/cayley_dickson.sio'
PARENT_REL='stdlib/hardware/pireus/operator_genesis_bilinear.sio'
TARGET_PROFILE_REL='stdlib/hardware/pireus/target_profile.sio'
U250_REL='stdlib/hardware/pireus/u250_dual_card_admission.sio'
MODULE_REL='stdlib/hardware/pireus/operator_genome.sio'
EXAMPLE_REL='examples/pireus_operator_genome.sio'
TEST_REL='tests/stdlib/hardware/test_pireus_operator_genome.sio'
FIRST_RECEIPT_REL='tools/pireus/operator_genome.first.v3'
FREEZE_REL='tools/pireus/operator_genome.freeze.v3'
PARITY_RECEIPT_REL='tools/pireus/operator_genome.parity-open.v3'
EVIDENCE_REL='tools/pireus/evidence/operator_genome_v3.txt'

GARDEN_COMMIT='038261eff749e62e78502f9f7525393cc4651c56'
EXECUTABLE_COMMIT='37c7a1192c81e1d9bc1be22615787899b88cf9d0'
FIRST_EVIDENCE_COMMIT='52a60403ca56c3827a6f35b179a3a08254fe84f1'
SEMANTIC_FREEZE_COMMIT='a794bfa2325dc6c5c545c95a7247dc1e3b1db4d4'
FREEZE_RECEIPT_COMMIT='1ad4cea31255801aee219fcaba14856b5e74bf3b'

GARDEN_SHA256='1dfe7551cc07465a3f25506d13c7efe5c61a73127985ec269294f817ce92eda9'
CONTRACT_SHA256='2ebb668835106d6ce612cb62c4ef291ac5044125e22f18e176538e249480ed1b'
BASE_SHA256='e7dd98de0644013ebf6e0d435fddb7f893720f684c96c3fbe20cc11b1f518fed'
PARENT_SHA256='31f5fe668c100f0aa27b4c4405c022c127e5445a743d5029e2d913da8dfd8a44'
PARENT_SEMANTICS_SHA256='bb5560806ea7a84a0cc5f88ec5d4adbea4004ec6b2560af6e4d8de31b3a88d3b'
TARGET_PROFILE_SHA256='d41726a8a7eba62132e3763cf6a71938de746ec9d58ce8a20caa40709546d6a4'
U250_SHA256='bf952aa999dad0e74871a0fc78dd6fe67479840a8f334de1c639ceaabd37eafb'
FIRST_SOURCE_SHA256='a0693e455d716a0f0b5ac73a27be8fbf5ebea82ce7e9720477b710b9ae7ab5ed'
MODULE_SHA256='92765416ad8854376a779ef452f89497e2df77f225bf5a4eb5f74f4cd9004a6d'
EXAMPLE_SHA256='470d401ab6730e44146259b3efe249af7618eda057aeebc7d70099598ffff8e5'
TEST_SHA256='6b285bc93d071af0048441ba88ca401126ba8157c653e6ece00d7e0eb473c8b6'
FIRST_RECEIPT_SHA256='9c837ef4f70535ad588c66dc3756553e3a2a5242e239153ed49a56fc7cac1984'
FREEZE_SHA256='0b4486ae3c7d0034ffb82208f19330b710ed7d7e92115e93a6f411b354dd03f6'
PARITY_RECEIPT_SHA256='b2100377695575e024e333a4519687a0ff727989198f7ee0213d0f78c36bc7eb'
FIRST_EVIDENCE_SHA256='8158ef30dbc31f2bd5aaae477643b3f93ae16cbfa4e8391d8ddd3b06f1a2cfe6'
EVIDENCE_SHA256='3e79844d3dbd9034e0d8706bf0c3055cba9a7dda0fcfb2daae959e9dbf0c1905'
SOURCE_MANIFEST_SHA256='4ab3f1b5574b8b49afa75703a72800ed5a54915aa94e015ecf39fa66b21745d1'
SEMANTICS_SHA256='99d5e3417550ad3f7b8223b29f25b3d8d8616ac425d4615c37c7f2f402668926'
WRAPPER_SHA256='ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008'
RESOLVER_SHA256='a7e37545490745a58731933b0e07db69843cfeea30e739ed554b82d099ec3d84'
COMPILER_SHA256='6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
GUARDIAN_BUILD_SHA256='af7c1098143d0aad108684646df4c72fecca03404557f5494206713486ca09b6'
GUARDIAN_POLICY_SHA256='64bb0118793fe46dcb392abc1a9212eb15bd55047461576a3ef1a6cefa3f17da'
GUARDIAN_MAIN_SHA256='99b6fe7e1c687c3a4e76cfe1585e4826e753f473dff8676dd287eb2f9e0021bc'
GUARDIAN_SELFTEST_SHA256='c9c7f839fb262dbf616716e3c5f0601bb03cfbcbbcfe2fbe09bd0b39894e2a9f'
PARENT_GATE_SHA256='8acf9c5af632334ae924a0a9c8c77828acb58e7e33225ac9d304ecaf37c13eb9'
TOOLCHAIN_SHA256='a4d9e4290d4373baa095cdcdd2f6582587323fa392166815ec6676c6178c3590'
HARDWARE_SHA256='ba649e3df628e482654987fc29b86489bfdfc9f69f6bfdc141f3a8d238ee33cd'
COMMAND_SHA256='6b45abd53b271649d963d551f405a93b0bd88e0786d807270a6fa0f850ad1493'
TEST_COMMAND_SHA256='8d30a6bcfcfe2f14e5c7d7f1b941bd4834c039b8e26b34a85d67e066ed2cceb2'
RESULT_CONTRACT_SHA256='40b7ffa0e84ff6657b22e1c770133ad5b6d872d5029fa41c1b838352660a0920'
TEST_OUTPUT_SHA256='9b3f26f065c6302e75173fb4c05545c5ca3a0ccd6e533dc6a9e3bee973607b92'

FIRST_PREEXEC_FRAME_SHA256='49cb33fa5bb6c250d3aae589c1e35aeea7a2f841ed96c20d81b11dd4d6d1c856'
FIRST_TEST_PREEXEC_FRAME_SHA256='e3f1e14406c824989cdead8d2fb1ea0f79401f468357d715679af8e0afa30cbc'
FREEZE_FRAME_SHA256='330a4ffe40c98c6c16ce8aec83056b545c98f91c00e557eb18d105ee20da8e01'
TEST_FREEZE_FRAME_SHA256='d9c95f6b5dd5f57fdc495cd5f793bf893daa6f8537174c4c47b9265458542e95'
POLICY_MISSING_FRAME_SHA256='1d2eefc65c9e0be016666cd2c8250b3348e7a6551c0f19ee2de691ff7852b5bb'
POLICY_TIMEOUT_FRAME_SHA256='b64e899634d9d847f8ba1b9b9c6f8fc17d65e0db08eb39e2d9c1347e171222e1'
PYTHON_FRAME_SHA256='dc262a07e0a49e7aadbd1524fed95eb9c40c74e7a299458591ab01a0c84cfa5c'
RUST_FRAME_SHA256='d9110f067f0f59620eb78f2b13ddb0c2a08747bc36ab7379f4212028b82738d1'
LLM_PROMOTION_FRAME_SHA256='7311b0eafddd0e83058bc157fac1b0a0730483c247b717fa967a4aed59c32716'
CPP_AUTHORITY_FRAME_SHA256='a93dc73937ebae64863f9c8c7f8683d2e0e1ea940a0abf8b65065c87f94f0a9b'
PARITY_PREFREEZE_FRAME_SHA256='c9689a48187b36c9a0df114a9398d0d711e1c632a88f16a7911d932cbc0004c2'
PARITY_OPEN_FRAME_SHA256='e7c3849d9633cef7b91d8dd90380980fde1594466342fdf83123a91931eb2e68'
PYTHON_TOOLCHAIN_SHA256='497ce0938df96d9bf3c159472d251a946c2d6bd832220937d1b885f7759b05ba'
PYTHON_COMMAND_SHA256='72566473f0019fb50e65175a0ac019af5ad2f495ccfc30b664b7735791e968fc'
RUST_TOOLCHAIN_SHA256='b58640570ba9ffcdb2c2d241e4ce8ece9c7d75c6b1e59e308dee3e5f0e10b56d'
RUST_COMMAND_SHA256='0421c7020d972449b7a1f0492945c18a10be46f6a16ade7f8efad157e2e01b00'
ZERO='0 0 0 0 0 0 0 0'

fail() {
  printf 'pireus operator genome: FAIL: %s\n' "$*" >&2
  exit 1
}

sha_file() { sha256sum "$1" | cut -d' ' -f1; }
sha_text() { printf '%s\n' "$1" | sha256sum | cut -d' ' -f1; }

require_hash() {
  local path="$1" expected="$2"
  [[ -f "${path}" ]] || fail "missing file: ${path}"
  [[ "$(sha_file "${path}")" == "${expected}" ]] ||
    fail "hash drift: ${path}"
}

require_line() {
  local path="$1" expected="$2"
  grep -Fqx -- "${expected}" "${path}" ||
    fail "missing exact line in ${path}: ${expected}"
}

transcript_admitted() {
  local path="$1"
  [[ "$(sha_file "${path}")" == "${EVIDENCE_SHA256}" ]] &&
    grep -Fqx -- ' class=26' "${path}" &&
    grep -Fqx -- ' quadratic_code=198' "${path}" &&
    grep -Fqx -- ' matrix=1128' "${path}" &&
    grep -Fqx -- ' negatives=104' "${path}" &&
    grep -Fqx -- ' groups=32' "${path}" &&
    grep -Fqx -- ' same_chunk_groups=16' "${path}" &&
    grep -Fqx -- ' cross_chunk_groups=16' "${path}" &&
    grep -Fqx -- ' permutation_failures=0' "${path}" &&
    grep -Fqx -- ' coverage_failures=0' "${path}" &&
    grep -Fqx -- ' unresolved_obligations=40' "${path}" &&
    grep -Fqx -- ' frozen_match=1' "${path}" &&
    grep -Fqx -- ' frozen_mismatch_code=0' "${path}" &&
    grep -Fqx -- ' valid=1' "${path}" &&
    [[ "$(wc -l < "${path}")" -eq 625 ]] &&
    [[ "$(wc -c < "${path}")" -eq 10919 ]]
}

sha_limbs() {
  local hex="$1" out='' i part
  for ((i = 0; i < 8; i++)); do
    part="${hex:$((i * 8)):8}"
    out="${out}${out:+ }$((16#${part}))"
  done
  printf '%s' "${out}"
}

preexec_frame() {
  local command_hash="$1"
  printf '9020 1 2 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${FIRST_SOURCE_SHA256}")" "${ZERO}" \
    "$(sha_limbs "${PARENT_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${command_hash}")" "${ZERO}" "${ZERO}"
}

freeze_frame() {
  local policy="$1" command_hash="$2"
  printf '9020 2 3 1 1 %s 1 1 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${policy}" "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PARENT_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${command_hash}")" \
    "$(sha_limbs "${RESULT_CONTRACT_SHA256}")" "${ZERO}"
}

forbidden_frame() {
  local language="$1" toolchain="$2" command="$3"
  printf '9020 3 4 %s 7 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${language}" "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PARENT_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${toolchain}")" "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${command}")" "${ZERO}" "${ZERO}"
}

llm_promotion_frame() {
  printf '9020 3 5 6 6 1 0 0 0 1 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PARENT_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${COMMAND_SHA256}")" "${ZERO}" "${ZERO}"
}

cpp_authority_frame() {
  printf '9020 3 4 4 1 1 1 1 1 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PARENT_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${COMMAND_SHA256}")" \
    "$(sha_limbs "${RESULT_CONTRACT_SHA256}")" "${ZERO}"
}

parity_frame() {
  local stage="$1"
  printf '9020 %s 4 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${stage}" "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${COMMAND_SHA256}")" "${ZERO}" "${ZERO}"
}

authorize() {
  local label="$1" frame="$2" expected_sha="$3" expected_rc="$4" expected="$5"
  local decision rc
  [[ "$(sha_text "${frame}")" == "${expected_sha}" ]] ||
    fail "Guardian frame drift: ${label}"
  set +e
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  rc=$?
  set -e
  [[ "${rc}" -eq "${expected_rc}" ]] ||
    fail "Guardian exit drift for ${label}: expected ${expected_rc}, got ${rc}"
  [[ "${decision}" == "${expected}" ]] ||
    fail "Guardian decision drift for ${label}: ${decision}"
  printf 'GUARDIAN_DECISION label=%s decision=%s\n' "${label}" "${decision}"
}

attempt_forbidden_oracle() {
  local label="$1" language="$2" toolchain="$3" command="$4" frame_sha="$5"
  local frame decision rc process_launched=false
  frame="$(forbidden_frame "${language}" "${toolchain}" "${command}")"
  [[ "$(sha_text "${frame}")" == "${frame_sha}" ]] ||
    fail "Guardian frame drift: ${label}"
  set +e
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  rc=$?
  set -e
  if [[ "${rc}" -eq 0 ]]; then
    process_launched=true
    fail "Guardian allowed ${label}; outer gate refused process launch"
  fi
  [[ "${rc}" -eq 110 ]] || fail "${label} denial exit drift: ${rc}"
  [[ "${decision}" == \
    'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN' ]] ||
    fail "${label} denial drift: ${decision}"
  [[ "${process_launched}" == false ]] || fail "${label} process launched"
  printf 'GUARDIAN_DECISION label=%s decision=%s process_launched=false\n' \
    "${label}" "${decision}"
}

require_hash "${ROOT}/${GARDEN_REL}" "${GARDEN_SHA256}"
require_hash "${ROOT}/${CONTRACT_REL}" "${CONTRACT_SHA256}"
require_hash "${ROOT}/${BASE_REL}" "${BASE_SHA256}"
require_hash "${ROOT}/${PARENT_REL}" "${PARENT_SHA256}"
require_hash "${ROOT}/${TARGET_PROFILE_REL}" "${TARGET_PROFILE_SHA256}"
require_hash "${ROOT}/${U250_REL}" "${U250_SHA256}"
require_hash "${ROOT}/${MODULE_REL}" "${MODULE_SHA256}"
require_hash "${ROOT}/${EXAMPLE_REL}" "${EXAMPLE_SHA256}"
require_hash "${ROOT}/${TEST_REL}" "${TEST_SHA256}"
require_hash "${ROOT}/${FIRST_RECEIPT_REL}" "${FIRST_RECEIPT_SHA256}"
require_hash "${ROOT}/${FREEZE_REL}" "${FREEZE_SHA256}"
require_hash "${ROOT}/${PARITY_RECEIPT_REL}" "${PARITY_RECEIPT_SHA256}"
require_hash "${ROOT}/${EVIDENCE_REL}" "${EVIDENCE_SHA256}"
require_hash "${ROOT}/bin/souc" "${WRAPPER_SHA256}"
require_hash "${ROOT}/scripts/lib/resolve_souc.sh" "${RESOLVER_SHA256}"
require_hash "${ROOT}/bin/souc-lean-single-x86_64" "${COMPILER_SHA256}"
require_hash "${GUARDIAN}" "${GUARDIAN_SHA256}"
require_hash "${ROOT}/scripts/dev/build_sounio_loom_language_authority.sh" \
  "${GUARDIAN_BUILD_SHA256}"
require_hash "${ROOT}/stdlib/coordination/loom_language_authority.sio" \
  "${GUARDIAN_POLICY_SHA256}"
require_hash "${ROOT}/tools/loom/language_authority_main.sio" \
  "${GUARDIAN_MAIN_SHA256}"
require_hash "${ROOT}/scripts/ci/sounio_loom_language_authority_selftest.sh" \
  "${GUARDIAN_SELFTEST_SHA256}"
require_hash "${ROOT}/scripts/ci/pireus_operator_genesis_bilinear.sh" \
  "${PARENT_GATE_SHA256}"
[[ -x "${GUARDIAN}" ]] || fail 'native Loom Guardian unavailable'

git -C "${ROOT}" merge-base --is-ancestor "${GARDEN_COMMIT}" \
  "${EXECUTABLE_COMMIT}" || fail 'Garden does not precede Sounio executable'
git -C "${ROOT}" merge-base --is-ancestor "${EXECUTABLE_COMMIT}" \
  "${FIRST_EVIDENCE_COMMIT}" || fail 'first executable does not precede first evidence'
git -C "${ROOT}" merge-base --is-ancestor "${FIRST_EVIDENCE_COMMIT}" \
  "${SEMANTIC_FREEZE_COMMIT}" || fail 'first evidence does not precede matcher'
git -C "${ROOT}" merge-base --is-ancestor "${SEMANTIC_FREEZE_COMMIT}" \
  "${FREEZE_RECEIPT_COMMIT}" || fail 'matcher does not precede freeze receipt'
git -C "${ROOT}" merge-base --is-ancestor "${FREEZE_RECEIPT_COMMIT}" HEAD ||
  fail 'freeze receipt is not an ancestor of HEAD'

[[ "$(git -C "${ROOT}" show "${EXECUTABLE_COMMIT}:${MODULE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${FIRST_SOURCE_SHA256}" ]] || fail 'first executable source hash drift'
if git -C "${ROOT}" show "${EXECUTABLE_COMMIT}:${MODULE_REL}" |
    grep -Fq 'pireus_operator_genome_matches_frozen_semantics'; then
  fail 'frozen matcher existed in first executable commit'
fi
for forbidden_golden in 'expected_negative_signs' 'expected_group_negative_mask' \
    'expected_fixture_bits' 'expected_genome_digest'; do
  if git -C "${ROOT}" show "${EXECUTABLE_COMMIT}:${MODULE_REL}" |
      grep -Fq "${forbidden_golden}"; then
    fail "golden ${forbidden_golden} existed before first execution"
  fi
done
[[ "$(git -C "${ROOT}" show "${FIRST_EVIDENCE_COMMIT}:${EVIDENCE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${FIRST_EVIDENCE_SHA256}" ]] || fail 'first evidence object drift'
[[ "$(git -C "${ROOT}" show "${FIRST_EVIDENCE_COMMIT}:${EVIDENCE_REL}" | wc -l)" -eq 623 ]] ||
  fail 'first evidence line count drift'
[[ "$(git -C "${ROOT}" show "${FIRST_EVIDENCE_COMMIT}:${EVIDENCE_REL}" | wc -c)" -eq 10879 ]] ||
  fail 'first evidence byte count drift'
if git -C "${ROOT}" show "${FIRST_EVIDENCE_COMMIT}:${EVIDENCE_REL}" |
    grep -Fq 'frozen_match='; then
  fail 'first evidence contains post-execution matcher result'
fi
[[ "$(git -C "${ROOT}" show "${SEMANTIC_FREEZE_COMMIT}:${MODULE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${MODULE_SHA256}" ]] || fail 'frozen source object drift'
[[ "$(git -C "${ROOT}" show "${FREEZE_RECEIPT_COMMIT}:${FREEZE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${FREEZE_SHA256}" ]] || fail 'freeze receipt object drift'
[[ "$(git -C "${ROOT}" show "${FREEZE_RECEIPT_COMMIT}:${CONTRACT_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${CONTRACT_SHA256}" ]] || fail 'contract object drift'

toolchain_record='engine=lean_single wrapper=ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008 resolver=a7e37545490745a58731933b0e07db69843cfeea30e739ed554b82d099ec3d84 compiler=6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2'
[[ "$(sha_text "${toolchain_record}")" == "${TOOLCHAIN_SHA256}" ]] ||
  fail 'toolchain receipt drift'
hardware_record="host=$(hostname) arch=$(uname -m) kernel=$(uname -s) $(uname -r) online_cpus=$(getconf _NPROCESSORS_ONLN) model=$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -n 1)"
[[ "${hardware_record}" == 'host=sounio-workspace-control-0 arch=x86_64 kernel=Linux 7.0.2-5-pve online_cpus=64 model=INTEL(R) XEON(R) GOLD 6526Y' ]] ||
  fail 'live Xeon hardware does not match frozen receipt'
[[ "$(sha_text "${hardware_record}")" == "${HARDWARE_SHA256}" ]] ||
  fail 'hardware receipt drift'
[[ "$(sha_text 'SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pireus_operator_genome.sio')" == \
  "${COMMAND_SHA256}" ]] || fail 'authority command drift'
[[ "$(sha_text 'SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/stdlib/hardware/test_pireus_operator_genome.sio')" == \
  "${TEST_COMMAND_SHA256}" ]] || fail 'test command drift'
frozen_result_record="schema=pireus-operator-genome.v3 cells=256 negative_signs=104 groups=32 same_chunk=16 cross_chunk=16 targets=4 unresolved_obligations=40 genome_sha256=${SEMANTICS_SHA256} parity_open=false claim_ready=false"
[[ "$(sha_text "${frozen_result_record}")" == "${RESULT_CONTRACT_SHA256}" ]] ||
  fail 'result contract receipt drift'

require_line "${ROOT}/${FREEZE_REL}" 'stage=SEMANTICS_FROZEN'
require_line "${ROOT}/${FREEZE_REL}" 'producing_language=Sounio'
require_line "${ROOT}/${FREEZE_REL}" 'language_role=SEMANTIC_AUTHORITY'
require_line "${ROOT}/${FREEZE_REL}" "source_manifest_sha256=${SOURCE_MANIFEST_SHA256}"
require_line "${ROOT}/${FREEZE_REL}" "semantics_sha256=${SEMANTICS_SHA256}"
require_line "${ROOT}/${FREEZE_REL}" 'expected_negative_signs=104'
require_line "${ROOT}/${FREEZE_REL}" 'expected_microprogram_groups=32'
require_line "${ROOT}/${FREEZE_REL}" 'expected_unresolved_target_obligations=40'
require_line "${ROOT}/${FREEZE_REL}" 'target_lowering_admitted=false'
require_line "${ROOT}/${FREEZE_REL}" 'algorithmic_novelty=false'
require_line "${ROOT}/${FREEZE_REL}" 'claim_ready=false'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" 'stage=PARITY_OPEN'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" 'first_authority_transcript_lines=623'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" 'authority_transcript_lines=625'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" 'opening_language_code=2'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" 'opening_language_role_code=2'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" 'opening_receipt_kind=STAGED_TRANSITION_RECEIPT'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" 'formal_parity_status=OPEN_NOT_EXECUTED'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" 'effect_parity_status=OPEN_NOT_EXECUTED'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" 'material_parity_status=OPEN_NOT_EXECUTED'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" 'target_names=Darwin_Xeon,Apple_Silicon,DGX_Spark,dual_AMD_Alveo_U250'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" 'target_obligations_unresolved=40'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" 'parity_processes_launched=false'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" 'target_processes_launched=false'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" 'claim_ready=false'
require_line "${ROOT}/stdlib/coordination/loom_language_authority.sio" \
  '    if language == 2 { return 2 }'
require_line "${ROOT}/stdlib/coordination/loom_language_authority.sio" \
  '    if action == 4 && stage == 3 { return 4 }'
transcript_admitted "${ROOT}/${EVIDENCE_REL}" ||
  fail 'canonical frozen authority transcript was not admitted'

"${ROOT}/scripts/ci/sounio_loom_language_authority_selftest.sh" >/dev/null
[[ "$(sha_text "$(preexec_frame "${COMMAND_SHA256}")")" == "${FIRST_PREEXEC_FRAME_SHA256}" ]] ||
  fail 'first authority frame cannot be reconstructed'
[[ "$(sha_text "$(preexec_frame "${TEST_COMMAND_SHA256}")")" == "${FIRST_TEST_PREEXEC_FRAME_SHA256}" ]] ||
  fail 'first test frame cannot be reconstructed'
authorize FREEZE "$(freeze_frame 1 "${COMMAND_SHA256}")" \
  "${FREEZE_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'
authorize TEST_FREEZE "$(freeze_frame 1 "${TEST_COMMAND_SHA256}")" \
  "${TEST_FREEZE_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'
attempt_forbidden_oracle PYTHON_ORACLE 7 "${PYTHON_TOOLCHAIN_SHA256}" \
  "${PYTHON_COMMAND_SHA256}" "${PYTHON_FRAME_SHA256}"
attempt_forbidden_oracle RUST_ORACLE 8 "${RUST_TOOLCHAIN_SHA256}" \
  "${RUST_COMMAND_SHA256}" "${RUST_FRAME_SHA256}"
authorize POLICY_MISSING "$(freeze_frame 0 "${COMMAND_SHA256}")" \
  "${POLICY_MISSING_FRAME_SHA256}" 101 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=101 reason=policy-missing next_stage=SOUNIO_EXECUTABLE'
authorize POLICY_TIMEOUT "$(freeze_frame 2 "${COMMAND_SHA256}")" \
  "${POLICY_TIMEOUT_FRAME_SHA256}" 102 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=102 reason=policy-timeout next_stage=SOUNIO_EXECUTABLE'
authorize PARITY_PREFREEZE "$(parity_frame 2)" \
  "${PARITY_PREFREEZE_FRAME_SHA256}" 112 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=112 reason=wrong-stage next_stage=SOUNIO_EXECUTABLE'
authorize LLM_PROMOTION "$(llm_promotion_frame)" \
  "${LLM_PROMOTION_FRAME_SHA256}" 119 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=119 reason=review-promoted-to-authority next_stage=SEMANTICS_FROZEN'
authorize CPP_AUTHORITY "$(cpp_authority_frame)" \
  "${CPP_AUTHORITY_FRAME_SHA256}" 113 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=SEMANTICS_FROZEN'

umask 077
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/pireus-operator-genome-v3.XXXXXX")"
cleanup() { rm -rf "${TMP_ROOT}"; }
trap cleanup EXIT
(
  cd "${ROOT}"
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check "${EXAMPLE_REL}" >/dev/null
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check "${TEST_REL}" >/dev/null
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run "${EXAMPLE_REL}" >"${TMP_ROOT}/authority.txt"
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run "${TEST_REL}" >"${TMP_ROOT}/test.txt"
)
cmp -s "${TMP_ROOT}/authority.txt" "${ROOT}/${EVIDENCE_REL}" ||
  fail 'authority transcript replay drift'
transcript_admitted "${TMP_ROOT}/authority.txt" ||
  fail 'authority transcript replay was not admitted'
require_hash "${TMP_ROOT}/test.txt" "${TEST_OUTPUT_SHA256}"
require_line "${TMP_ROOT}/test.txt" 'PIREUS_OPERATOR_GENOME_EXECUTABLE_OK'

cp "${ROOT}/${EVIDENCE_REL}" "${TMP_ROOT}/tamper-mask.txt"
sed -i '0,/^ negative_mask=2$/s// negative_mask=3/' "${TMP_ROOT}/tamper-mask.txt"
cmp -s "${TMP_ROOT}/tamper-mask.txt" "${ROOT}/${EVIDENCE_REL}" &&
  fail 'mask sabotage did not mutate transcript'
transcript_admitted "${TMP_ROOT}/tamper-mask.txt" &&
  fail 'mask-tampered transcript was admitted'
cp "${ROOT}/${EVIDENCE_REL}" "${TMP_ROOT}/tamper-count.txt"
sed -i 's/^ negatives=104$/ negatives=103/' "${TMP_ROOT}/tamper-count.txt"
transcript_admitted "${TMP_ROOT}/tamper-count.txt" &&
  fail 'negative-count-tampered transcript was admitted'
cp "${ROOT}/${EVIDENCE_REL}" "${TMP_ROOT}/tamper-digest.txt"
sed -i 's/^ genome=2580931393$/ genome=2580931394/' "${TMP_ROOT}/tamper-digest.txt"
transcript_admitted "${TMP_ROOT}/tamper-digest.txt" &&
  fail 'digest-tampered transcript was admitted'
cp "${ROOT}/${EVIDENCE_REL}" "${TMP_ROOT}/tamper-freeze.txt"
sed -i 's/^ frozen_match=1$/ frozen_match=0/' "${TMP_ROOT}/tamper-freeze.txt"
transcript_admitted "${TMP_ROOT}/tamper-freeze.txt" &&
  fail 'freeze-tampered transcript was admitted'

"${ROOT}/scripts/ci/pireus_operator_genesis_bilinear.sh" >/dev/null
authorize PARITY_OPEN "$(parity_frame 3)" \
  "${PARITY_OPEN_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'

printf '%s\n' \
  'pireus operator genome: PASS stage=PARITY_OPEN language=Sounio cells=256 groups=32 targets=4 unresolved=40 formal=OPEN_NOT_EXECUTED effect=OPEN_NOT_EXECUTED material=OPEN_NOT_EXECUTED claim_ready=false python_process_launched=false rust_process_launched=false'
