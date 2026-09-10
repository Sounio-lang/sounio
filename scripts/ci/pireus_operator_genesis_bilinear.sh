#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"

GARDEN_REL='tools/pireus/GARDEN_PIREUS_OPERATOR_GENESIS_BILINEAR_V2.md'
CONTRACT_REL='tools/pireus/PIREUS_OPERATOR_GENESIS_BILINEAR_CONTRACT_V2.md'
BASE_REL='stdlib/algebra/cayley_dickson.sio'
MODULE_REL='stdlib/hardware/pireus/operator_genesis_bilinear.sio'
EXAMPLE_REL='examples/pireus_operator_genesis_bilinear.sio'
TEST_REL='tests/stdlib/hardware/test_pireus_operator_genesis_bilinear.sio'
FREEZE_REL='tools/pireus/operator_genesis_bilinear.freeze.v2'
EVIDENCE_REL='tools/pireus/evidence/operator_genesis_bilinear_v2.txt'

GARDEN_COMMIT='eb6e7997dc252583469ead1208bdec931180e6ee'
EXECUTABLE_COMMIT='34619031a92441f6145a853b430bdc508dc22fb4'
SEMANTIC_FREEZE_COMMIT='bdfb27cb3c9cf0d3e68392c13668e4792fc71e05'
RECEIPT_COMMIT='469a48d7f9382ecdee8ad2915ab156e79faf3675'

GARDEN_SHA256='8c900c40decd7ae5499652b0cd8d5a4af8cab1beaf59e0fad0b004625b899f3e'
CONTRACT_SHA256='0cb51e12e17be8500be1de679c9ce95d67b8dbffb0750be511833cb76d8548e8'
BASE_SHA256='e7dd98de0644013ebf6e0d435fddb7f893720f684c96c3fbe20cc11b1f518fed'
FIRST_SOURCE_SHA256='82ddc6e2d22771d40461eb21d143f00bd2567ea3eec43895cdbc75d54374097e'
MODULE_SHA256='31f5fe668c100f0aa27b4c4405c022c127e5445a743d5029e2d913da8dfd8a44'
EXAMPLE_SHA256='7683b233fb2bc1854f30ce12e29353f16af69de5462ed45d3ef294c5da9a980d'
TEST_SHA256='b5dc48405af27be5510b59e51f09a25cfffe63d0ffc0c36e6fad3074e7d63676'
FREEZE_SHA256='38f4d5c0a46029283bc21fd901a60e1f7f08332b48317fd40548abf91fe2e6aa'
EVIDENCE_SHA256='d8fa8bac03d9b09f970f6bd328f9b295165c1e56823c799a46771886123cacd0'
SOURCE_MANIFEST_SHA256='3933cb6540f271e351357b6458ccaae39061e25bb9b6b3ab6dcbfd0c2ad488fd'
SEMANTICS_SHA256='bb5560806ea7a84a0cc5f88ec5d4adbea4004ec6b2560af6e4d8de31b3a88d3b'
PARENT_SEMANTICS_SHA256='6dccbfce89b3910050b0f69b9aa3784c7afae23b3875b42c59b03bcc4af6db1a'
WRAPPER_SHA256='ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008'
RESOLVER_SHA256='a7e37545490745a58731933b0e07db69843cfeea30e739ed554b82d099ec3d84'
COMPILER_SHA256='6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
GUARDIAN_BUILD_SHA256='af7c1098143d0aad108684646df4c72fecca03404557f5494206713486ca09b6'
GUARDIAN_POLICY_SHA256='64bb0118793fe46dcb392abc1a9212eb15bd55047461576a3ef1a6cefa3f17da'
GUARDIAN_MAIN_SHA256='99b6fe7e1c687c3a4e76cfe1585e4826e753f473dff8676dd287eb2f9e0021bc'
GUARDIAN_SELFTEST_SHA256='c9c7f839fb262dbf616716e3c5f0601bb03cfbcbbcfe2fbe09bd0b39894e2a9f'
PARENT_GATE_SHA256='e330d2f6711a9ddc45a22bccabc066ae76fcbd472a86c9566fb9b03a31e793e8'
TOOLCHAIN_SHA256='5feb92bb4a13a9ec55bb3b76732eb8a5dfdcc28bcc38632813e0e6655f1eaed5'
HARDWARE_SHA256='c6851804d7c88d44f6d2ca5f12cd53d93020cae489b3191747239d2c735a2f1d'
COMMAND_SHA256='67615e74e1187a46f78eaec0928248d63981f2ca268a605cd0dd69f0fae733e2'
TEST_COMMAND_SHA256='6f71ff8679394feea569ba5c51fa04c34322d77fa92b61911d7f970d41eb2991'
RESULT_CONTRACT_SHA256='f624718235225aa3cd0ba14e405cb1f6b17e707d7010079e0985b53dd176ea2f'
TEST_OUTPUT_SHA256='99986edcc08b7c7b4ba7a275d97709d9a11e828529a808b62ba711e48588d959'

PREEXEC_FRAME_SHA256='fc477360ba8e6ee9591d3b0638a8ed194d3f6fcf286b5fb5dc3c68b6ef07fd56'
FREEZE_FRAME_SHA256='66deb0da38bb55740598f062dcec125d435d2155b2c1fc019d7104d1dab902a6'
POLICY_MISSING_FRAME_SHA256='c0eee38f18b097a5c02789dc70f1675f87a7e91bb4d1e3872173e28f196e13b9'
POLICY_TIMEOUT_FRAME_SHA256='40ad28c32f710669fafc0f25bc2d7ddac32a113c08284e26112a56c8d1757790'
PYTHON_FRAME_SHA256='df4f9ff3855a4cc9d36863f8c6cc6afc03bf04ab8960184c2a6638833d275a20'
RUST_FRAME_SHA256='6cba014c33a5b3cd72d621b32429c243108edcccd9ab36ee9b1364b7205a98b8'
LLM_PROMOTION_FRAME_SHA256='d4a750a6be6a1f12e57437039456ba4b25e51cdbb68e3ad3db764e0fd628794f'
CPP_AUTHORITY_FRAME_SHA256='a46b8c7879623d3d1ff5cb8871b11c729fce8e620e24330a2f462ef6f8854c68'
PARITY_PREFREEZE_FRAME_SHA256='a74ff671ccadecc7080ad6785e22ae693c15748d30a1ea4d5bfeded5397a1d26'
PYTHON_TOOLCHAIN_SHA256='801730b8c774cd63b5f0ac0d9bac4210483ff3a17856b254a9c32a21990ebd82'
PYTHON_COMMAND_SHA256='897897cffd2386320d75e8d27dc760bda3e27168cefcf6bb968260ad92af6bc4'
RUST_TOOLCHAIN_SHA256='0be0ccd70aea4f17230926f5d26b605ce07255dc8cd9c611e95af22a15128a7d'
RUST_COMMAND_SHA256='55ad8cea7944a961cf393a3473ca218b48a0eabccf62d075511474376f09bae6'
ZERO='0 0 0 0 0 0 0 0'

fail() {
  printf 'pireus operator genesis bilinear: FAIL: %s\n' "$*" >&2
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
    grep -Fqx -- ' matrix=1128' "${path}" &&
    grep -Fqx -- ' commutator=90' "${path}" &&
    grep -Fqx -- ' associator=1848' "${path}" &&
    grep -Fqx -- ' relative_semantic_novelty=1' "${path}" &&
    grep -Fqx -- ' algebraic_novelty=0' "${path}" &&
    grep -Fqx -- ' frozen_match=1' "${path}" &&
    grep -Fqx -- ' frozen_mismatch_code=0' "${path}" &&
    [[ "$(wc -l < "${path}")" -eq 527 ]] &&
    [[ "$(wc -c < "${path}")" -eq 7840 ]]
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
  printf '9020 1 2 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${FIRST_SOURCE_SHA256}")" "${ZERO}" \
    "$(sha_limbs "${PARENT_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${COMMAND_SHA256}")" "${ZERO}" "${ZERO}"
}

freeze_frame() {
  local policy="${1:-1}"
  printf '9020 2 3 1 1 %s 1 1 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${policy}" "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PARENT_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${COMMAND_SHA256}")" \
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

parity_prefreeze_frame() {
  printf '9020 2 4 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
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
  [[ "${process_launched}" == false ]] ||
    fail "${label} process launched before denial"
  printf 'GUARDIAN_DECISION label=%s decision=%s process_launched=false\n' \
    "${label}" "${decision}"
}

require_hash "${ROOT}/${GARDEN_REL}" "${GARDEN_SHA256}"
require_hash "${ROOT}/${CONTRACT_REL}" "${CONTRACT_SHA256}"
require_hash "${ROOT}/${BASE_REL}" "${BASE_SHA256}"
require_hash "${ROOT}/${MODULE_REL}" "${MODULE_SHA256}"
require_hash "${ROOT}/${EXAMPLE_REL}" "${EXAMPLE_SHA256}"
require_hash "${ROOT}/${TEST_REL}" "${TEST_SHA256}"
require_hash "${ROOT}/${FREEZE_REL}" "${FREEZE_SHA256}"
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
require_hash "${ROOT}/scripts/ci/pireus_operator_genesis_gl4.sh" \
  "${PARENT_GATE_SHA256}"
[[ -x "${GUARDIAN}" ]] || fail 'native Loom Guardian unavailable'

git -C "${ROOT}" merge-base --is-ancestor "${GARDEN_COMMIT}" \
  "${EXECUTABLE_COMMIT}" || fail 'Garden does not precede Sounio executable'
git -C "${ROOT}" merge-base --is-ancestor "${EXECUTABLE_COMMIT}" \
  "${SEMANTIC_FREEZE_COMMIT}" || fail 'first executable does not precede matcher'
git -C "${ROOT}" merge-base --is-ancestor "${SEMANTIC_FREEZE_COMMIT}" \
  "${RECEIPT_COMMIT}" || fail 'matcher does not precede receipt'
git -C "${ROOT}" merge-base --is-ancestor "${RECEIPT_COMMIT}" HEAD ||
  fail 'receipt is not an ancestor of HEAD'

[[ "$(git -C "${ROOT}" show "${EXECUTABLE_COMMIT}:${MODULE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${FIRST_SOURCE_SHA256}" ]] || fail 'first executable source hash drift'
if git -C "${ROOT}" show "${EXECUTABLE_COMMIT}:${MODULE_REL}" |
    grep -Fq 'pireus_operator_genesis_bilinear_matches_frozen_semantics'; then
  fail 'frozen matcher existed in first executable commit'
fi
[[ "$(git -C "${ROOT}" show "${SEMANTIC_FREEZE_COMMIT}:${MODULE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${MODULE_SHA256}" ]] || fail 'frozen matcher source hash drift'
[[ "$(git -C "${ROOT}" show "${SEMANTIC_FREEZE_COMMIT}:${EXAMPLE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${EXAMPLE_SHA256}" ]] || fail 'frozen example hash drift'
[[ "$(git -C "${ROOT}" show "${SEMANTIC_FREEZE_COMMIT}:${TEST_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${TEST_SHA256}" ]] || fail 'frozen test hash drift'
[[ "$(git -C "${ROOT}" show "${RECEIPT_COMMIT}:${CONTRACT_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${CONTRACT_SHA256}" ]] || fail 'receipt contract object drift'
[[ "$(git -C "${ROOT}" show "${RECEIPT_COMMIT}:${FREEZE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${FREEZE_SHA256}" ]] || fail 'freeze receipt object drift'
[[ "$(git -C "${ROOT}" show "${RECEIPT_COMMIT}:${EVIDENCE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${EVIDENCE_SHA256}" ]] || fail 'evidence object drift'

actual_manifest="$({
  cd "${ROOT}"
  sha256sum "${MODULE_REL}" "${EXAMPLE_REL}" "${TEST_REL}"
} | sha256sum | cut -d' ' -f1)"
[[ "${actual_manifest}" == "${SOURCE_MANIFEST_SHA256}" ]] ||
  fail 'source manifest drift'
actual_semantics="$(cat "${ROOT}/${MODULE_REL}" "${ROOT}/${EXAMPLE_REL}" \
  "${ROOT}/${TEST_REL}" | sha256sum | cut -d' ' -f1)"
[[ "${actual_semantics}" == "${SEMANTICS_SHA256}" ]] ||
  fail 'semantics bundle drift'

toolchain_record="$(printf '%s\n' \
  'engine=lean_single' 'wrapper=bin/souc' \
  "wrapper_sha256=${WRAPPER_SHA256}" \
  'resolver=scripts/lib/resolve_souc.sh' \
  "resolver_sha256=${RESOLVER_SHA256}" \
  'compiler=bin/souc-lean-single-x86_64' \
  "compiler_sha256=${COMPILER_SHA256}")"
[[ "$(sha_text "${toolchain_record}")" == "${TOOLCHAIN_SHA256}" ]] ||
  fail 'toolchain receipt drift'
live_kernel="$(uname -s) $(uname -r)"
live_architecture="$(uname -m)"
live_logical_cpus="$(getconf _NPROCESSORS_ONLN)"
live_cpu_model="$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -n 1)"
declare -A live_sockets=()
declare -A live_cores=()
while IFS=, read -r cpu socket core; do
  [[ "${cpu}" == \#* ]] && continue
  [[ -n "${cpu}" && -n "${socket}" && -n "${core}" ]] || continue
  live_sockets["${socket}"]=1
  live_cores["${socket},${core}"]=1
done < <(lscpu -p=CPU,SOCKET,CORE)
live_socket_count="${#live_sockets[@]}"
[[ "${live_socket_count}" -gt 0 ]] || fail 'live socket discovery failed'
live_cores_per_socket=$((${#live_cores[@]} / live_socket_count))
hardware_record="$(printf '%s\n' \
  "kernel=${live_kernel}" "architecture=${live_architecture}" \
  "logical_cpus=${live_logical_cpus}" "cpu_model=${live_cpu_model}" \
  "sockets=${live_socket_count}" \
  "cores_per_socket=${live_cores_per_socket}")"
[[ "${live_kernel}" == 'Linux 7.0.2-5-pve' &&
    "${live_architecture}" == 'x86_64' &&
    "${live_logical_cpus}" == '64' &&
    "${live_cpu_model}" == 'INTEL(R) XEON(R) GOLD 6526Y' &&
    "${live_socket_count}" == '2' &&
    "${live_cores_per_socket}" == '16' ]] ||
  fail 'live Xeon hardware does not match frozen receipt'
[[ "$(sha_text "${hardware_record}")" == "${HARDWARE_SHA256}" ]] ||
  fail 'hardware receipt drift'
[[ "$(sha_text 'SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pireus_operator_genesis_bilinear.sio')" == \
  "${COMMAND_SHA256}" ]] || fail 'authority command receipt drift'
[[ "$(sha_text 'SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/stdlib/hardware/test_pireus_operator_genesis_bilinear.sio')" == \
  "${TEST_COMMAND_SHA256}" ]] || fail 'test command receipt drift'

result_record="$(printf '%s\n' \
  'raw_candidates=65536' 'quadratic_codes=1024' 'bucket_size=64' \
  'gauge_fiber_checks=65536' 'gauge_fiber_failures=0' \
  'gl4_matrices=20160' 'declared_actions=40320' \
  'admitted_actions=336' 'admitted_no_swap=168' 'admitted_swap=168' \
  'affine_inverse_checks=344064' 'declared_class_count=32' \
  'class_representatives=0,1,8,9,16,19,20,24,27,28,64,65,66,72,73,74,80,82,83,84,88,90,91,92,192,193,198,199,200,201,206,207' \
  'class_minimum_matrices=0,1,32768,32769,2,35,36,32770,32803,32804,8,9,40,32776,32777,32808,10,42,43,44,32778,32810,32811,32812,72,73,1128,1129,32840,32841,33896,33897' \
  'class_quadratic_sizes=1,7,1,7,21,7,28,21,7,28,7,7,42,7,7,42,42,21,21,84,42,21,21,84,84,84,28,28,84,84,28,28' \
  'v1_classes=4' 'corpus_membership_hits=0,336,0' \
  'selected_class=26' 'selected_quadratic_code=198' \
  'selected_matrix=1128' 'selected_quadratic_size=28' \
  'selected_raw_size=1792' 'selected_square_negatives=5' \
  'selected_commutator_defects=90' 'selected_associator_defects=1848' \
  'selected_nearest_corpus=2' 'selected_structural_delta=0,120,10' \
  'negative_cases=21' 'relative_semantic_novelty=true' \
  'relative_grammar_extension_novelty=true' \
  'relative_algebraic_novelty=false' 'algebra_isomorphism_complete=false' \
  'all_sign_tables_exhausted=false' 'orbit_hamming_distance=false' \
  'algorithmic_novelty=false' 'material_novelty=false' \
  'scientific_novelty=false' 'global_novelty=false' \
  'historical_novelty=false' 'priority_claim=false' \
  'parity_open=false' 'claim_ready=false')"
[[ "$(sha_text "${result_record}")" == "${RESULT_CONTRACT_SHA256}" ]] ||
  fail 'result contract receipt drift'

require_line "${ROOT}/${FREEZE_REL}" 'stage=SEMANTICS_FROZEN'
require_line "${ROOT}/${FREEZE_REL}" 'producing_language=Sounio'
require_line "${ROOT}/${FREEZE_REL}" 'language_role=SEMANTIC_AUTHORITY'
require_line "${ROOT}/${FREEZE_REL}" 'expected_declared_class_count=32'
require_line "${ROOT}/${FREEZE_REL}" 'expected_selected_matrix=1128'
require_line "${ROOT}/${FREEZE_REL}" 'relative_semantic_novelty=true'
require_line "${ROOT}/${FREEZE_REL}" 'relative_grammar_extension_novelty=true'
require_line "${ROOT}/${FREEZE_REL}" 'relative_algebraic_novelty=false'
require_line "${ROOT}/${FREEZE_REL}" 'all_sign_tables_exhausted=false'
require_line "${ROOT}/${FREEZE_REL}" 'parity_open=false'
require_line "${ROOT}/${FREEZE_REL}" 'claim_ready=false'
transcript_admitted "${ROOT}/${EVIDENCE_REL}" ||
  fail 'canonical authority transcript was not admitted'

"${ROOT}/scripts/ci/sounio_loom_language_authority_selftest.sh" >/dev/null
authorize PREEXEC "$(preexec_frame)" "${PREEXEC_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE'
authorize FREEZE "$(freeze_frame 1)" "${FREEZE_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'
attempt_forbidden_oracle PYTHON_ORACLE 7 "${PYTHON_TOOLCHAIN_SHA256}" \
  "${PYTHON_COMMAND_SHA256}" "${PYTHON_FRAME_SHA256}"
attempt_forbidden_oracle RUST_ORACLE 8 "${RUST_TOOLCHAIN_SHA256}" \
  "${RUST_COMMAND_SHA256}" "${RUST_FRAME_SHA256}"
authorize POLICY_MISSING "$(freeze_frame 0)" \
  "${POLICY_MISSING_FRAME_SHA256}" 101 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=101 reason=policy-missing next_stage=SOUNIO_EXECUTABLE'
authorize POLICY_TIMEOUT "$(freeze_frame 2)" \
  "${POLICY_TIMEOUT_FRAME_SHA256}" 102 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=102 reason=policy-timeout next_stage=SOUNIO_EXECUTABLE'
authorize PARITY_PREFREEZE "$(parity_prefreeze_frame)" \
  "${PARITY_PREFREEZE_FRAME_SHA256}" 112 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=112 reason=wrong-stage next_stage=SOUNIO_EXECUTABLE'
authorize LLM_PROMOTION "$(llm_promotion_frame)" \
  "${LLM_PROMOTION_FRAME_SHA256}" 119 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=119 reason=review-promoted-to-authority next_stage=SEMANTICS_FROZEN'
authorize CPP_AUTHORITY "$(cpp_authority_frame)" \
  "${CPP_AUTHORITY_FRAME_SHA256}" 113 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=SEMANTICS_FROZEN'

umask 077
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/pireus-bilinear-v2.XXXXXX")"
cleanup() { rm -rf "${TMP_ROOT}"; }
trap cleanup EXIT

(
  cd "${ROOT}"
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check "${EXAMPLE_REL}" >/dev/null
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check "${TEST_REL}" >/dev/null
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run "${EXAMPLE_REL}" \
    > "${TMP_ROOT}/authority.txt"
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run "${TEST_REL}" \
    > "${TMP_ROOT}/test.txt"
)
cmp -s "${TMP_ROOT}/authority.txt" "${ROOT}/${EVIDENCE_REL}" ||
  fail 'authority transcript replay drift'
[[ "$(sha_file "${TMP_ROOT}/authority.txt")" == "${EVIDENCE_SHA256}" ]] ||
  fail 'authority transcript hash drift'
[[ "$(sha_file "${TMP_ROOT}/test.txt")" == "${TEST_OUTPUT_SHA256}" ]] ||
  fail 'structural test output drift'

cp "${ROOT}/${EVIDENCE_REL}" "${TMP_ROOT}/tamper-class.txt"
sed -i 's/^ class=26$/ class=25/' "${TMP_ROOT}/tamper-class.txt"
if transcript_admitted "${TMP_ROOT}/tamper-class.txt"; then
  fail 'class-tampered transcript was admitted'
fi
cp "${ROOT}/${EVIDENCE_REL}" "${TMP_ROOT}/tamper-matrix.txt"
sed -i 's/^ matrix=1128$/ matrix=1129/' "${TMP_ROOT}/tamper-matrix.txt"
if transcript_admitted "${TMP_ROOT}/tamper-matrix.txt"; then
  fail 'matrix-tampered transcript was admitted'
fi
cp "${ROOT}/${EVIDENCE_REL}" "${TMP_ROOT}/tamper-digest.txt"
sed -i 's/^:327108787$/:327108788/' "${TMP_ROOT}/tamper-digest.txt"
if transcript_admitted "${TMP_ROOT}/tamper-digest.txt"; then
  fail 'digest-tampered transcript was admitted'
fi

"${ROOT}/scripts/ci/pireus_operator_genesis_gl4.sh" >/dev/null

printf '%s\n' \
  'pireus operator genesis bilinear: PASS stage=SEMANTICS_FROZEN language=Sounio raw=65536 quotient=1024 stabilizer=336 classes=32 selected_matrix=1128 relative_semantic_novelty=true relative_grammar_extension_novelty=true algebraic_novelty=false claim_ready=false python_process_launched=false rust_process_launched=false'
