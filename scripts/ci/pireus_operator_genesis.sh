#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"

GARDEN_REL='tools/pireus/GARDEN_PIREUS_OPERATOR_GENESIS_V0.md'
BASE_REL='stdlib/algebra/cayley_dickson.sio'
MODULE_REL='stdlib/hardware/pireus/operator_genesis.sio'
EXAMPLE_REL='examples/pireus_operator_genesis.sio'
TEST_REL='tests/stdlib/hardware/test_pireus_operator_genesis.sio'
FREEZE_REL='tools/pireus/operator_genesis.freeze.v0'
EVIDENCE_REL='tools/pireus/evidence/operator_genesis_v0.txt'

GARDEN_COMMIT='aaef53eb0a6f15a6d0041f347cc107ed69310de4'
EXECUTABLE_COMMIT='d034de5927eee7e4382c39926c5d5ab79a347a79'
FREEZE_COMMIT='999efd27bc6def4cf0756f870568302070659363'
STATIC_REPAIR_COMMIT='4ce3307d544d5191a68c2372dfc266ced526a70a'

GARDEN_SHA256='13501a4f3e16d6a593e260a7a9e3bc9c44c3063e11b51ab3c9b4ee73f24c6355'
BASE_SHA256='e7dd98de0644013ebf6e0d435fddb7f893720f684c96c3fbe20cc11b1f518fed'
FIRST_SOURCE_SHA256='94fc6e8da8fd0cc4871d3f70e9b6ad5829bac83701b5fef2daf1aa9fda42c23d'
ORIGINAL_FROZEN_SOURCE_SHA256='21570e60d8f46699b7c78c19ae5f76380a881b8308feb61c6a743be4703ac8a8'
MODULE_SHA256='f77d74a75a952b1d59d3f755ae72828f4503fed942f5fd732bc9069e9fde001f'
EXAMPLE_SHA256='654996673644089d586693bdba7c6a2df63e7ab34c138762fcd2c614f5fa4593'
TEST_SHA256='8cae5eeb1b4570529651da9e4438cc72d5ac8662ed7d216ec3c45b08d9d58b3e'
ORIGINAL_FREEZE_SHA256='42bd58a239bf22b7d416fe21d07099e8e86b7e6f4c054d570879e3f210d56559'
FREEZE_SHA256='62154fb985a3c3c6b4bb56708405fdd9ff7aeee7f0ff96197ab081a99c690db9'
EVIDENCE_SHA256='baee74e5c174b2d8581cf5fb49346441b431e011bedaa677144bbf1ca977ac80'
SOURCE_MANIFEST_SHA256='4f0c539c484bb89fd2b8d9351de0c0514c9b8ff5b09a172d319c654194aa559b'
SEMANTICS_SHA256='6ae5a589fecc8c6545680ee996431cfce87a6beb9e0e300fe8041fa5107087e7'
PARENT_SEMANTICS_SHA256='da782da938ee5f9e0a49cb1f95dfbb6acac8aa706c9eb6d711565adcb9031502'
WRAPPER_SHA256='ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008'
COMPILER_SHA256='6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2'
TOOLCHAIN_SHA256='0dd7961c7b9b16f0fd218092c651e9181e91cb1e1e4631fd17f0a756452c1556'
HARDWARE_SHA256='c6851804d7c88d44f6d2ca5f12cd53d93020cae489b3191747239d2c735a2f1d'
COMMAND_SHA256='16cfc7a50dab9b522d4b6271d477ee0e41c396673c151189c211685f58a528bd'
TEST_COMMAND_SHA256='1c5fd8d624ec1c0c7105c613255ac7f89c9005fe79a2654ef9d034f42ad4eb8a'
RESULT_CONTRACT_SHA256='ec11a4337e4764f323c886fc2d314543fe04bc2fad1d481ff4aa3e0bd5fe8ba6'
TEST_OUTPUT_SHA256='51821c467b10128ec3ef439e6f926452007f00e98e8eba02583988b463f14f68'

PREEXEC_FRAME_SHA256='7da1d207631579193e5357f9657aea805358f538f02924c9537ce77a7af8c542'
FREEZE_FRAME_SHA256='cbae9a494d2ba178ab17501acd9f946326adfdb26cc72cfdba6e1a7a5f30102b'
PYTHON_FRAME_SHA256='026afd0059b8d61386bc2d2e25cd12b406b40841bc797c2b8c2f934d67275009'
POLICY_MISSING_FRAME_SHA256='dc672add237d0d14bf7548a800f73fbe28bc661c8bc3e0f59437f7fb4c22414e'
POLICY_TIMEOUT_FRAME_SHA256='cd75cd9285300fba2a127c328df26adee724d16be515c5dea1beabdd9ac1c413'
LLM_PROMOTION_FRAME_SHA256='ce61fce06bd44d360eb4e0c739378248de7117f9a204d59f4827404305c94bab'
CPP_AUTHORITY_FRAME_SHA256='578341d9fa6ade2b1983f9ade0f71b7426f3dabadfe523676904c6b782b54b66'
PYTHON_TOOLCHAIN_SHA256='55e9130a02434a36ba13ba9730cbc1972b7b7868a40bb1a5353252ac8b6b5bb2'
PYTHON_COMMAND_SHA256='b8ed4562af1c60571aa54ff3c061cdf117dc12a0405f6176d8925ee2e926353b'
ZERO='0 0 0 0 0 0 0 0'

fail() {
  printf 'pireus operator genesis: FAIL: %s\n' "$*" >&2
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

python_frame() {
  printf '9020 3 4 7 7 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PARENT_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PYTHON_TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${PYTHON_COMMAND_SHA256}")" "${ZERO}" "${ZERO}"
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

require_hash "${ROOT}/${GARDEN_REL}" "${GARDEN_SHA256}"
require_hash "${ROOT}/${BASE_REL}" "${BASE_SHA256}"
require_hash "${ROOT}/${MODULE_REL}" "${MODULE_SHA256}"
require_hash "${ROOT}/${EXAMPLE_REL}" "${EXAMPLE_SHA256}"
require_hash "${ROOT}/${TEST_REL}" "${TEST_SHA256}"
require_hash "${ROOT}/${FREEZE_REL}" "${FREEZE_SHA256}"
require_hash "${ROOT}/${EVIDENCE_REL}" "${EVIDENCE_SHA256}"
require_hash "${ROOT}/bin/souc" "${WRAPPER_SHA256}"
require_hash "${ROOT}/bin/souc-lean-single-x86_64" "${COMPILER_SHA256}"
[[ -x "${GUARDIAN}" ]] || fail 'native Loom Guardian unavailable'

git -C "${ROOT}" merge-base --is-ancestor "${GARDEN_COMMIT}" \
  "${EXECUTABLE_COMMIT}" || fail 'Garden does not precede Sounio executable'
git -C "${ROOT}" merge-base --is-ancestor "${EXECUTABLE_COMMIT}" \
  "${FREEZE_COMMIT}" || fail 'Sounio executable does not precede freeze'
git -C "${ROOT}" merge-base --is-ancestor "${FREEZE_COMMIT}" HEAD ||
  fail 'freeze commit is not an ancestor of HEAD'
git -C "${ROOT}" merge-base --is-ancestor "${FREEZE_COMMIT}" \
  "${STATIC_REPAIR_COMMIT}" || fail 'freeze does not precede static repair'
git -C "${ROOT}" merge-base --is-ancestor "${STATIC_REPAIR_COMMIT}" HEAD ||
  fail 'static repair commit is not an ancestor of HEAD'
[[ "$(git -C "${ROOT}" show "${EXECUTABLE_COMMIT}:${MODULE_REL}" | sha256sum | cut -d' ' -f1)" == "${FIRST_SOURCE_SHA256}" ]] ||
  fail 'first executable source hash drift'
if git -C "${ROOT}" show "${EXECUTABLE_COMMIT}:${MODULE_REL}" |
    grep -Fq 'pireus_operator_genesis_matches_frozen_semantics'; then
  fail 'frozen matcher existed in first executable commit'
fi
[[ "$(git -C "${ROOT}" show "${FREEZE_COMMIT}:${MODULE_REL}" | sha256sum | cut -d' ' -f1)" == "${ORIGINAL_FROZEN_SOURCE_SHA256}" ]] ||
  fail 'original frozen source hash drift'
[[ "$(git -C "${ROOT}" show "${FREEZE_COMMIT}:${FREEZE_REL}" | sha256sum | cut -d' ' -f1)" == "${ORIGINAL_FREEZE_SHA256}" ]] ||
  fail 'original freeze artifact hash drift'
[[ "$(git -C "${ROOT}" show "${STATIC_REPAIR_COMMIT}:${MODULE_REL}" | sha256sum | cut -d' ' -f1)" == "${MODULE_SHA256}" ]] ||
  fail 'static repair source hash drift'

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
  'compiler=bin/souc-lean-single-x86_64' \
  "compiler_sha256=${COMPILER_SHA256}")"
[[ "$(sha_text "${toolchain_record}")" == "${TOOLCHAIN_SHA256}" ]] ||
  fail 'toolchain receipt drift'
hardware_record="$(printf '%s\n' \
  'kernel=Linux 7.0.2-5-pve' 'architecture=x86_64' 'logical_cpus=64' \
  'cpu_model=INTEL(R) XEON(R) GOLD 6526Y' 'sockets=2' \
  'cores_per_socket=16')"
[[ "$(sha_text "${hardware_record}")" == "${HARDWARE_SHA256}" ]] ||
  fail 'hardware receipt drift'
[[ "$(sha_text 'SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pireus_operator_genesis.sio')" == "${COMMAND_SHA256}" ]] ||
  fail 'authority command receipt drift'
[[ "$(sha_text 'SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/stdlib/hardware/test_pireus_operator_genesis.sio')" == "${TEST_COMMAND_SHA256}" ]] ||
  fail 'test command receipt drift'
result_record="$(printf '%s\n' \
  'selected_ordinal=4' 'selected_mask=13' 'score=100' \
  'canonical_action=19' 'nearest_corpus=2' 'nearest_action=17' \
  'minimum_hamming_distance=100' 'comparisons=144' 'positive=144' \
  'negative=112' 'other=0' 'commutator_defects=210' \
  'associator_defects=1848' \
  'displacement_negative_counts=7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7' \
  'relative_semantic_novelty=true' \
  'declared_action_inequivalence=true' \
  'relative_algebraic_novelty=false' 'global_novelty=false' \
  'scientific_novelty=false' 'parity_open=false' 'claim_ready=false')"
[[ "$(sha_text "${result_record}")" == "${RESULT_CONTRACT_SHA256}" ]] ||
  fail 'result contract receipt drift'

require_line "${ROOT}/${FREEZE_REL}" 'stage=SEMANTICS_FROZEN'
require_line "${ROOT}/${FREEZE_REL}" 'producing_language=Sounio'
require_line "${ROOT}/${FREEZE_REL}" 'language_role=SEMANTIC_AUTHORITY'
require_line "${ROOT}/${FREEZE_REL}" 'static_check_repair=multiline-array-terminal-commas-removed'
require_line "${ROOT}/${FREEZE_REL}" 'static_check_repair_semantic_output=byte-identical'
require_line "${ROOT}/${FREEZE_REL}" 'expected_phase_mask=13'
require_line "${ROOT}/${FREEZE_REL}" 'expected_score=100'
require_line "${ROOT}/${FREEZE_REL}" 'relative_algebraic_novelty=false'
require_line "${ROOT}/${FREEZE_REL}" 'algebra_isomorphism_complete=false'
require_line "${ROOT}/${FREEZE_REL}" 'parity_open=false'
require_line "${ROOT}/${FREEZE_REL}" 'claim_ready=false'
require_line "${ROOT}/${EVIDENCE_REL}" ' frozen_match=1'
require_line "${ROOT}/${EVIDENCE_REL}" ' frozen_mismatch_code=0'
require_line "${ROOT}/${EVIDENCE_REL}" ' relative_algebraic_novelty=0'
require_line "${ROOT}/${EVIDENCE_REL}" ' declared_action_inequivalence=1'
[[ "$(wc -l < "${ROOT}/${EVIDENCE_REL}")" -eq 333 ]] ||
  fail 'authority output line-count drift'
[[ "$(wc -c < "${ROOT}/${EVIDENCE_REL}")" -eq 5236 ]] ||
  fail 'authority output byte-count drift'

"${ROOT}/scripts/ci/sounio_loom_language_authority_selftest.sh" >/dev/null
authorize PREEXEC "$(preexec_frame)" "${PREEXEC_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE'
authorize FREEZE "$(freeze_frame 1)" "${FREEZE_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'
authorize PYTHON_ORACLE "$(python_frame)" "${PYTHON_FRAME_SHA256}" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
authorize POLICY_MISSING "$(freeze_frame 0)" \
  "${POLICY_MISSING_FRAME_SHA256}" 101 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=101 reason=policy-missing next_stage=SOUNIO_EXECUTABLE'
authorize POLICY_TIMEOUT "$(freeze_frame 2)" \
  "${POLICY_TIMEOUT_FRAME_SHA256}" 102 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=102 reason=policy-timeout next_stage=SOUNIO_EXECUTABLE'
authorize LLM_PROMOTION "$(llm_promotion_frame)" \
  "${LLM_PROMOTION_FRAME_SHA256}" 119 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=119 reason=review-promoted-to-authority next_stage=SEMANTICS_FROZEN'
authorize CPP_AUTHORITY "$(cpp_authority_frame)" \
  "${CPP_AUTHORITY_FRAME_SHA256}" 113 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=SEMANTICS_FROZEN'

(
  cd "${ROOT}"
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check "${EXAMPLE_REL}" >/dev/null
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check "${TEST_REL}" >/dev/null
)

work="$(mktemp -d "${TMPDIR:-/tmp}/pireus-operator-genesis-v0.XXXXXX")"
trap 'rm -rf "${work}"' EXIT
(
  cd "${ROOT}"
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run "${EXAMPLE_REL}"
) >"${work}/authority.txt"
[[ "$(sha_file "${work}/authority.txt")" == "${EVIDENCE_SHA256}" ]] ||
  fail 'replayed authority output hash drift'
cmp -s "${work}/authority.txt" "${ROOT}/${EVIDENCE_REL}" ||
  fail 'replayed authority output is not byte-identical'
(
  cd "${ROOT}"
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run "${TEST_REL}"
) >"${work}/test.txt"
[[ "$(sha_file "${work}/test.txt")" == "${TEST_OUTPUT_SHA256}" ]] ||
  fail 'dedicated test output drift'

cp "${ROOT}/${EVIDENCE_REL}" "${work}/tampered.txt"
sed -i '0,/ mask=13/s// mask=12/' "${work}/tampered.txt"
[[ "$(sha_file "${work}/tampered.txt")" != "${EVIDENCE_SHA256}" ]] ||
  fail 'tampered transcript retained authority hash'
if cmp -s "${work}/tampered.txt" "${ROOT}/${EVIDENCE_REL}"; then
  fail 'tampered transcript was accepted as byte-identical'
fi

printf '%s\n' \
  'PIREUS_OPERATOR_GENESIS_GATE_PASS=true stage=SEMANTICS_FROZEN language=Sounio role=SEMANTIC_AUTHORITY candidates=16 actions=48 corpus=3 comparisons_per_candidate=144 selected_mask=13 score=100 nearest_corpus=2 minimum_hamming_distance=100 relative_semantic_novelty=true declared_action_inequivalence=true relative_algebraic_novelty=false algebra_isomorphism_complete=false python_oracle=E110 python_process_launched=false llm_promotion=E119 cpp_authority=E113 policy_missing=E101 policy_timeout=E102 tampered=REFUSED formal_parity_open=false effect_parity_open=false material_parity_count=0 parity_open=false claim_ready=false engine=lean_single explicit_bootstrap_fallback=true'
