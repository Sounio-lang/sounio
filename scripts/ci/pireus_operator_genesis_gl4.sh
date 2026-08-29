#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"

GARDEN_REL='tools/pireus/GARDEN_PIREUS_OPERATOR_GENESIS_GL4_V1.md'
CONTRACT_REL='tools/pireus/PIREUS_OPERATOR_GENESIS_GL4_CONTRACT_V1.md'
BASE_REL='stdlib/algebra/cayley_dickson.sio'
MODULE_REL='stdlib/hardware/pireus/operator_genesis_gl4.sio'
EXAMPLE_REL='examples/pireus_operator_genesis_gl4.sio'
TEST_REL='tests/stdlib/hardware/test_pireus_operator_genesis_gl4.sio'
FREEZE_REL='tools/pireus/operator_genesis_gl4.freeze.v1'
EVIDENCE_REL='tools/pireus/evidence/operator_genesis_gl4_v1.txt'

GARDEN_COMMIT='47a1e8a3ef96f71e1cb941ab0b92d81a5e30a0f7'
EXECUTABLE_COMMIT='fd2ff888712a2c085a683d4db5c20b8aa06ecfbc'
CLASS_EXPOSURE_COMMIT='c62eb1120ef6ec776b9b85205c5eeb8da8938f7b'
SEMANTIC_KEY_REPAIR_COMMIT='01d8645fb841d6c9f0da85384c7baf1e2cf22008'
SEMANTIC_FREEZE_COMMIT='2915618ce52ee42bc510c290d3336e559a94ce14'

GARDEN_SHA256='a37a27c9cc592a88031569eea128f25533cff43100448b6c7a241a6427c51ca8'
CONTRACT_SHA256='14c18fdcd5ac44c6bbadaa88816f1a4b1222414223d11f7771483bb9da874da7'
BASE_SHA256='e7dd98de0644013ebf6e0d435fddb7f893720f684c96c3fbe20cc11b1f518fed'
FIRST_SOURCE_SHA256='1b53e67145ee44a8f3001cf33c1b3327fafacb51e064f2b37eeddf62b4befb1b'
CLASS_SOURCE_SHA256='89fc9788c3a08065cd954034e25bf0b60b63f491a868d279d42925559355e97f'
REPAIR_SOURCE_SHA256='383b3a8a7121b497e594ba47b340a737f859689b194c45695f92ee88d0344847'
MODULE_SHA256='d8920404cd958f70234a340750264ca594767c2ed7973415f5bf4dbb3737a8da'
EXAMPLE_SHA256='0119fedf22f68c8c6faa1467db644672e46a24d2310a8ca409875ccb3e7f664c'
TEST_SHA256='342f05456337a1e3880cf7b2320655904473c7c8f3818b206eee1eae1d2e21c7'
FREEZE_SHA256='22e94dcd67fee406f5c79a7e98da5f1308e2e0509a7cef972d85c69e93786ab5'
EVIDENCE_SHA256='c4c8788968425aea59e671e5c474f1db074bf63daf9b24be750537ea7eeb38a6'
SOURCE_MANIFEST_SHA256='5b8105ad9f7ae57de90cbdd4f438930e0dbd9726cd8796bb28bc0784f2121dbc'
SEMANTICS_SHA256='6dccbfce89b3910050b0f69b9aa3784c7afae23b3875b42c59b03bcc4af6db1a'
PARENT_SEMANTICS_SHA256='6ae5a589fecc8c6545680ee996431cfce87a6beb9e0e300fe8041fa5107087e7'
WRAPPER_SHA256='ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008'
RESOLVER_SHA256='a7e37545490745a58731933b0e07db69843cfeea30e739ed554b82d099ec3d84'
COMPILER_SHA256='6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2'
TOOLCHAIN_SHA256='5feb92bb4a13a9ec55bb3b76732eb8a5dfdcc28bcc38632813e0e6655f1eaed5'
HARDWARE_SHA256='c6851804d7c88d44f6d2ca5f12cd53d93020cae489b3191747239d2c735a2f1d'
COMMAND_SHA256='fb7d6bc8fb4671920cbf4d3d0dc1606a5fd9222adc4bd1a0a5cf0b38989ccb9d'
TEST_COMMAND_SHA256='1cf72bcd6b65b78fdaac0d8ae6ac125c4d07685e92f5f60cc7c8556f10506aaa'
RESULT_CONTRACT_SHA256='cd3025a97d12926ea2fbbc3c9bcdccce9777bf09ada4e87f851dba11d453d51d'
TEST_OUTPUT_SHA256='9c014e83b261f9ac0ffee7f6b2cdd12be253553f725ac30ef9e69fa4e2c1d470'

PREEXEC_FRAME_SHA256='7b33233d70b4dffa95543b58bba5da695232fac8cd7686909c8073f302658798'
FREEZE_FRAME_SHA256='1eff1c1bf096e83782c1bd4973ca01ea817825110148962f5d7f5db21d5a518f'
PYTHON_FRAME_SHA256='bc6acce4a4966eb9f55f49a6d2ef0b57b13cb8b74fed0ff790a588d4ff0d807d'
POLICY_MISSING_FRAME_SHA256='0a4a631c6ea635589c5a0fe457d9d76d90c2795606924f217478830d6ce5265a'
POLICY_TIMEOUT_FRAME_SHA256='afc4c6b69a4ae395def290ffbd5baa7d9cb1dbc59c906d037f1a79a9520d496e'
LLM_PROMOTION_FRAME_SHA256='5965ff6267ced9f5c73cdd713a1be6490391387689987d9c2163188b6ef01d3b'
CPP_AUTHORITY_FRAME_SHA256='8fde098f1582d5a727ad95b48849e268bbf88c96de40f2a613e7cf2da605f390'
PYTHON_TOOLCHAIN_SHA256='497ce0938df96d9bf3c159472d251a946c2d6bd832220937d1b885f7759b05ba'
PYTHON_COMMAND_SHA256='6ac158e899dcd452ab7e2284cefb16721c50b6bd9f72385c8bbc75a085e8b3e5'
ZERO='0 0 0 0 0 0 0 0'

fail() {
  printf 'pireus operator genesis gl4: FAIL: %s\n' "$*" >&2
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

attempt_python_oracle() {
  local frame decision rc python_process_launched=false
  frame="$(python_frame)"
  [[ "$(sha_text "${frame}")" == "${PYTHON_FRAME_SHA256}" ]] ||
    fail 'Guardian frame drift: PYTHON_ORACLE'
  set +e
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  rc=$?
  set -e
  if [[ "${rc}" -eq 0 ]]; then
    python_process_launched=true
    fail 'Guardian allowed Python oracle; launch refused by outer fail-closed gate'
  fi
  [[ "${rc}" -eq 110 ]] || fail "Python denial exit drift: ${rc}"
  [[ "${decision}" == \
    'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN' ]] ||
    fail "Python denial drift: ${decision}"
  [[ "${python_process_launched}" == false ]] ||
    fail 'Python process launched before denial'
  printf 'GUARDIAN_DECISION label=PYTHON_ORACLE decision=%s python_process_launched=false\n' \
    "${decision}"
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
[[ -x "${GUARDIAN}" ]] || fail 'native Loom Guardian unavailable'

git -C "${ROOT}" merge-base --is-ancestor "${GARDEN_COMMIT}" \
  "${EXECUTABLE_COMMIT}" || fail 'Garden does not precede Sounio executable'
git -C "${ROOT}" merge-base --is-ancestor "${EXECUTABLE_COMMIT}" \
  "${CLASS_EXPOSURE_COMMIT}" || fail 'first executable does not precede class exposure'
git -C "${ROOT}" merge-base --is-ancestor "${CLASS_EXPOSURE_COMMIT}" \
  "${SEMANTIC_KEY_REPAIR_COMMIT}" || fail 'semantic-key repair order drift'
git -C "${ROOT}" merge-base --is-ancestor "${SEMANTIC_KEY_REPAIR_COMMIT}" \
  "${SEMANTIC_FREEZE_COMMIT}" || fail 'semantic-key repair does not precede matcher'
git -C "${ROOT}" merge-base --is-ancestor "${SEMANTIC_FREEZE_COMMIT}" HEAD ||
  fail 'semantic freeze is not an ancestor of HEAD'

[[ "$(git -C "${ROOT}" show "${EXECUTABLE_COMMIT}:${MODULE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${FIRST_SOURCE_SHA256}" ]] || fail 'first executable source hash drift'
if git -C "${ROOT}" show "${EXECUTABLE_COMMIT}:${MODULE_REL}" |
    grep -Fq 'pireus_operator_genesis_gl4_matches_frozen_semantics'; then
  fail 'frozen matcher existed in first executable commit'
fi
[[ "$(git -C "${ROOT}" show "${CLASS_EXPOSURE_COMMIT}:${MODULE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${CLASS_SOURCE_SHA256}" ]] || fail 'canonical class source hash drift'
[[ "$(git -C "${ROOT}" show "${SEMANTIC_KEY_REPAIR_COMMIT}:${MODULE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${REPAIR_SOURCE_SHA256}" ]] || fail 'semantic-key repair source hash drift'
[[ "$(git -C "${ROOT}" show "${SEMANTIC_FREEZE_COMMIT}:${MODULE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${MODULE_SHA256}" ]] || fail 'frozen matcher source hash drift'
[[ "$(git -C "${ROOT}" show "${SEMANTIC_FREEZE_COMMIT}:${EXAMPLE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${EXAMPLE_SHA256}" ]] || fail 'frozen example hash drift'
[[ "$(git -C "${ROOT}" show "${SEMANTIC_FREEZE_COMMIT}:${TEST_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${TEST_SHA256}" ]] || fail 'frozen test hash drift'

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
hardware_record="$(printf '%s\n' \
  'kernel=Linux 7.0.2-5-pve' 'architecture=x86_64' 'logical_cpus=64' \
  'cpu_model=INTEL(R) XEON(R) GOLD 6526Y' 'sockets=2' \
  'cores_per_socket=16')"
[[ "$(sha_text "${hardware_record}")" == "${HARDWARE_SHA256}" ]] ||
  fail 'hardware receipt drift'
[[ "$(sha_text 'SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pireus_operator_genesis_gl4.sio')" == \
  "${COMMAND_SHA256}" ]] || fail 'authority command receipt drift'
[[ "$(sha_text 'SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/stdlib/hardware/test_pireus_operator_genesis_gl4.sio')" == \
  "${TEST_COMMAND_SHA256}" ]] || fail 'test command receipt drift'

result_record="$(printf '%s\n' \
  'candidate_count=16' 'corpus_count=3' \
  'corpus_identities=untwisted_xor,cayley_dickson_16,diagonal_bicharacter' \
  'declared_equivalence_class_count=4' \
  'candidate_masks=9,10,11,12,13,14,15,0,1,2,3,4,5,6,7,8' \
  'candidate_distances=96,96,96,96,96,96,96,0,96,96,96,96,96,96,96,112' \
  'candidate_classes=0,0,0,0,0,0,0,1,2,2,2,2,2,2,2,3' \
  'matrix_encodings=65536' 'gl4_matrices=20160' \
  'gauge_functions=32768' 'gauge_kernel=16' 'gauge_actions=2048' \
  'declared_action_universe=82575360' 'selected_ordinal=15' \
  'selected_mask=8' 'canonical_quotient_distance=112' \
  'canonical_matrix=62024' 'canonical_swap=false' 'nearest_corpus=1' \
  'witness_i=1' 'witness_j=6' 'witness_candidate_sign=-1' \
  'witness_corpus_sign=1' 'positive=144' 'negative=112' \
  'commutator_defects=210' 'associator_defects=1848' \
  'displacement_negative_counts=7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7' \
  'negative_cases=30' 'relative_semantic_novelty=true' \
  'declared_gl4_gauge_inequivalence=true' \
  'relative_monomial_gauge_novelty=true' \
  'relative_algebraic_novelty=false' 'algebra_isomorphism_complete=false' \
  'all_sign_tables_exhausted=false' 'global_novelty=false' \
  'scientific_novelty=false' 'parity_open=false' 'claim_ready=false')"
[[ "$(sha_text "${result_record}")" == "${RESULT_CONTRACT_SHA256}" ]] ||
  fail 'result contract receipt drift'

require_line "${ROOT}/${FREEZE_REL}" 'stage=SEMANTICS_FROZEN'
require_line "${ROOT}/${FREEZE_REL}" 'producing_language=Sounio'
require_line "${ROOT}/${FREEZE_REL}" 'language_role=SEMANTIC_AUTHORITY'
require_line "${ROOT}/${FREEZE_REL}" 'expected_corpus_count=3'
require_line "${ROOT}/${FREEZE_REL}" 'expected_corpus_identities=untwisted_xor,cayley_dickson_16,diagonal_bicharacter'
require_line "${ROOT}/${FREEZE_REL}" 'expected_declared_equivalence_class_count=4'
require_line "${ROOT}/${FREEZE_REL}" 'expected_phase_mask=8'
require_line "${ROOT}/${FREEZE_REL}" 'expected_canonical_quotient_distance=112'
require_line "${ROOT}/${FREEZE_REL}" 'relative_algebraic_novelty=false'
require_line "${ROOT}/${FREEZE_REL}" 'algebra_isomorphism_complete=false'
require_line "${ROOT}/${FREEZE_REL}" 'all_sign_tables_exhausted=false'
require_line "${ROOT}/${FREEZE_REL}" 'parity_open=false'
require_line "${ROOT}/${FREEZE_REL}" 'claim_ready=false'
require_line "${ROOT}/${EVIDENCE_REL}" ' frozen_match=1'
require_line "${ROOT}/${EVIDENCE_REL}" ' frozen_mismatch_code=0'
require_line "${ROOT}/${EVIDENCE_REL}" ' relative_algebraic_novelty=0'
require_line "${ROOT}/${EVIDENCE_REL}" ' declared_gl4_gauge_inequivalence=1'
require_line "${ROOT}/${EVIDENCE_REL}" 'PIREUS_GL4_DECLARED_EQUIVALENCE_CLASSES count=4'
[[ "$(wc -l < "${ROOT}/${EVIDENCE_REL}")" -eq 791 ]] ||
  fail 'authority output line-count drift'
[[ "$(wc -c < "${ROOT}/${EVIDENCE_REL}")" -eq 9322 ]] ||
  fail 'authority output byte-count drift'

"${ROOT}/scripts/ci/sounio_loom_language_authority_selftest.sh" >/dev/null
authorize PREEXEC "$(preexec_frame)" "${PREEXEC_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE'
authorize FREEZE "$(freeze_frame 1)" "${FREEZE_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'
attempt_python_oracle
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

work="$(mktemp -d "${TMPDIR:-/tmp}/pireus-operator-genesis-gl4-v1.XXXXXX")"
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

cp "${ROOT}/${EVIDENCE_REL}" "${work}/tampered-winner.txt"
sed -i '0,/ mask=8/s// mask=7/' "${work}/tampered-winner.txt"
[[ "$(sha_file "${work}/tampered-winner.txt")" != "${EVIDENCE_SHA256}" ]] ||
  fail 'tampered winner retained authority hash'
cmp -s "${work}/tampered-winner.txt" "${ROOT}/${EVIDENCE_REL}" &&
  fail 'tampered winner was accepted as byte-identical'

cp "${ROOT}/${EVIDENCE_REL}" "${work}/tampered-digest.txt"
sed -i '0,/canonical_digest=4064601639/s//canonical_digest=4064601638/' \
  "${work}/tampered-digest.txt"
[[ "$(sha_file "${work}/tampered-digest.txt")" != "${EVIDENCE_SHA256}" ]] ||
  fail 'tampered digest retained authority hash'
cmp -s "${work}/tampered-digest.txt" "${ROOT}/${EVIDENCE_REL}" &&
  fail 'tampered digest was accepted as byte-identical'

printf '%s\n' \
  'PIREUS_OPERATOR_GENESIS_GL4_GATE_PASS=true stage=SEMANTICS_FROZEN language=Sounio role=SEMANTIC_AUTHORITY candidates=16 declared_classes=4 gl4=20160 gauges=32768 gauge_actions=2048 declared_action_universe=82575360 selected_mask=8 canonical_quotient_distance=112 relative_semantic_novelty=true declared_gl4_gauge_inequivalence=true relative_monomial_gauge_novelty=true relative_algebraic_novelty=false algebra_isomorphism_complete=false all_sign_tables_exhausted=false python_oracle=E110 python_process_launched=false llm_promotion=E119 cpp_authority=E113 policy_missing=E101 policy_timeout=E102 tampered_winner=REFUSED tampered_digest=REFUSED formal_parity_open=false effect_parity_open=false material_parity_count=0 parity_open=false claim_ready=false engine=lean_single explicit_bootstrap_fallback=true'
