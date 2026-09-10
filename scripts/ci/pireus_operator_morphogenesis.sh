#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN_CANDIDATE="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
GUARDIAN="$(readlink -f "${GUARDIAN_CANDIDATE}")"

GARDEN_REL='tools/pireus/GARDEN_PIREUS_OPERATOR_MORPHOGENESIS_V12.md'
CONTRACT_REL='tools/pireus/PIREUS_OPERATOR_MORPHOGENESIS_CONTRACT_V12.md'
MODULE_REL='stdlib/hardware/pireus/operator_morphogenesis.sio'
EXAMPLE_REL='examples/pireus_operator_morphogenesis.sio'
TEST_REL='tests/stdlib/hardware/test_pireus_operator_morphogenesis.sio'
FIRST_RECEIPT_REL='tools/pireus/operator_morphogenesis.first.v12'
FREEZE_REL='tools/pireus/operator_morphogenesis.freeze.v12'
DECISIONS_REL='tools/pireus/operator_morphogenesis.freeze-decisions.v12'
FIRST_EVIDENCE_REL='tools/pireus/evidence/operator_morphogenesis_v12.first.txt'
FIRST_TEST_EVIDENCE_REL='tools/pireus/evidence/operator_morphogenesis_v12.test.txt'
FROZEN_EVIDENCE_REL='tools/pireus/evidence/operator_morphogenesis_v12.frozen.txt'
FROZEN_TEST_EVIDENCE_REL='tools/pireus/evidence/operator_morphogenesis_v12.test.frozen.txt'

GARDEN_COMMIT='1ea2499e82aed21f3919800c90b1ff6e239a30f7'
EXECUTABLE_COMMIT='558e50f36bfe5891468d0be3c12a556f6e38de62'
FIRST_RESULT_COMMIT='5ea9d13143d808ed4e5f40ac9318f6a1443f3a84'
MATCHER_COMMIT='da1391c14c3d210478dab558ead44f57a1dfcba8'

GARDEN_SHA256='b8963c0c2b5b0bb9bb2d981f5fc92d9484a058153b95aa76ee3650b2b6cfd4a4'
CONTRACT_SHA256='8e2a5d119c332958c5d8146f0265536afff08bf4a4eae79315f64a4521c3d167'
FIRST_MODULE_SHA256='aed4228009f13e7b7b698978783c17149804c957258bfbcf8f6baf6b7dce2925'
FIRST_EXAMPLE_SHA256='ab5ea27abcdc028193e8c763fbebdbf91e234a34f4f359645bbf9121ebe3890b'
FIRST_TEST_SHA256='b5bb5e2c3677f7296a317b64a584e8c2a6299a88c6357cff2144a16958c4a00c'
MODULE_SHA256='0a637f7f3ac84ac501be337f22dff37e16a05dbc4a51d2090441b9cba4c8d05c'
EXAMPLE_SHA256='ffcbcfa657c6d9a580a5822e809f588903d557e91ea3eb847664b70e8d5cfc24'
TEST_SHA256='3d3c829860fb7c24861b2ea44d4704aca4792dd7acb54025af04eb4887bd490f'
FIRST_RECEIPT_SHA256='4663d114ff3d64d74a0c3c58ce40ecf85d0862275adc4bf62691af4761344c4c'
FREEZE_SHA256='14277a28f21a044bd55bd670b5b7447789c2f4e2780251c861ee4880ef739de7'
DECISIONS_SHA256='ae4de5ee4cc3820b63f9c9afc465ee205047a1fc885558f9da999c9abe5680bf'
FIRST_EVIDENCE_SHA256='148dc490e1f6aaaf672e85fd06411755b7521930f3de5998f4c98b32af25f816'
FIRST_TEST_EVIDENCE_SHA256='20c18e01d69da0835841559e9ea9ecca2954ceaa35e2b2b27b287ab5e9f6cbc2'
FROZEN_EVIDENCE_SHA256='3960e728416d2d2f8884d4f8e4d1ed277941f97b949aeb93601b6a1fdc39ad8d'
FROZEN_TEST_EVIDENCE_SHA256='20c18e01d69da0835841559e9ea9ecca2954ceaa35e2b2b27b287ab5e9f6cbc2'
SOURCE_MANIFEST_SHA256='fdb08abdd1a689dd8a3c50ae9ad16948f6de42137e8c2660fe38b1450f9e3cdf'
SEMANTICS_SHA256='999c6e7a0051f702cf40bb2adab7dc91c4f026230830096377f525005067c2f4'
PARENT_SEMANTICS_SHA256='e8268af20770dbf292fb39f92793b7b89d1651b2e88193e0cb6ee765dfc1f1ff'

WRAPPER_SHA256='ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008'
RESOLVER_SHA256='a7e37545490745a58731933b0e07db69843cfeea30e739ed554b82d099ec3d84'
COMPILER_SHA256='6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2'
TOOLCHAIN_SHA256='e3d50071e3b62a78f5a357cb3fea628445b5f77b3d72f1df6a94b835e89127fe'
HARDWARE_SHA256='c0f97ff854b02fab317b02e2916c493edc14c97a045a19e73e80cd6041a070df'
COMMAND_SHA256='323d52ecd3f5a6f76e5e661050a15c1ce499a4e065d2b5e6676ff0b7f9f02086'
TEST_COMMAND_SHA256='b5a60ac13a0b47ccd14a6d54d394931b5c4833e197cce9a8254095cc775e59f0'
PYTHON_TOOLCHAIN_SHA256='497ce0938df96d9bf3c159472d251a946c2d6bd832220937d1b885f7759b05ba'
PYTHON_COMMAND_SHA256='4dddce6f6b58207af55dadb95939fddc3a8894535747e3ddae964977004da116'
RUST_TOOLCHAIN_SHA256='b58640570ba9ffcdb2c2d241e4ce8ece9c7d75c6b1e59e308dee3e5f0e10b56d'
RUST_COMMAND_SHA256='a84ac699a1fa2fd6d6441ea9fa8639ebd34032fd753996ba712e4dd1c61df009'
PARITY_COMMAND_SHA256='b718d8973e9e9051a7710f34db0af2cd22fe6e8691ff9e31f04c6c27e2b1975c'
CPP_COMMAND_SHA256='c0e4637e6741d1ab8c4b9cfd389fddda3b65955f0a2fab3b08fc386d815bc85a'
LLM_COMMAND_SHA256='f897adc2714c243d39e2b9f9def8ff538ae39114529c5dba46fbed98e959b70d'
CLAIM_COMMAND_SHA256='2b68e58aca81ad0667e348d8f2c9ae4c856f551a032b53d9964ed3139f4f8154'

GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
GUARDIAN_POLICY_SHA256='64bb0118793fe46dcb392abc1a9212eb15bd55047461576a3ef1a6cefa3f17da'
GUARDIAN_FREEZE_SHA256='5fe5e5c9cdcb83935770f58df52f2d614d11f8abde519c4a2505ca20998fae2e'
GUARDIAN_BUILD_SHA256='af7c1098143d0aad108684646df4c72fecca03404557f5494206713486ca09b6'
GUARDIAN_MAIN_SHA256='99b6fe7e1c687c3a4e76cfe1585e4826e753f473dff8676dd287eb2f9e0021bc'
GUARDIAN_SELFTEST_SHA256='c9c7f839fb262dbf616716e3c5f0601bb03cfbcbbcfe2fbe09bd0b39894e2a9f'

MAIN_FREEZE_FRAME_SHA256='c15a92478e7a88a884442f19e93e0190965748a02fca2bb67496eb5f873fd042'
TEST_FREEZE_FRAME_SHA256='b7df22a29f5935f34c08c0f9c824c15501e56bf91f2c0594174353e570a16a1a'
PARITY_PREFREEZE_FRAME_SHA256='dd2ece4854e09f8a46ccf9d0515d809437d3ea8ce68850cb21a78c63609f64f3'
POLICY_MISSING_FRAME_SHA256='4a70c9aa4432bb9d8bd97cfb0a3ac04a348a5cfe85a6cc4da2a75d370817c11d'
POLICY_TIMEOUT_FRAME_SHA256='759dcd924de5836e9a1006f241ad31bd8af83ec86dfeee7dc4d51d16be9d0655'
POLICY_ERROR_FRAME_SHA256='fb25cc756e380f3b27237d825d0966da346d911313be1a8904986d9871816b1d'
PYTHON_FRAME_SHA256='9071290fa34fe3b5fcf7137f26e4fb26e8dd07d6f1312a3ac548c12d9c78f8f7'
RUST_FRAME_SHA256='042ef2e4760df31e48cd99b3c588e47aefd834386ed2d291061c2241d448b78d'
CPP_FRAME_SHA256='a127cbe2b6da90cde5bb58a54f2dc25be83a1d190e9afef0307b51b828e56bc7'
LLM_FRAME_SHA256='73d8452f657397d6ee528933fb56b1df10655b95a0522c1aeb0cb190efbb5369'
CLAIM_FRAME_SHA256='222981f3ad5791b0301bfe21a8f53cba0dda7211e567fba8edfd3d63379c9c80'

ZERO='0 0 0 0 0 0 0 0'

fail() {
  printf 'pireus operator morphogenesis: FAIL: %s\n' "$*" >&2
  exit 1
}

sha_file() { sha256sum "$1" | cut -d' ' -f1; }
sha_text() { printf '%s\n' "$1" | sha256sum | cut -d' ' -f1; }

require_hash() {
  local path="$1" expected="$2"
  [[ -f "${path}" ]] || fail "missing file: ${path}"
  [[ "$(sha_file "${path}")" == "${expected}" ]] || fail "hash drift: ${path}"
}

require_line() {
  local path="$1" expected="$2"
  grep -Fqx -- "${expected}" "${path}" || fail "missing exact line in ${path}: ${expected}"
}

require_committed_hash() {
  local commit="$1" path="$2" expected="$3"
  [[ "$(git -C "${ROOT}" show "${commit}:${path}" | sha256sum | cut -d' ' -f1)" == "${expected}" ]] ||
    fail "committed hash drift: ${commit}:${path}"
}

sha_limbs() {
  local hex="$1" out='' i part
  [[ "${#hex}" -eq 64 && "${hex}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid SHA-256: ${hex}"
  for ((i=0; i<8; i++)); do
    part="${hex:$((i*8)):8}"
    out="${out}${out:+ }$((16#${part}))"
  done
  printf '%s' "${out}"
}

authority_frame() {
  local stage="$1" action="$2" language="$3" role="$4" policy="$5"
  local semantic_write="$6" expected_write="$7" parity_valid="$8" review_promoted="$9"
  local parent_hash="${10}" toolchain_hash="${11}" command_hash="${12}" result_hash="${13}"
  local result_limbs="${ZERO}"
  [[ "${result_hash}" == zero ]] || result_limbs="$(sha_limbs "${result_hash}")"
  printf '9020 %s %s %s %s %s %s %s %s %s 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${stage}" "${action}" "${language}" "${role}" "${policy}" \
    "${semantic_write}" "${expected_write}" "${parity_valid}" "${review_promoted}" \
    "$(sha_limbs "${MODULE_SHA256}")" "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${parent_hash}")" "$(sha_limbs "${toolchain_hash}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" "$(sha_limbs "${command_hash}")" \
    "${result_limbs}" "${ZERO}"
}

check_guardian() {
  local label="$1" frame="$2" expected_sha="$3" expected_rc="$4" expected="$5"
  local decision rc
  [[ "$(sha_text "${frame}")" == "${expected_sha}" ]] || fail "Guardian frame drift: ${label}"
  set +e
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  rc=$?
  set -e
  [[ "${rc}" -eq "${expected_rc}" ]] || fail "Guardian exit drift for ${label}: ${rc}"
  [[ "${decision}" == "${expected}" ]] || fail "Guardian decision drift for ${label}: ${decision}"
  printf 'GUARDIAN_DECISION label=%s decision=%s dispatch_authorized=%s process_launched=false\n' \
    "${label}" "${decision}" "$([[ "${expected_rc}" -eq 0 ]] && printf true || printf false)"
}

run_sounio_authorized() {
  local label="$1" frame="$2" expected_sha="$3" source="$4" stdout="$5" stderr="$6"
  local status
  check_guardian "${label}" "${frame}" "${expected_sha}" 0 \
    'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'
  set +e
  (
    cd "${ROOT}"
    SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run "${source}" >"${stdout}" 2>"${stderr}"
  )
  status=$?
  set -e
  [[ "${status}" -eq 0 ]] || fail "${label} exited ${status}"
  [[ ! -s "${stderr}" ]] || fail "${label} emitted stderr"
  printf 'GUARDIAN_DISPATCH label=%s process_launched=true exit_code=%s stderr_empty=true\n' "${label}" "${status}"
}

freeze_frame() {
  local command_sha="$1" result_sha="$2"
  authority_frame 2 3 1 1 1 1 1 0 0 "${PARENT_SEMANTICS_SHA256}" \
    "${TOOLCHAIN_SHA256}" "${command_sha}" "${result_sha}"
}

negative_frame() {
  local stage="$1" language="$2" role="$3" policy="$4" semantic_write="$5"
  local expected_write="$6" review_promoted="$7" toolchain="$8" command="$9" action="${10}"
  authority_frame "${stage}" "${action}" "${language}" "${role}" "${policy}" \
    "${semantic_write}" "${expected_write}" 0 "${review_promoted}" \
    "${SEMANTICS_SHA256}" "${toolchain}" "${command}" zero
}

transcript_admitted() {
  local path="$1"
  [[ "$(sha_file "${path}")" == "${FROZEN_EVIDENCE_SHA256}" ]] &&
    [[ "$(wc -l < "${path}")" -eq 84308 ]] &&
    [[ "$(wc -c < "${path}")" -eq 1076155 ]] &&
    grep -Fqx 'SOUNIO_AUTHORITY schema=pireus-operator-morphogenesis.v12 role=SEMANTIC_AUTHORITY stage=SOUNIO_EXECUTABLE matcher_free=1' "${path}" &&
    grep -Fqx 'PIREUS_POM_ARCHIVE initial=96' "${path}" &&
    grep -Fqx ' final=128' "${path}" &&
    grep -Fqx ' generated=16' "${path}" &&
    grep -Fqx ' fixed=0' "${path}" &&
    grep -Fqx ' pairs=16' "${path}" &&
    grep -Fqx ' closure_checks=3680' "${path}" &&
    grep -Fqx ' closure_failures=0' "${path}" &&
    grep -Fqx ' dedup_checks=4560' "${path}" &&
    grep -Fqx ' phase_checks=1776' "${path}" &&
    grep -Fqx ' anf_coefficients=3600' "${path}" &&
    grep -Fqx ' certificates=3552' "${path}" &&
    grep -Fqx ' direct_comparisons=3552' "${path}" &&
    grep -Fqx ' unit_checks=496' "${path}" &&
    grep -Fqx ' microprogram_checks=4096' "${path}" &&
    [[ "$(grep -c '^PIREUS_POM_EPOCH epoch=' "${path}")" -eq 16 ]] &&
    [[ "$(grep -c '^PIREUS_POM_CERTIFICATE id=' "${path}")" -eq 3552 ]] &&
    [[ "$(grep -c '^PIREUS_POM_GENOME epoch=' "${path}")" -eq 3600 ]] &&
    [[ "$(grep -c '^PIREUS_POM_MICROPROGRAM epoch=' "${path}")" -eq 4096 ]] &&
    grep -Fqx ' constructive_bounded_relative_novelty=1' "${path}" &&
    grep -Fqx ' algebraic=0' "${path}" &&
    grep -Fqx ' algorithmic=0' "${path}" &&
    grep -Fqx ' material=0' "${path}" &&
    grep -Fqx ' scientific=0' "${path}" &&
    grep -Fqx ' historical=0' "${path}" &&
    grep -Fqx ' global=0' "${path}" &&
    grep -Fqx ' priority=0' "${path}" &&
    grep -Fqx ' claim_ready=0' "${path}" &&
    grep -Fqx ' frozen_mismatch_code=0' "${path}" &&
    grep -Fqx ' frozen_match=1' "${path}" &&
    grep -Fqx ' valid=1' "${path}"
}

require_hash "${ROOT}/${GARDEN_REL}" "${GARDEN_SHA256}"
require_hash "${ROOT}/${CONTRACT_REL}" "${CONTRACT_SHA256}"
require_hash "${ROOT}/${MODULE_REL}" "${MODULE_SHA256}"
require_hash "${ROOT}/${EXAMPLE_REL}" "${EXAMPLE_SHA256}"
require_hash "${ROOT}/${TEST_REL}" "${TEST_SHA256}"
require_hash "${ROOT}/${FIRST_RECEIPT_REL}" "${FIRST_RECEIPT_SHA256}"
require_hash "${ROOT}/${FREEZE_REL}" "${FREEZE_SHA256}"
require_hash "${ROOT}/${DECISIONS_REL}" "${DECISIONS_SHA256}"
require_hash "${ROOT}/${FIRST_EVIDENCE_REL}" "${FIRST_EVIDENCE_SHA256}"
require_hash "${ROOT}/${FIRST_TEST_EVIDENCE_REL}" "${FIRST_TEST_EVIDENCE_SHA256}"
require_hash "${ROOT}/${FROZEN_EVIDENCE_REL}" "${FROZEN_EVIDENCE_SHA256}"
require_hash "${ROOT}/${FROZEN_TEST_EVIDENCE_REL}" "${FROZEN_TEST_EVIDENCE_SHA256}"
require_hash "${ROOT}/bin/souc" "${WRAPPER_SHA256}"
require_hash "${ROOT}/scripts/lib/resolve_souc.sh" "${RESOLVER_SHA256}"
require_hash "${ROOT}/bin/souc-lean-single-x86_64" "${COMPILER_SHA256}"
require_hash "${GUARDIAN}" "${GUARDIAN_SHA256}"
require_hash "${ROOT}/stdlib/coordination/loom_language_authority.sio" "${GUARDIAN_POLICY_SHA256}"
require_hash "${ROOT}/tools/loom/language_authority.freeze.v1" "${GUARDIAN_FREEZE_SHA256}"
require_hash "${ROOT}/scripts/dev/build_sounio_loom_language_authority.sh" "${GUARDIAN_BUILD_SHA256}"
require_hash "${ROOT}/tools/loom/language_authority_main.sio" "${GUARDIAN_MAIN_SHA256}"
require_hash "${ROOT}/scripts/ci/sounio_loom_language_authority_selftest.sh" "${GUARDIAN_SELFTEST_SHA256}"

git -C "${ROOT}" merge-base --is-ancestor "${GARDEN_COMMIT}" "${EXECUTABLE_COMMIT}" || fail 'Garden does not precede executable'
git -C "${ROOT}" merge-base --is-ancestor "${EXECUTABLE_COMMIT}" "${FIRST_RESULT_COMMIT}" || fail 'Executable does not precede first result'
git -C "${ROOT}" merge-base --is-ancestor "${FIRST_RESULT_COMMIT}" "${MATCHER_COMMIT}" || fail 'First result does not precede matcher'
git -C "${ROOT}" merge-base --is-ancestor "${MATCHER_COMMIT}" HEAD || fail 'Matcher commit is not in current history'
[[ "$(git -C "${ROOT}" rev-parse "${GARDEN_COMMIT}")" == "${GARDEN_COMMIT}" ]] || fail 'Garden manifest commit drift'
[[ "$(git -C "${ROOT}" rev-parse "${EXECUTABLE_COMMIT}")" == "${EXECUTABLE_COMMIT}" ]] || fail 'Executable manifest commit drift'
[[ "$(git -C "${ROOT}" rev-parse "${FIRST_RESULT_COMMIT}")" == "${FIRST_RESULT_COMMIT}" ]] || fail 'First-result manifest commit drift'
[[ "$(git -C "${ROOT}" rev-parse "${MATCHER_COMMIT}")" == "${MATCHER_COMMIT}" ]] || fail 'Matcher manifest commit drift'
MATCHER_DELTA_PATHS="$(git -C "${ROOT}" diff --name-only "${FIRST_RESULT_COMMIT}" "${MATCHER_COMMIT}")"
[[ "${MATCHER_DELTA_PATHS}" == $'.claude/llm_offload_log.md\nexamples/pireus_operator_morphogenesis.sio\nstdlib/hardware/pireus/operator_morphogenesis.sio\ntests/stdlib/hardware/test_pireus_operator_morphogenesis.sio' ]] || fail 'Matcher delta escaped its pinned path allowlist'
require_committed_hash "${EXECUTABLE_COMMIT}" "${MODULE_REL}" "${FIRST_MODULE_SHA256}"
require_committed_hash "${EXECUTABLE_COMMIT}" "${EXAMPLE_REL}" "${FIRST_EXAMPLE_SHA256}"
require_committed_hash "${EXECUTABLE_COMMIT}" "${TEST_REL}" "${FIRST_TEST_SHA256}"
require_committed_hash "${FIRST_RESULT_COMMIT}" "${FIRST_RECEIPT_REL}" "${FIRST_RECEIPT_SHA256}"
require_committed_hash "${FIRST_RESULT_COMMIT}" "${FIRST_EVIDENCE_REL}" "${FIRST_EVIDENCE_SHA256}"
require_committed_hash "${FIRST_RESULT_COMMIT}" "${FIRST_TEST_EVIDENCE_REL}" "${FIRST_TEST_EVIDENCE_SHA256}"
require_committed_hash "${MATCHER_COMMIT}" "${MODULE_REL}" "${MODULE_SHA256}"
require_committed_hash "${MATCHER_COMMIT}" "${EXAMPLE_REL}" "${EXAMPLE_SHA256}"
require_committed_hash "${MATCHER_COMMIT}" "${TEST_REL}" "${TEST_SHA256}"

[[ "$(sed -n '/source_manifest_begin/,/source_manifest_end/p' "${ROOT}/${FREEZE_REL}" | sed '1d;$d' | sha256sum | cut -d' ' -f1)" == "${SOURCE_MANIFEST_SHA256}" ]] || fail 'source manifest digest drift'
[[ "$(sed -n '/semantics_material_begin/,/semantics_material_end/p' "${ROOT}/${FREEZE_REL}" | sed '1d;$d' | sha256sum | cut -d' ' -f1)" == "${SEMANTICS_SHA256}" ]] || fail 'semantics digest drift'
require_line "${ROOT}/${CONTRACT_REL}" 'Status: SEMANTICS_FROZEN'
require_line "${ROOT}/${FREEZE_REL}" 'stage=SEMANTICS_FROZEN'
require_line "${ROOT}/${FREEZE_REL}" 'parity_open=false'
require_line "${ROOT}/${FREEZE_REL}" 'claim_ready=false'
require_line "${ROOT}/${FREEZE_REL}" 'spark_material_route=KUBERNETES'
require_line "${ROOT}/${FREEZE_REL}" 'spark_k8s_nodes=spark-3c59:spark-8e54'
require_line "${ROOT}/${FREEZE_REL}" 'slurm_route_used=false'
require_line "${ROOT}/${FREEZE_REL}" 'python_process_launched=false'
require_line "${ROOT}/${FREEZE_REL}" 'rust_process_launched=false'
require_line "${ROOT}/${FREEZE_REL}" 'raw_elf_process_launched=false'
require_line "${ROOT}/${FREEZE_REL}" 'initial_archive_images=96'
require_line "${ROOT}/${FREEZE_REL}" 'final_archive_images=128'
require_line "${ROOT}/${FREEZE_REL}" 'generated_epochs=16'
require_line "${ROOT}/${FREEZE_REL}" 'orbit_pairs=16'
require_line "${ROOT}/${FREEZE_REL}" 'phase=225_BIT_INTERIOR_TRUTH_TABLE_RELATIVE_TO_PINNED_CD_SIGMA'
require_line "${ROOT}/${FREEZE_REL}" 'certificate_count=3552'
require_line "${ROOT}/${FREEZE_REL}" 'formal_parity_status=NOT_OPENED'
require_line "${ROOT}/${FREEZE_REL}" 'effect_parity_status=NOT_OPENED'
require_line "${ROOT}/${FREEZE_REL}" 'material_parity_status=NOT_OPENED'
require_line "${ROOT}/${FREEZE_REL}" 'explicit_bootstrap_fallback_used=true'
require_line "${ROOT}/${DECISIONS_REL}" 'decision_01_process_launched=true'
require_line "${ROOT}/${DECISIONS_REL}" 'decision_02_process_launched=true'
require_line "${ROOT}/${DECISIONS_REL}" 'decision_03_process_launched=true'
require_line "${ROOT}/${DECISIONS_REL}" 'decision_13_process_launched=true'
require_line "${ROOT}/${DECISIONS_REL}" 'decision_14_process_launched=true'
require_line "${ROOT}/${DECISIONS_REL}" 'decision_01=ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'
require_line "${ROOT}/${DECISIONS_REL}" 'decision_02=ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'
require_line "${ROOT}/${DECISIONS_REL}" 'decision_04=DENY code=112 reason=wrong-stage next_stage=SOUNIO_EXECUTABLE'
require_line "${ROOT}/${DECISIONS_REL}" 'decision_05=DENY code=101 reason=policy-missing next_stage=SEMANTICS_FROZEN'
require_line "${ROOT}/${DECISIONS_REL}" 'decision_06=DENY code=102 reason=policy-timeout next_stage=SEMANTICS_FROZEN'
require_line "${ROOT}/${DECISIONS_REL}" 'decision_07=DENY code=103 reason=policy-error next_stage=SEMANTICS_FROZEN'
require_line "${ROOT}/${DECISIONS_REL}" 'decision_08=DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
require_line "${ROOT}/${DECISIONS_REL}" 'decision_08_process_launched=false'
require_line "${ROOT}/${DECISIONS_REL}" 'decision_09=DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
require_line "${ROOT}/${DECISIONS_REL}" 'decision_09_process_launched=false'
require_line "${ROOT}/${DECISIONS_REL}" 'decision_10=DENY code=113 reason=semantic-authority-required next_stage=SEMANTICS_FROZEN'
require_line "${ROOT}/${DECISIONS_REL}" 'decision_11=DENY code=119 reason=review-promoted-to-authority next_stage=SEMANTICS_FROZEN'
require_line "${ROOT}/${DECISIONS_REL}" 'decision_12=DENY code=112 reason=wrong-stage next_stage=SEMANTICS_FROZEN'
transcript_admitted "${ROOT}/${FROZEN_EVIDENCE_REL}" || fail 'frozen transcript refused'

check_guardian PARITY_PREFREEZE "$(negative_frame 2 2 2 1 0 0 0 "${TOOLCHAIN_SHA256}" "${PARITY_COMMAND_SHA256}" 4)" \
  "${PARITY_PREFREEZE_FRAME_SHA256}" 112 'SOUNIO_LANGUAGE_AUTHORITY_DENY code=112 reason=wrong-stage next_stage=SOUNIO_EXECUTABLE'
check_guardian POLICY_MISSING "$(negative_frame 3 2 2 0 0 0 0 "${TOOLCHAIN_SHA256}" "${PARITY_COMMAND_SHA256}" 4)" \
  "${POLICY_MISSING_FRAME_SHA256}" 101 'SOUNIO_LANGUAGE_AUTHORITY_DENY code=101 reason=policy-missing next_stage=SEMANTICS_FROZEN'
check_guardian POLICY_TIMEOUT "$(negative_frame 3 2 2 2 0 0 0 "${TOOLCHAIN_SHA256}" "${PARITY_COMMAND_SHA256}" 4)" \
  "${POLICY_TIMEOUT_FRAME_SHA256}" 102 'SOUNIO_LANGUAGE_AUTHORITY_DENY code=102 reason=policy-timeout next_stage=SEMANTICS_FROZEN'
check_guardian POLICY_ERROR "$(negative_frame 3 2 2 3 0 0 0 "${TOOLCHAIN_SHA256}" "${PARITY_COMMAND_SHA256}" 4)" \
  "${POLICY_ERROR_FRAME_SHA256}" 103 'SOUNIO_LANGUAGE_AUTHORITY_DENY code=103 reason=policy-error next_stage=SEMANTICS_FROZEN'
check_guardian PYTHON_ORACLE "$(negative_frame 3 7 7 1 0 0 0 "${PYTHON_TOOLCHAIN_SHA256}" "${PYTHON_COMMAND_SHA256}" 4)" \
  "${PYTHON_FRAME_SHA256}" 110 'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
check_guardian RUST_ORACLE "$(negative_frame 3 8 7 1 0 0 0 "${RUST_TOOLCHAIN_SHA256}" "${RUST_COMMAND_SHA256}" 4)" \
  "${RUST_FRAME_SHA256}" 110 'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
check_guardian CPP_SEMANTIC_WRITE "$(authority_frame 3 4 4 4 1 1 1 1 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${CPP_COMMAND_SHA256}" zero)" \
  "${CPP_FRAME_SHA256}" 113 'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=SEMANTICS_FROZEN'
check_guardian LLM_PROMOTION "$(negative_frame 3 6 6 1 0 0 1 "${TOOLCHAIN_SHA256}" "${LLM_COMMAND_SHA256}" 5)" \
  "${LLM_FRAME_SHA256}" 119 'SOUNIO_LANGUAGE_AUTHORITY_DENY code=119 reason=review-promoted-to-authority next_stage=SEMANTICS_FROZEN'
check_guardian CLAIM_PREFREEZE "$(negative_frame 3 1 1 1 0 0 0 "${TOOLCHAIN_SHA256}" "${CLAIM_COMMAND_SHA256}" 7)" \
  "${CLAIM_FRAME_SHA256}" 112 'SOUNIO_LANGUAGE_AUTHORITY_DENY code=112 reason=wrong-stage next_stage=SEMANTICS_FROZEN'

TMPDIR_PIREUS="$(mktemp -d)"
trap 'rm -rf "${TMPDIR_PIREUS}"' EXIT
run_sounio_authorized MAIN_FREEZE \
  "$(freeze_frame "${COMMAND_SHA256}" "${FROZEN_EVIDENCE_SHA256}")" \
  "${MAIN_FREEZE_FRAME_SHA256}" "${EXAMPLE_REL}" \
  "${TMPDIR_PIREUS}/main.txt" "${TMPDIR_PIREUS}/main.stderr"
run_sounio_authorized TEST_FREEZE \
  "$(freeze_frame "${TEST_COMMAND_SHA256}" "${FROZEN_TEST_EVIDENCE_SHA256}")" \
  "${TEST_FREEZE_FRAME_SHA256}" "${TEST_REL}" \
  "${TMPDIR_PIREUS}/test.txt" "${TMPDIR_PIREUS}/test.stderr"
cmp -s "${TMPDIR_PIREUS}/main.txt" "${ROOT}/${FROZEN_EVIDENCE_REL}" || fail 'main replay differs from frozen evidence'
cmp -s "${TMPDIR_PIREUS}/test.txt" "${ROOT}/${FROZEN_TEST_EVIDENCE_REL}" || fail 'test replay differs from frozen evidence'

printf '%s\n' \
  'PIREUS_OPERATOR_MORPHOGENESIS_GATE_PASS=true stage=SEMANTICS_FROZEN language=Sounio role=SEMANTIC_AUTHORITY initial_archive=96 final_archive=128 epochs=16 c2_pairs=16 interior_cells=225 certificates=3552 sounio_executable_certificate_generation_complete=true sounio_in_run_archive_noncollision_complete=true formal_parity_open=false effect_parity_open=false material_parity_open=false spark_route_policy=KUBERNETES spark_nodes=spark-3c59:spark-8e54 slurm_route_used=false python_dispatch=E110 python_process_launched=false rust_dispatch=E110 rust_process_launched=false cpp_semantic_write=E113 llm_promotion=E119 policy_missing=E101 policy_timeout=E102 policy_error=E103 claim_ready=false engine=lean_single explicit_bootstrap_fallback=true raw_elf_process_launched=false'
