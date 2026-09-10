#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN_CANDIDATE="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
GUARDIAN="$(readlink -f "${GUARDIAN_CANDIDATE}")"

GARDEN_REL='tools/pireus/GARDEN_PIREUS_OPERATOR_ORBIT_CANONICALIZATION_V13.md'
CONTRACT_REL='tools/pireus/PIREUS_OPERATOR_ORBIT_CANONICALIZATION_CONTRACT_V13.md'
MODULE_REL='stdlib/hardware/pireus/operator_orbit_canonicalization.sio'
EXAMPLE_REL='examples/pireus_operator_orbit_canonicalization.sio'
TEST_REL='tests/stdlib/hardware/test_pireus_operator_orbit_canonicalization.sio'
FIRST_TRANSCRIPT_REL='tools/pireus/operator_orbit_canonicalization.first.v13'
FIRST_RECEIPT_REL='tools/pireus/evidence/operator_orbit_canonicalization_v13.first.txt'
FREEZE_REL='tools/pireus/operator_orbit_canonicalization.freeze.v13'
FROZEN_EVIDENCE_REL='tools/pireus/evidence/operator_orbit_canonicalization_v13.frozen.txt'
TEST_EVIDENCE_REL='tools/pireus/evidence/operator_orbit_canonicalization_v13.test.txt'

GARDEN_COMMIT='86755b3027a3c5d0b7d5961e4012cab95d4c8c31'
EXECUTABLE_COMMIT='73704f7afed6780c3a317b739cbd35fe94dbe395'
FIRST_EVIDENCE_COMMIT='22fbabe81cf365c0b542d8a425ec4c081f31e390'
MATCHER_COMMIT='00200c2aa5a021cdc8d91de2d231f3e573d372bb'
FREEZE_COMMIT='51d92c3c7868c1a838a2dab8c3cdddb4d5ad1313'

GARDEN_SHA256='1020135c8a65151b1e51b3ec142844c551369dbbcefaf345340428254da8c862'
PREFREEZE_CONTRACT_SHA256='de23274bd1b1a88c37e539d8d39bbeceb2fec1cd95ccc8e47abb5f78aff79b3e'
STATUS_CONTRACT_SHA256='02f7626b3cc6bcb4a54286c895786b6d105c06b045543c9c4e2f6915ba87c5b3'
FIRST_MODULE_SHA256='3136968a83bbba18d56c543895d6bbd9530ccf6c59db78ac6b6f2fa3bd26c9e4'
FIRST_EXAMPLE_SHA256='c6fb970ecf4c0dd9742bf9e561854f82f48c435cf75a192ded90996db3016202'
FIRST_TEST_SHA256='396d80445738496b568a99e92123031a51126be3b8c155d89f4e78996c99919d'
MODULE_SHA256='7ada1b17bf91fdb3f4c48877d2485f71a65bb4159d88cb7e4b288c77bfe3cdae'
EXAMPLE_SHA256='74cb0c97f06e83422c1d7584697f762dfdebc7060e391aca292c43fbe68a67fc'
TEST_SHA256='fcb83b48dadd1dd85cfaf29abc75f50608ac208d7f29960232e05fbe36440d14'
FIRST_TRANSCRIPT_SHA256='16af63f5e0f8aa7e5c899f4c395404b83fb402f6bbdb5f20dea2a3d10ad2e19f'
FIRST_RECEIPT_SHA256='be0ef127e7c40cf0167cb55189c39245dbfd93ffed990d64a003cadf3f19f38b'
FREEZE_COMMIT_SHA256='ab76f07423df320fd8f65738b9dc6516b60e0fd9a231d94e5b2a3b65efacf50d'
FREEZE_SHA256='11893a34450729ff06ac40ade86c90decb7a6947daea3cc108cae17f73572f84'
FROZEN_EVIDENCE_SHA256='57fb629ec4b4610171d77daefa88959214035439458edfbbfee0eb72c4d98686'
TEST_EVIDENCE_SHA256='49637ca4266d16c9bb1476c010d5c6c902b800ea6c0c22d9f5bdebdd49ace7b6'
SOURCE_MANIFEST_SHA256='022fda14573d31009c3740f0cb374b8ac06b1047fa0a90ae9ac5f44074c3e44d'
SEMANTICS_SHA256='0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c'
PARENT_SEMANTICS_SHA256='999c6e7a0051f702cf40bb2adab7dc91c4f026230830096377f525005067c2f4'

WRAPPER_SHA256='ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008'
RESOLVER_SHA256='a7e37545490745a58731933b0e07db69843cfeea30e739ed554b82d099ec3d84'
COMPILER_SHA256='6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2'
TOOLCHAIN_SHA256='2ce5194cdc517de8d7a0063e09d4f4e7b6b5701a23fa1031a3ec9e8f56486b6e'
HARDWARE_SHA256='0f0526af4e296f77355cdb80e70843aceb2ebdbad7c50d43e5479fcee693b401'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
GUARDIAN_POLICY_SHA256='64bb0118793fe46dcb392abc1a9212eb15bd55047461576a3ef1a6cefa3f17da'
GUARDIAN_FREEZE_SHA256='5fe5e5c9cdcb83935770f58df52f2d614d11f8abde519c4a2505ca20998fae2e'
GUARDIAN_BUILD_SHA256='af7c1098143d0aad108684646df4c72fecca03404557f5494206713486ca09b6'
GUARDIAN_MAIN_SHA256='99b6fe7e1c687c3a4e76cfe1585e4826e753f473dff8676dd287eb2f9e0021bc'
GUARDIAN_SELFTEST_SHA256='c9c7f839fb262dbf616716e3c5f0601bb03cfbcbbcfe2fbe09bd0b39894e2a9f'

PYTHON_TOOLCHAIN_SHA256='422c73f42aff0794a916a210455ea8a5d2bbbd29430963b94c29966017abc517'
PYTHON_COMMAND_SHA256='6275cbe734054d3a8e754709d3a82c664ecedf23c3ba306339e2d3807fca6353'
RUST_TOOLCHAIN_SHA256='927b200c4865a65aab686af6fa2fa90a7bd4c49453e486114ee7aa88e2495ad5'
RUST_COMMAND_SHA256='b8349b56743edbe19cbe0abebc5655259438910d32303a2c04b948a091ed026a'
PARITY_COMMAND_SHA256='89033cbad572b0a6803712cea458abb6deb86faa6fc1040b8fee3339e1076749'
CPP_COMMAND_SHA256='2fa0d998a7ea477ec29cba8b503aeb034172e79245fd7feb2ff9490651d5a20f'
LLM_COMMAND_SHA256='5077845a76ea5f3448a8ae51b723282b34a8facf5aeca3023f8b5175b7b03e9a'
CLAIM_COMMAND_SHA256='2afa719fda98b7a6e50096401713347fa2d769cb4f3afffff04f491b8444d907'
CI_COMMAND_SHA256='7862ad63b2b8f52f3da637c19b499887bffaca11fe5749684c1dc7786bdafc86'
WRITE_COMMAND_SHA256='ec24f57f6a61aa065e679b79adf91e2cd91535b81859ee83e19cd12744769f68'
EDIT_COMMAND_SHA256='872402b36864df4b7d5841c28ad1420981cc0a976732ec80fef265a48acc7cc1'
COMMIT_COMMAND_SHA256='b552e48d204f0c05d2517481bea4d68da85821a110e3e4103b939d946a249c49'

PARITY_PREFREEZE_FRAME_SHA256='42348bb7ee9bf59ccdf810e8228e0924cd6e64b70d5cd71e352c781b4309d120'
POLICY_MISSING_FRAME_SHA256='d711fc78e551b1b9bfd9e32846fb6900a268a4138eb1074912a85af35994b0dc'
POLICY_TIMEOUT_FRAME_SHA256='dcdee0a3f1546d8f2ee89cde102a844e36700a03000e9201137092adb88c5009'
POLICY_ERROR_FRAME_SHA256='e520af99b7f53d4eea7638a6006bdf365aeec8c8e9b612b249ce2fc08a861b4b'
PYTHON_FRAME_SHA256='6d1566dbac54c363c688b1c2c242909538a422bdfb82301f60a391054ad1bc0d'
RUST_FRAME_SHA256='812a2bfd97cc9cd4d59e0b9deb0761e9112fe2a092832f1b6bbb4cc9a331c44a'
CPP_FRAME_SHA256='a06ffb229a81d7cbfaaf05cf8b4c611b8970a0cbd3a70fd10e0952665f6df158'
LLM_FRAME_SHA256='aef25a44912ef116628dc09d5fa86692bcb278794d3a9f845d89d49383f7c762'
CLAIM_FRAME_SHA256='5591f204a7441a8dee193a5d06ecec9ae6ab4ba69e4fcd6c7c90ddb1576e585f'
WRITE_FRAME_SHA256='562dffc13b31303ff05703d08cf8944aa8483e923e977649246b7e89ed595716'
EDIT_FRAME_SHA256='f5b834ccbca53fa155dfd8fac606e97282595044e7e985b42bb2483a719b72fa'
COMMIT_FRAME_SHA256='22f65a4c386156d65a2839f37fd0a6e2c570df704a4e0e0a6dfb7c8a36e9c274'
CI_FRAME_SHA256='7363d9bd1b34c464b87fb17c7d80250cd70c09c216355af5e533887927eb56de'

ZERO='0 0 0 0 0 0 0 0'

fail() {
  printf 'pireus operator orbit canonicalization: FAIL: %s\n' "$*" >&2
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
    "$(sha_limbs "${SOURCE_MANIFEST_SHA256}")" "$(sha_limbs "${SEMANTICS_SHA256}")" \
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
  printf 'GUARDIAN_DECISION label=%s decision=%s process_launched=false\n' "${label}" "${decision}"
}

frozen_transcript_admitted() {
  local path="$1"
  require_line "${path}" 'SOUNIO_AUTHORITY schema=pireus-operator-orbit-canonicalization.v13 role=SEMANTIC_AUTHORITY stage=SOUNIO_EXECUTABLE matcher_free=0 expected_results=1'
  require_line "${path}" 'PIREUS_POC_RESULT valid=1'
  require_line "${path}" ' baseline_images=128'
  require_line "${path}" ' baseline_classes=30'
  require_line "${path}" ' attempts=33'
  require_line "${path}" ' collapses=1'
  require_line "${path}" ' admitted=32'
  require_line "${path}" ' final_classes=62'
  require_line "${path}" 'PIREUS_POC_BOUNDARY exact_declared_orbit=1'
  require_line "${path}" ' bounded_relative_novelty=1'
  require_line "${path}" ' full_space=0'
  require_line "${path}" ' nonlinear_permutation=0'
  require_line "${path}" ' isotopy=0'
  require_line "${path}" ' unrestricted_isomorphism=0'
  require_line "${path}" ' algebraic=0'
  require_line "${path}" ' algorithmic=0'
  require_line "${path}" ' material=0'
  require_line "${path}" ' performance=0'
  require_line "${path}" ' scientific=0'
  require_line "${path}" ' global=0'
  require_line "${path}" ' historical=0'
  require_line "${path}" ' priority=0'
  require_line "${path}" ' claim_ready=0'
  require_line "${path}" ' frozen_match=1'
  require_line "${path}" ' frozen_mismatch_code=0'
  [[ "$(grep -c '^PIREUS_POC_BASELINE image=' "${path}")" -eq 128 ]] || fail 'baseline record count drift'
  [[ "$(grep -c '^PIREUS_POC_ATTEMPT attempt=' "${path}")" -eq 33 ]] || fail 'attempt record count drift'
  [[ "$(grep -c '^PIREUS_POC_ADMITTED admitted=' "${path}")" -eq 32 ]] || fail 'admission record count drift'
  [[ "$(grep -c '^PIREUS_POC_SEPARATOR admitted=' "${path}")" -eq 1456 ]] || fail 'separator record count drift'
  [[ "$(grep -c '^PIREUS_POC_RAW_MICROPROGRAM admitted=' "${path}")" -eq 32 ]] || fail 'raw microprogram record count drift'
  [[ "$(grep -c '^PIREUS_POC_CANON_MICROPROGRAM admitted=' "${path}")" -eq 32 ]] || fail 'canonical microprogram record count drift'
}

require_hash "${ROOT}/${GARDEN_REL}" "${GARDEN_SHA256}"
require_hash "${ROOT}/${CONTRACT_REL}" "${STATUS_CONTRACT_SHA256}"
require_hash "${ROOT}/${MODULE_REL}" "${MODULE_SHA256}"
require_hash "${ROOT}/${EXAMPLE_REL}" "${EXAMPLE_SHA256}"
require_hash "${ROOT}/${TEST_REL}" "${TEST_SHA256}"
require_hash "${ROOT}/${FIRST_TRANSCRIPT_REL}" "${FIRST_TRANSCRIPT_SHA256}"
require_hash "${ROOT}/${FIRST_RECEIPT_REL}" "${FIRST_RECEIPT_SHA256}"
require_hash "${ROOT}/${FREEZE_REL}" "${FREEZE_SHA256}"
require_hash "${ROOT}/${FROZEN_EVIDENCE_REL}" "${FROZEN_EVIDENCE_SHA256}"
require_hash "${ROOT}/${TEST_EVIDENCE_REL}" "${TEST_EVIDENCE_SHA256}"
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
git -C "${ROOT}" merge-base --is-ancestor "${EXECUTABLE_COMMIT}" "${FIRST_EVIDENCE_COMMIT}" || fail 'Executable does not precede first evidence'
git -C "${ROOT}" merge-base --is-ancestor "${FIRST_EVIDENCE_COMMIT}" "${MATCHER_COMMIT}" || fail 'First evidence does not precede matcher'
git -C "${ROOT}" merge-base --is-ancestor "${MATCHER_COMMIT}" "${FREEZE_COMMIT}" || fail 'Matcher does not precede freeze'
git -C "${ROOT}" merge-base --is-ancestor "${FREEZE_COMMIT}" HEAD || fail 'Freeze commit is not in current history'
require_committed_hash "${GARDEN_COMMIT}" "${GARDEN_REL}" "${GARDEN_SHA256}"
require_committed_hash "${GARDEN_COMMIT}" "${CONTRACT_REL}" "${PREFREEZE_CONTRACT_SHA256}"
require_committed_hash "${EXECUTABLE_COMMIT}" "${MODULE_REL}" "${FIRST_MODULE_SHA256}"
require_committed_hash "${EXECUTABLE_COMMIT}" "${EXAMPLE_REL}" "${FIRST_EXAMPLE_SHA256}"
require_committed_hash "${EXECUTABLE_COMMIT}" "${TEST_REL}" "${FIRST_TEST_SHA256}"
require_committed_hash "${FIRST_EVIDENCE_COMMIT}" "${FIRST_TRANSCRIPT_REL}" "${FIRST_TRANSCRIPT_SHA256}"
require_committed_hash "${FIRST_EVIDENCE_COMMIT}" "${FIRST_RECEIPT_REL}" "${FIRST_RECEIPT_SHA256}"
require_committed_hash "${MATCHER_COMMIT}" "${MODULE_REL}" "${MODULE_SHA256}"
require_committed_hash "${MATCHER_COMMIT}" "${EXAMPLE_REL}" "${EXAMPLE_SHA256}"
require_committed_hash "${MATCHER_COMMIT}" "${TEST_REL}" "${TEST_SHA256}"
require_committed_hash "${FREEZE_COMMIT}" "${FREEZE_REL}" "${FREEZE_COMMIT_SHA256}"
require_committed_hash "${FREEZE_COMMIT}" "${FROZEN_EVIDENCE_REL}" "${FROZEN_EVIDENCE_SHA256}"
require_committed_hash "${FREEZE_COMMIT}" "${TEST_EVIDENCE_REL}" "${TEST_EVIDENCE_SHA256}"

if git -C "${ROOT}" grep -q 'pireus_operator_orbit_canonicalization_matches_frozen_semantics' "${EXECUTABLE_COMMIT}" -- "${MODULE_REL}" "${EXAMPLE_REL}" "${TEST_REL}"; then
  fail 'matcher leaked into matcher-free executable commit'
fi
MATCHER_DELTA_PATHS="$(git -C "${ROOT}" diff --name-only "${FIRST_EVIDENCE_COMMIT}" "${MATCHER_COMMIT}")"
[[ "${MATCHER_DELTA_PATHS}" == $'.claude/llm_offload_log.md\nexamples/pireus_operator_orbit_canonicalization.sio\nstdlib/hardware/pireus/operator_orbit_canonicalization.sio\ntests/stdlib/hardware/test_pireus_operator_orbit_canonicalization.sio' ]] || fail 'matcher delta escaped its exact allowlist'

[[ "$(sed -n '/^source_manifest_begin$/,/^source_manifest_end$/p' "${ROOT}/${FREEZE_REL}" | sed '1d;$d' | sha256sum | cut -d' ' -f1)" == "${SOURCE_MANIFEST_SHA256}" ]] || fail 'source manifest digest drift'
[[ "$(sed -n '/^semantics_material_begin$/,/^semantics_material_end$/p' "${ROOT}/${FREEZE_REL}" | sed '1d;$d' | sha256sum | cut -d' ' -f1)" == "${SEMANTICS_SHA256}" ]] || fail 'semantics digest drift'
require_line "${ROOT}/${CONTRACT_REL}" 'Status: SEMANTICS_FROZEN'
require_line "${ROOT}/${FREEZE_REL}" 'status=SEMANTICS_FROZEN'
require_line "${ROOT}/${FREEZE_REL}" 'stage=SEMANTICS_FROZEN'
require_line "${ROOT}/${FREEZE_REL}" 'semantics_frozen=true'
require_line "${ROOT}/${FREEZE_REL}" 'formal_parity_status=NOT_OPENED'
require_line "${ROOT}/${FREEZE_REL}" 'effect_parity_status=NOT_OPENED'
require_line "${ROOT}/${FREEZE_REL}" 'material_parity_status=NOT_OPENED'
require_line "${ROOT}/${FREEZE_REL}" 'claim_ready=false'
require_line "${ROOT}/${FREEZE_REL}" 'u250_cards_declared=2'
require_line "${ROOT}/${FREEZE_REL}" 'u250_cards_installed=1'
require_line "${ROOT}/${FREEZE_REL}" 'u250_cards_pending_installation=1'
require_line "${ROOT}/${FREEZE_REL}" 'u250_second_card_enumeration_failure=false'
require_line "${ROOT}/${FREEZE_REL}" 'spark_material_route=KUBERNETES_ONLY'
require_line "${ROOT}/${FREEZE_REL}" 'spark_nodes_used_for_semantic_freeze=false'
require_line "${ROOT}/${FREEZE_REL}" 'spark_slurm_route_used=false'
require_line "${ROOT}/${FREEZE_REL}" 'material_target_processes_launched=0'
require_line "${ROOT}/${FREEZE_REL}" 'python_process_launched=false'
require_line "${ROOT}/${FREEZE_REL}" 'rust_process_launched=false'
require_line "${ROOT}/${FREEZE_REL}" 'llm_offload_role=REVIEW_ONLY'
require_line "${ROOT}/${FREEZE_REL}" 'llm_confirmed_result=false'
require_line "${ROOT}/${FREEZE_REL}" 'frozen_replay_job_state=COMPLETED'
require_line "${ROOT}/${FREEZE_REL}" 'frozen_replay_job_exit_code=0:0'
require_line "${ROOT}/${FREEZE_REL}" 'separator_derivation=32_TIMES_30_PLUS_32_TIMES_31_DIV_2'
require_line "${ROOT}/${TEST_EVIDENCE_REL}" 'PIREUS_OPERATOR_ORBIT_CANONICALIZATION_EXECUTABLE_OK'
frozen_transcript_admitted "${ROOT}/${FROZEN_EVIDENCE_REL}"
[[ "$((32 * 30 + 32 * 31 / 2))" -eq 1456 ]] || fail 'separator closed-form derivation drift'

check_guardian PARITY_PREFREEZE "$(authority_frame 2 4 2 2 1 0 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${PARITY_COMMAND_SHA256}" zero)" \
  "${PARITY_PREFREEZE_FRAME_SHA256}" 112 'SOUNIO_LANGUAGE_AUTHORITY_DENY code=112 reason=wrong-stage next_stage=SOUNIO_EXECUTABLE'
check_guardian POLICY_MISSING "$(authority_frame 3 11 1 1 0 0 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${CI_COMMAND_SHA256}" zero)" \
  "${POLICY_MISSING_FRAME_SHA256}" 101 'SOUNIO_LANGUAGE_AUTHORITY_DENY code=101 reason=policy-missing next_stage=SEMANTICS_FROZEN'
check_guardian POLICY_TIMEOUT "$(authority_frame 3 11 1 1 2 0 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${CI_COMMAND_SHA256}" zero)" \
  "${POLICY_TIMEOUT_FRAME_SHA256}" 102 'SOUNIO_LANGUAGE_AUTHORITY_DENY code=102 reason=policy-timeout next_stage=SEMANTICS_FROZEN'
check_guardian POLICY_ERROR "$(authority_frame 3 11 1 1 3 0 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${CI_COMMAND_SHA256}" zero)" \
  "${POLICY_ERROR_FRAME_SHA256}" 103 'SOUNIO_LANGUAGE_AUTHORITY_DENY code=103 reason=policy-error next_stage=SEMANTICS_FROZEN'
check_guardian PYTHON_ORACLE "$(authority_frame 3 2 7 7 1 1 1 0 0 "${SEMANTICS_SHA256}" "${PYTHON_TOOLCHAIN_SHA256}" "${PYTHON_COMMAND_SHA256}" zero)" \
  "${PYTHON_FRAME_SHA256}" 110 'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
check_guardian RUST_ORACLE "$(authority_frame 3 2 8 7 1 1 1 0 0 "${SEMANTICS_SHA256}" "${RUST_TOOLCHAIN_SHA256}" "${RUST_COMMAND_SHA256}" zero)" \
  "${RUST_FRAME_SHA256}" 110 'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
check_guardian CPP_SEMANTIC_WRITE "$(authority_frame 3 8 4 4 1 1 1 1 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${CPP_COMMAND_SHA256}" "${FROZEN_EVIDENCE_SHA256}")" \
  "${CPP_FRAME_SHA256}" 113 'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=SEMANTICS_FROZEN'
check_guardian LLM_PROMOTION "$(authority_frame 3 5 6 6 1 0 0 0 1 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${LLM_COMMAND_SHA256}" zero)" \
  "${LLM_FRAME_SHA256}" 119 'SOUNIO_LANGUAGE_AUTHORITY_DENY code=119 reason=review-promoted-to-authority next_stage=SEMANTICS_FROZEN'
check_guardian CLAIM_PREFREEZE "$(authority_frame 3 7 1 1 1 0 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${CLAIM_COMMAND_SHA256}" zero)" \
  "${CLAIM_FRAME_SHA256}" 112 'SOUNIO_LANGUAGE_AUTHORITY_DENY code=112 reason=wrong-stage next_stage=SEMANTICS_FROZEN'

check_guardian WRITE "$(authority_frame 3 8 1 1 1 1 1 0 0 "${PARENT_SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${WRITE_COMMAND_SHA256}" "${FROZEN_EVIDENCE_SHA256}")" \
  "${WRITE_FRAME_SHA256}" 0 'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'
check_guardian EDIT_APPLY_PATCH "$(authority_frame 3 9 1 1 1 1 1 0 0 "${PARENT_SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${EDIT_COMMAND_SHA256}" "${FROZEN_EVIDENCE_SHA256}")" \
  "${EDIT_FRAME_SHA256}" 0 'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'
check_guardian COMMIT "$(authority_frame 3 10 1 1 1 1 1 0 0 "${PARENT_SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${COMMIT_COMMAND_SHA256}" "${FROZEN_EVIDENCE_SHA256}")" \
  "${COMMIT_FRAME_SHA256}" 0 'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'
check_guardian CI "$(authority_frame 3 11 1 1 1 0 0 0 0 "${PARENT_SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${CI_COMMAND_SHA256}" "${FROZEN_EVIDENCE_SHA256}")" \
  "${CI_FRAME_SHA256}" 0 'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'

printf '%s\n' \
  'PIREUS_OPERATOR_ORBIT_CANONICALIZATION_GATE_PASS=true stage=SEMANTICS_FROZEN language=Sounio role=SEMANTIC_AUTHORITY baseline_classes=30 admitted_classes=32 final_classes=62 separator_certificates=1456 separator_derivation=32_TIMES_30_PLUS_32_TIMES_31_DIV_2 exact_declared_orbit=true orbit_scope=GL4_F2_X_SWAP_X_2048_BASIS_FIXED_GAUGES bounded_relative_novelty=true broader_novelty=false formal_parity_open=false effect_parity_open=false material_parity_open=false python_dispatch=E110 python_process_launched=false rust_dispatch=E110 rust_process_launched=false cpp_semantic_write=E113 llm_promotion=E119 policy_missing=E101 policy_timeout=E102 policy_error=E103 write_fixture=ALLOW edit_apply_patch_fixture=ALLOW commit_fixture=ALLOW ci_fixture=ALLOW spark_route=KUBERNETES_ONLY spark_nodes_used=false slurm_cpu_validation=true u250_declared=2 u250_installed=1 u250_pending_installation=1 claim_ready=false'
