#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"

GARDEN_REL='tools/pireus/GARDEN_PIREUS_QUOTIENT_NOVELTY_FORGE_V5.md'
CONTRACT_REL='tools/pireus/PIREUS_QUOTIENT_NOVELTY_FORGE_CONTRACT_V5.md'
MODULE_REL='stdlib/hardware/pireus/quotient_novelty_forge.sio'
EXAMPLE_REL='examples/pireus_quotient_novelty_forge.sio'
TEST_REL='tests/stdlib/hardware/test_pireus_quotient_novelty_forge.sio'
FAILED_RECEIPT_REL='tools/pireus/quotient_novelty_forge.first-failure.v5'
FIRST_RECEIPT_REL='tools/pireus/quotient_novelty_forge.first-attempt2.v5'
FIRST_DECISIONS_REL='tools/pireus/quotient_novelty_forge.guardian-decisions.attempt2.v5'
FREEZE_REL='tools/pireus/quotient_novelty_forge.freeze.v5'
PARITY_REL='tools/pireus/quotient_novelty_forge.parity-open.v5'
DECISIONS_REL='tools/pireus/quotient_novelty_forge.guardian-decisions.v5'
FAILED_EVIDENCE_REL='tools/pireus/evidence/quotient_novelty_forge_v5.txt'
FIRST_EVIDENCE_REL='tools/pireus/evidence/quotient_novelty_forge_v5.attempt2.txt'
FROZEN_EVIDENCE_REL='tools/pireus/evidence/quotient_novelty_forge_v5.frozen.txt'
TEST_EVIDENCE_REL='tools/pireus/evidence/quotient_novelty_forge_v5.test.txt'

GARDEN_COMMIT='4de9a4679f83b906497d2b9dea87d1d2ec7986b2'
EXECUTABLE_COMMIT='5b0ace38927ffcf29884e3c0b2aed303d7fbc62e'
FAILED_ATTEMPT_COMMIT='d69835f65f7a6b4214703c5b12724463d3375410'
COMPACT_REPAIR_COMMIT='c118ebf998a23a55430d2e611e0745704c953566'
FIRST_EVIDENCE_COMMIT='5768d08797b9592e1b081ff2529ed4995b30ec9b'
SEMANTIC_FREEZE_COMMIT='cbd64ff6ed16d48bf88d7654aa6e9b592a577b2c'
CONTRACT_COMMIT='fdbc328ac29636ac58377b3601e2a3788f69620a'
FREEZE_RECEIPT_COMMIT='067dface7daf7d8642e775fe26f49ce568f77536'
PARITY_RECEIPT_COMMIT='aed0d3d7feb77dee1761531f24d552b7265b3b0a'

GARDEN_SHA256='601d665087cf136c47e1f6e4082fc757513fe10162b376d237330f77996e5ca0'
CONTRACT_SHA256='6ded3328c2888674c4a793d55e6760d63ee97aa1faae5ed845044c4f4f886b40'
FIRST_SOURCE_SHA256='799565db9a23ad99e300226ba500f06e7cd801a1acecd9ce63f21e5e8da936c4'
REPAIRED_SOURCE_SHA256='0a2490e0d21f8b9c4004cd6d6fe1caf5c03a2d239bda8d029a427327c42bd0ac'
MODULE_SHA256='791d85d4b336d854c6ed3b2e662e8f09b05f8a6f6d1dc4c03807c87150751667'
EXAMPLE_SHA256='676eb027934c4671ccb2890fff6c72ba48974b3f0ecebcf4131aa4307c4c3b6d'
TEST_SHA256='daaf11c7b4ebc7232ea32e0d4a750e843940697619f4355f397d6dc038072222'
FAILED_RECEIPT_SHA256='719e0b1aa0beab615d0a7e0f996f4a33f398e11d4173f928feeaacd033514111'
FIRST_RECEIPT_SHA256='7f55ceaa0669315636ee1e4b7f55b5bffec65a5cdfcb748af8996848fdb5987f'
FIRST_DECISIONS_SHA256='3c4f1a32bf29160c87000ab27471800af2d7bd65b3cbf02e7125ec4f35b88add'
FREEZE_SHA256='640a271bbe1966a3993e72be8fe019b1152530372cfb3ab91ede92011c0fc8c7'
PARITY_SHA256='108ac3dd8df394e01a5a3293aab8d9fe312d522245ed2ee02e8bc5db37fa2943'
DECISIONS_SHA256='03cda273575f35680e4448287b76f0187b242545f33e2bf268bcad8d7d9f1fb2'
FAILED_EVIDENCE_SHA256='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
FIRST_EVIDENCE_SHA256='7dfc4fa51b689d9295cf37252ffac0b54a108086a6c88dd9ba8c374769ac644f'
FROZEN_EVIDENCE_SHA256='1ec072889d2fefe69dcb842fa3bb894e4462f615bfbc6e1011e56e2b19bebae3'
TEST_EVIDENCE_SHA256='807f8c53046a16bc1239e27a6c6c9553148858fb73fbf26068d52093cd1fd929'
SOURCE_MANIFEST_SHA256='f56dbd8e143c277d92e6f61da68ad4ed71abb561a8469c6a18d0f7d48b33e652'
SEMANTICS_SHA256='9dde3079369c2c90f66805c18abf93a302f5a1d5facf909e39292283ed65bb21'
PARENT_SEMANTICS_SHA256='e8268af20770dbf292fb39f92793b7b89d1651b2e88193e0cb6ee765dfc1f1ff'
WRAPPER_SHA256='ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008'
RESOLVER_SHA256='a7e37545490745a58731933b0e07db69843cfeea30e739ed554b82d099ec3d84'
COMPILER_SHA256='6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
GUARDIAN_BUILD_SHA256='af7c1098143d0aad108684646df4c72fecca03404557f5494206713486ca09b6'
GUARDIAN_POLICY_SHA256='64bb0118793fe46dcb392abc1a9212eb15bd55047461576a3ef1a6cefa3f17da'
GUARDIAN_MAIN_SHA256='99b6fe7e1c687c3a4e76cfe1585e4826e753f473dff8676dd287eb2f9e0021bc'
GUARDIAN_SELFTEST_SHA256='c9c7f839fb262dbf616716e3c5f0601bb03cfbcbbcfe2fbe09bd0b39894e2a9f'
PARENT_GATE_SHA256='6d9fe145a2bdde85400e5861f6e3b6a99f9da2f9de21f28ac7f97b52ffa52f66'
TOOLCHAIN_SHA256='a4d9e4290d4373baa095cdcdd2f6582587323fa392166815ec6676c6178c3590'
HARDWARE_SHA256='ba649e3df628e482654987fc29b86489bfdfc9f69f6bfdc141f3a8d238ee33cd'
COMMAND_SHA256='3389f7daff8acf2d8f38a0e0fcc3efd67439c660822925ab36193130d3fb8828'
TEST_COMMAND_SHA256='2d67dd1932f3ff1c905ae929466f2f9267a05f5c2adb9a0618b920df3cc77a5b'
RESULT_CONTRACT_SHA256='8f2716cc7efdde209291724c2e32cf8be5bcda30785539663cdf105cd4c4d792'

FIRST_FAILED_FRAME_SHA256='47634e30cc0823b71fd95735f58de4120a77be1160247f0e2b111e4845067fd1'
FIRST_SUCCESS_FRAME_SHA256='7ae6e7eb3d3dd39be94b42ac85ff47b6c6029fdde51bc2aea35fe24a00f057ab'
FREEZE_FRAME_SHA256='34cc302ee852bc245bf73225acb8cb29d7e2e1c731673d26f28c6eff717b6ad4'
TEST_FREEZE_FRAME_SHA256='90fefcb91bd1bc8b31bd87766b99470203f5a1fdbf0da17fc4c0ac49f0b03711'
PYTHON_FRAME_SHA256='7f543ebe70461b6943b82cd71abaa7f6fc63f88b2dcdccda0abb03dbe2f54006'
RUST_FRAME_SHA256='bd70e6930742b7c5587c41324cc03e0bae43dc2a77524c044f1063294171144a'
POLICY_MISSING_FRAME_SHA256='fa4948ff6f32d899b9c199a2c26ad7e6367e07bd40d56781870bcde5f0b62d80'
POLICY_TIMEOUT_FRAME_SHA256='f62248f1b125834147c33d1fb3151dfba2ba281b2374b253667b89a4f1a01458'
PARITY_PREFREEZE_FRAME_SHA256='5bd163c4f5f3ad8f35240f20f5d72920d04e94afcb90c22f86734e6d14832220'
LLM_FRAME_SHA256='d4dc906ccfac3de46da674cd8cb261e89c1a08eabe35dbca3edf3cf9c6d728fe'
CPP_FRAME_SHA256='27e9af0b2f1d01ce4d81915b20284db714e7330519971ae5d0989d3e5545b21b'
PARITY_OPEN_FRAME_SHA256='cf8db7f8c67afc1a58a12828b1119c0151357c57250780de3f1f17601cb1a4b6'
PYTHON_TOOLCHAIN_SHA256='c7dc38f3c922874a68445613786420f394fd6d55920a4e987d6cec975928fb5f'
PYTHON_COMMAND_SHA256='ff51bdf117d70b7558edd406754f0c55e81cd99e7070e64be178b06b396877c0'
RUST_TOOLCHAIN_SHA256='478b7abcb1fc9eae176fbbe999eaf2d0798d5cc6ffe51700b90436b41a655569'
RUST_COMMAND_SHA256='084f0452e053590db48aa5089cf963223e4444bb8ca920a2eecc5c253be005a2'

POLICY_MISSING=0
POLICY_PRESENT=1
POLICY_TIMEOUT=2
ZERO='0 0 0 0 0 0 0 0'

fail() {
  printf 'pireus quotient novelty forge: FAIL: %s\n' "$*" >&2
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
  local source_hash="$1" command_hash="$2"
  printf '9020 1 2 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${source_hash}")" "${ZERO}" \
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

llm_frame() {
  printf '9020 3 5 6 6 1 0 0 0 1 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PARENT_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${COMMAND_SHA256}")" "${ZERO}" "${ZERO}"
}

cpp_frame() {
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

attempt_forbidden_oracle() {
  local label="$1" language="$2" toolchain="$3" command="$4" frame_sha="$5"
  authorize "${label}" "$(forbidden_frame "${language}" "${toolchain}" "${command}")" \
    "${frame_sha}" 110 \
    'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
  printf 'GUARDIAN_DISPATCH label=%s process_launched=false\n' "${label}"
}

transcript_admitted() {
  local path="$1" expected_sha="$2" expected_lines="$3" expected_bytes="$4" frozen="$5"
  [[ "$(sha_file "${path}")" == "${expected_sha}" ]] &&
    [[ "$(wc -l < "${path}")" -eq "${expected_lines}" ]] &&
    [[ "$(wc -c < "${path}")" -eq "${expected_bytes}" ]] &&
    grep -Fqx -- \
      'SOUNIO_AUTHORITY schema=pireus-quotient-novelty-forge.v5 role=SEMANTIC_AUTHORITY stage=SOUNIO_EXECUTABLE' \
      "${path}" &&
    grep -Fqx -- 'PIREUS_QUOTIENT_ACTIONS matrix_encodings=65536' "${path}" &&
    grep -Fqx -- ' invertible=20160' "${path}" &&
    grep -Fqx -- ' considered=40320' "${path}" &&
    grep -Fqx -- ' admitted=12' "${path}" &&
    grep -Fqx -- ' admitted_no_swap=6' "${path}" &&
    grep -Fqx -- ' admitted_swap=6' "${path}" &&
    grep -Fqx -- ' action_code_failures=0' "${path}" &&
    grep -Fqx -- ' closure_failures=0' "${path}" &&
    grep -Fqx -- ' gauge_equivariance_failures=0' "${path}" &&
    [[ "$(grep -c '^ classes=48$' "${path}")" -eq 2 ]] &&
    [[ "$(grep -c '^ classes=14$' "${path}")" -eq 1 ]] &&
    grep -Fqx -- ' min_size=2' "${path}" &&
    grep -Fqx -- ' max_size=4' "${path}" &&
    grep -Fqx -- ' size_sum=48' "${path}" &&
    grep -Fqx -- ' typed_novelty=1' "${path}" &&
    grep -Fqx -- ' semantics_frozen=0' "${path}" &&
    grep -Fqx -- ' parity_open=0' "${path}" &&
    grep -Fqx -- ' claim_ready=0' "${path}" &&
    grep -Fqx -- ' total=31' "${path}" &&
    grep -Fqx -- 'PIREUS_QUOTIENT_SUMMARY error=0' "${path}" &&
    grep -Fqx -- ' valid=1' "${path}" || return 1
  if [[ "${frozen}" == true ]]; then
    grep -Fqx -- ' frozen_match=1' "${path}" &&
      grep -Fqx -- ' frozen_mismatch_code=0' "${path}"
  else
    ! grep -Fq -- 'frozen_match=' "${path}" &&
      ! grep -Fq -- 'frozen_mismatch_code=' "${path}"
  fi
}

for pair in \
  "${GARDEN_REL}:${GARDEN_SHA256}" \
  "${CONTRACT_REL}:${CONTRACT_SHA256}" \
  "${MODULE_REL}:${MODULE_SHA256}" \
  "${EXAMPLE_REL}:${EXAMPLE_SHA256}" \
  "${TEST_REL}:${TEST_SHA256}" \
  "${FAILED_RECEIPT_REL}:${FAILED_RECEIPT_SHA256}" \
  "${FIRST_RECEIPT_REL}:${FIRST_RECEIPT_SHA256}" \
  "${FIRST_DECISIONS_REL}:${FIRST_DECISIONS_SHA256}" \
  "${FREEZE_REL}:${FREEZE_SHA256}" \
  "${PARITY_REL}:${PARITY_SHA256}" \
  "${DECISIONS_REL}:${DECISIONS_SHA256}" \
  "${FAILED_EVIDENCE_REL}:${FAILED_EVIDENCE_SHA256}" \
  "${FIRST_EVIDENCE_REL}:${FIRST_EVIDENCE_SHA256}" \
  "${FROZEN_EVIDENCE_REL}:${FROZEN_EVIDENCE_SHA256}" \
  "${TEST_EVIDENCE_REL}:${TEST_EVIDENCE_SHA256}"; do
  require_hash "${ROOT}/${pair%%:*}" "${pair#*:}"
done
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
require_hash "${ROOT}/scripts/ci/pireus_cubic_operator_forge.sh" \
  "${PARENT_GATE_SHA256}"
[[ -x "${GUARDIAN}" ]] || fail 'native Loom Guardian unavailable'

chronology=(
  "${GARDEN_COMMIT}:${EXECUTABLE_COMMIT}"
  "${EXECUTABLE_COMMIT}:${FAILED_ATTEMPT_COMMIT}"
  "${FAILED_ATTEMPT_COMMIT}:${COMPACT_REPAIR_COMMIT}"
  "${COMPACT_REPAIR_COMMIT}:${FIRST_EVIDENCE_COMMIT}"
  "${FIRST_EVIDENCE_COMMIT}:${SEMANTIC_FREEZE_COMMIT}"
  "${SEMANTIC_FREEZE_COMMIT}:${CONTRACT_COMMIT}"
  "${CONTRACT_COMMIT}:${FREEZE_RECEIPT_COMMIT}"
  "${FREEZE_RECEIPT_COMMIT}:${PARITY_RECEIPT_COMMIT}"
  "${PARITY_RECEIPT_COMMIT}:HEAD"
)
for edge in "${chronology[@]}"; do
  git -C "${ROOT}" merge-base --is-ancestor "${edge%%:*}" "${edge#*:}" ||
    fail "authority chronology drift: ${edge}"
done

[[ "$(git -C "${ROOT}" show "${EXECUTABLE_COMMIT}:${MODULE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${FIRST_SOURCE_SHA256}" ]] || fail 'first executable source hash drift'
if git -C "${ROOT}" show "${EXECUTABLE_COMMIT}:${MODULE_REL}" |
    grep -Fq 'pireus_quotient_novelty_matches_frozen_semantics'; then
  fail 'frozen matcher existed in first executable'
fi
for forbidden_golden in 'FROZEN_ADMITTED_ACTIONS' 'frozen_actions: [i64; 8]' \
    'frozen_class_counts: [i64; 3]'; do
  if git -C "${ROOT}" show "${EXECUTABLE_COMMIT}:${MODULE_REL}" |
      grep -Fq "${forbidden_golden}"; then
    fail "golden ${forbidden_golden} existed before Sounio observation"
  fi
done
for historical_rel in "${EXAMPLE_REL}" "${TEST_REL}"; do
  for forbidden_golden in 'frozen_match=' 'admitted_actions != 12' \
      'class_counts: [i64; 3] = [48, 48, 14]'; do
    if git -C "${ROOT}" show "${EXECUTABLE_COMMIT}:${historical_rel}" |
        grep -Fq "${forbidden_golden}"; then
      fail "golden ${forbidden_golden} existed in ${historical_rel} before observation"
    fi
  done
done
[[ "$(git -C "${ROOT}" show "${COMPACT_REPAIR_COMMIT}:${MODULE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${REPAIRED_SOURCE_SHA256}" ]] || fail 'repaired matcher-free source drift'
if git -C "${ROOT}" show "${COMPACT_REPAIR_COMMIT}:${MODULE_REL}" |
    grep -Fq 'pireus_quotient_novelty_matches_frozen_semantics'; then
  fail 'frozen matcher existed before first successful observation'
fi
[[ "$(git -C "${ROOT}" show "${FIRST_EVIDENCE_COMMIT}:${FIRST_EVIDENCE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${FIRST_EVIDENCE_SHA256}" ]] || fail 'first successful evidence object drift'
[[ "$(git -C "${ROOT}" show "${SEMANTIC_FREEZE_COMMIT}:${MODULE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${MODULE_SHA256}" ]] || fail 'frozen source object drift'
[[ "$(git -C "${ROOT}" show "${FREEZE_RECEIPT_COMMIT}:${FREEZE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${FREEZE_SHA256}" ]] || fail 'freeze receipt object drift'
[[ "$(git -C "${ROOT}" show "${PARITY_RECEIPT_COMMIT}:${PARITY_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${PARITY_SHA256}" ]] || fail 'parity receipt object drift'

require_line "${ROOT}/${FAILED_RECEIPT_REL}" \
  'status=ATTEMPT_FAILED_NOT_A_SEMANTIC_RESULT'
require_line "${ROOT}/${FAILED_RECEIPT_REL}" 'semantic_result_present=false'
require_line "${ROOT}/${FAILED_RECEIPT_REL}" \
  "stdout_sha256=${FAILED_EVIDENCE_SHA256}"
require_line "${ROOT}/${FAILED_RECEIPT_REL}" \
  "stdout_path=${FAILED_EVIDENCE_REL}"
require_line "${ROOT}/${FIRST_RECEIPT_REL}" 'status=FIRST_SOUNIO_OBSERVATION'
require_line "${ROOT}/${FIRST_RECEIPT_REL}" 'semantic_result_present=true'
require_line "${ROOT}/${FREEZE_REL}" 'stage=SEMANTICS_FROZEN'
require_line "${ROOT}/${FREEZE_REL}" "semantics_sha256=${SEMANTICS_SHA256}"
require_line "${ROOT}/${FREEZE_REL}" "source_manifest_sha256=${SOURCE_MANIFEST_SHA256}"
require_line "${ROOT}/${FREEZE_REL}" 'expected_admitted_actions=12'
require_line "${ROOT}/${FREEZE_REL}" 'expected_q2_classes=14'
require_line "${ROOT}/${FREEZE_REL}" 'expected_q2_min_class_size=2'
require_line "${ROOT}/${FREEZE_REL}" 'expected_q2_max_class_size=4'
require_line "${ROOT}/${FREEZE_REL}" 'relative_algebraic_novelty=false'
require_line "${ROOT}/${FREEZE_REL}" 'claim_ready=false'
require_line "${ROOT}/${CONTRACT_REL}" 'Requested status: `SEMANTICS_FROZEN`'
require_line "${ROOT}/${CONTRACT_REL}" \
  'This contract stops at `SEMANTICS_FROZEN`. It opens no Lean, Koka, C++,'
require_line "${ROOT}/${PARITY_REL}" 'status=OPEN_NOT_EXECUTED'
require_line "${ROOT}/${PARITY_REL}" 'stage=PARITY_OPEN'
require_line "${ROOT}/${PARITY_REL}" 'lean_status=OPEN_NOT_EXECUTED'
require_line "${ROOT}/${PARITY_REL}" 'koka_status=OPEN_NOT_EXECUTED'
require_line "${ROOT}/${PARITY_REL}" 'cpp_status=OPEN_NOT_EXECUTED'
require_line "${ROOT}/${PARITY_REL}" 'python_processes_launched=0'
require_line "${ROOT}/${PARITY_REL}" 'rust_processes_launched=0'
require_line "${ROOT}/${PARITY_REL}" 'claim_ready=false'
require_line "${ROOT}/${DECISIONS_REL}" \
  'decision_05=DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
require_line "${ROOT}/${DECISIONS_REL}" 'decision_05_process_launched=false'
require_line "${ROOT}/${DECISIONS_REL}" \
  'decision_06=DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
require_line "${ROOT}/${DECISIONS_REL}" 'decision_06_process_launched=false'
require_line "${ROOT}/${DECISIONS_REL}" \
  'decision_12=ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
require_line "${ROOT}/${DECISIONS_REL}" 'parity_processes_launched=0'

toolchain_record="engine=lean_single wrapper=${WRAPPER_SHA256} resolver=${RESOLVER_SHA256} compiler=${COMPILER_SHA256}"
[[ "$(sha_text "${toolchain_record}")" == "${TOOLCHAIN_SHA256}" ]] ||
  fail 'toolchain receipt drift'
hardware_record="host=$(hostname) arch=$(uname -m) kernel=$(uname -s) $(uname -r) online_cpus=$(getconf _NPROCESSORS_ONLN) model=$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -n 1)"
[[ "${hardware_record}" == 'host=sounio-workspace-control-0 arch=x86_64 kernel=Linux 7.0.2-5-pve online_cpus=64 model=INTEL(R) XEON(R) GOLD 6526Y' ]] ||
  fail 'live Xeon hardware does not match frozen receipt'
[[ "$(sha_text "${hardware_record}")" == "${HARDWARE_SHA256}" ]] ||
  fail 'hardware receipt drift'
[[ "$(sha_text 'SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pireus_quotient_novelty_forge.sio')" == \
  "${COMMAND_SHA256}" ]] || fail 'authority command drift'
[[ "$(sha_text 'SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/stdlib/hardware/test_pireus_quotient_novelty_forge.sio')" == \
  "${TEST_COMMAND_SHA256}" ]] || fail 'test command drift'
result_record="schema=pireus-quotient-novelty-forge.v5 admitted_actions=12 q0_classes=48 q1_classes=48 q2_classes=14 q2_min=2 q2_max=4 unresolved=1920 selected_child=-1 forge_sha256=${SEMANTICS_SHA256} parity_open=false claim_ready=false"
[[ "$(sha_text "${result_record}")" == "${RESULT_CONTRACT_SHA256}" ]] ||
  fail 'result contract receipt drift'
source_manifest_record="garden=${GARDEN_SHA256} first_source=${FIRST_SOURCE_SHA256} repaired_source=${REPAIRED_SOURCE_SHA256} frozen_source=${MODULE_SHA256} example=${EXAMPLE_SHA256} test=${TEST_SHA256} failure_receipt=${FAILED_RECEIPT_SHA256} first_receipt=${FIRST_RECEIPT_SHA256} contract=${CONTRACT_SHA256} first_result=${FIRST_EVIDENCE_SHA256} frozen_result=${FROZEN_EVIDENCE_SHA256} test_result=${TEST_EVIDENCE_SHA256} parent_semantics=${PARENT_SEMANTICS_SHA256}"
[[ "$(sha_text "${source_manifest_record}")" == "${SOURCE_MANIFEST_SHA256}" ]] ||
  fail 'source manifest receipt drift'

[[ $((65536)) -eq 65536 ]] || fail 'matrix universe drift'
[[ $((20160 * 2)) -eq 40320 ]] || fail 'action universe drift'
[[ $((12 * 12)) -eq 144 ]] || fail 'closure cardinality drift'
[[ $((12 * 12 * 16)) -eq 2304 ]] || fail 'composition cardinality drift'
[[ $((12 * 11 * 256)) -eq 33792 ]] || fail 'gauge cardinality drift'
[[ $((48 * 256)) -eq 12288 ]] || fail 'parent reconstruction drift'
[[ $((48 * 4 * 10)) -eq 1920 ]] || fail 'target obligation drift'

transcript_admitted "${ROOT}/${FIRST_EVIDENCE_REL}" \
  "${FIRST_EVIDENCE_SHA256}" 2916 39979 false ||
  fail 'first successful transcript was not admitted'
transcript_admitted "${ROOT}/${FROZEN_EVIDENCE_REL}" \
  "${FROZEN_EVIDENCE_SHA256}" 2920 40021 true ||
  fail 'frozen transcript was not admitted'
cmp -n 39979 "${ROOT}/${FIRST_EVIDENCE_REL}" \
  "${ROOT}/${FROZEN_EVIDENCE_REL}" >/dev/null ||
  fail 'frozen transcript changed pre-matcher evidence'
[[ "$(tail -n 4 "${ROOT}/${FROZEN_EVIDENCE_REL}")" == \
  $' frozen_match=1\n\n frozen_mismatch_code=0' ]] ||
  fail 'frozen transcript causal suffix drift'
require_line "${ROOT}/${TEST_EVIDENCE_REL}" \
  'PIREUS_QUOTIENT_NOVELTY_FORGE_EXECUTABLE_OK'

parent_gate_output="$("${ROOT}/scripts/ci/pireus_cubic_operator_forge.sh")"
printf '%s\n' "${parent_gate_output}" | grep -Fqx -- \
  'pireus cubic operator forge: STAGE_REACHED_NOT_A_CLAIM stage=PARITY_OPEN language=Sounio children=48 bilinear_witness_checks=3145728 collisions=0 targets=4 unresolved=1920 selected_child=-1 formal=OPEN_NOT_EXECUTED effect=OPEN_NOT_EXECUTED material=OPEN_NOT_EXECUTED claim_ready=false python_process_launched=false rust_process_launched=false' ||
  fail 'parent gate terminal marker drift'
guardian_selftest_output="$("${ROOT}/scripts/ci/sounio_loom_language_authority_selftest.sh")"
printf '%s\n' "${guardian_selftest_output}" | grep -Fqx -- \
  'sounio-loom-language-authority-selftest: PASS language=Sounio cases=33 python=refused rust=refused policy_missing=refused llm_promotion=refused parent_laundering=refused ocaml_realization=admitted ocaml_prefreeze=refused ocaml_parent_laundering=refused ocaml_guardian=admitted ocaml_parity=refused cpp_bootstrap=admitted malformed=refused sabotage_python_rule=admits' ||
  fail 'Guardian selftest terminal marker drift'

authorize FIRST_FAILED_FRAME \
  "$(preexec_frame "${FIRST_SOURCE_SHA256}" "${COMMAND_SHA256}")" \
  "${FIRST_FAILED_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE'
authorize FIRST_SUCCESS_FRAME \
  "$(preexec_frame "${REPAIRED_SOURCE_SHA256}" "${COMMAND_SHA256}")" \
  "${FIRST_SUCCESS_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE'
authorize FREEZE "$(freeze_frame "${POLICY_PRESENT}" "${COMMAND_SHA256}")" \
  "${FREEZE_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'
authorize TEST_FREEZE \
  "$(freeze_frame "${POLICY_PRESENT}" "${TEST_COMMAND_SHA256}")" \
  "${TEST_FREEZE_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'
attempt_forbidden_oracle PYTHON_ORACLE 7 "${PYTHON_TOOLCHAIN_SHA256}" \
  "${PYTHON_COMMAND_SHA256}" "${PYTHON_FRAME_SHA256}"
attempt_forbidden_oracle RUST_ORACLE 8 "${RUST_TOOLCHAIN_SHA256}" \
  "${RUST_COMMAND_SHA256}" "${RUST_FRAME_SHA256}"
authorize POLICY_MISSING "$(freeze_frame "${POLICY_MISSING}" "${COMMAND_SHA256}")" \
  "${POLICY_MISSING_FRAME_SHA256}" 101 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=101 reason=policy-missing next_stage=SOUNIO_EXECUTABLE'
authorize POLICY_TIMEOUT "$(freeze_frame "${POLICY_TIMEOUT}" "${COMMAND_SHA256}")" \
  "${POLICY_TIMEOUT_FRAME_SHA256}" 102 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=102 reason=policy-timeout next_stage=SOUNIO_EXECUTABLE'
authorize PARITY_PREFREEZE "$(parity_frame 2)" \
  "${PARITY_PREFREEZE_FRAME_SHA256}" 112 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=112 reason=wrong-stage next_stage=SOUNIO_EXECUTABLE'
authorize LLM_PROMOTION "$(llm_frame)" "${LLM_FRAME_SHA256}" 119 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=119 reason=review-promoted-to-authority next_stage=SEMANTICS_FROZEN'
authorize CPP_AUTHORITY "$(cpp_frame)" "${CPP_FRAME_SHA256}" 113 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=SEMANTICS_FROZEN'
authorize PARITY_OPEN "$(parity_frame 3)" "${PARITY_OPEN_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'

main_output="$(mktemp /tmp/pireus-qnf-main.XXXXXX)"
test_output="$(mktemp /tmp/pireus-qnf-test.XXXXXX)"
trap 'rm -f "${main_output}" "${test_output}"' EXIT
(
  cd "${ROOT}"
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
    examples/pireus_quotient_novelty_forge.sio >"${main_output}"
)
(
  cd "${ROOT}"
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
    tests/stdlib/hardware/test_pireus_quotient_novelty_forge.sio >"${test_output}"
)
[[ "$(sha_file "${main_output}")" == "${FROZEN_EVIDENCE_SHA256}" ]] ||
  fail 'live frozen transcript drift'
[[ "$(sha_file "${test_output}")" == "${TEST_EVIDENCE_SHA256}" ]] ||
  fail 'live structural test output drift'

printf '%s\n' \
  'pireus quotient novelty forge: STAGE_REACHED_NOT_A_CLAIM stage=PARITY_OPEN language=Sounio admitted_actions=12 q0_classes=48 q1_classes=48 q2_classes=14 targets=4 unresolved=1920 selected_child=-1 formal=OPEN_NOT_EXECUTED effect=OPEN_NOT_EXECUTED material=OPEN_NOT_EXECUTED claim_ready=false python_process_launched=false rust_process_launched=false'
