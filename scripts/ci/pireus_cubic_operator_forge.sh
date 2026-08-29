#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"

GARDEN_REL='tools/pireus/GARDEN_PIREUS_CUBIC_OPERATOR_FORGE_V4.md'
CONTRACT_REL='tools/pireus/PIREUS_CUBIC_OPERATOR_FORGE_CONTRACT_V4.md'
PARENT_REL='stdlib/hardware/pireus/operator_genome.sio'
MODULE_REL='stdlib/hardware/pireus/cubic_operator_forge.sio'
EXAMPLE_REL='examples/pireus_cubic_operator_forge.sio'
TEST_REL='tests/stdlib/hardware/test_pireus_cubic_operator_forge.sio'
FIRST_RECEIPT_REL='tools/pireus/cubic_operator_forge.first.v4'
FREEZE_REL='tools/pireus/cubic_operator_forge.freeze.v4'
PARITY_RECEIPT_REL='tools/pireus/cubic_operator_forge.parity-open.v4'
GUARDIAN_DECISIONS_REL='tools/pireus/cubic_operator_forge.guardian-decisions.v4'
EVIDENCE_REL='tools/pireus/evidence/cubic_operator_forge_v4.txt'

GARDEN_COMMIT='ff7cf2b50ce11da0e086198885f33c597badae9c'
EXECUTABLE_COMMIT='35e33f2b43bd4e607e90490d57955f70a7823f4b'
FIRST_EVIDENCE_COMMIT='04a6672356a6ef4d982af636c82b18bae03fdce9'
SEMANTIC_FREEZE_COMMIT='7d0c1896bc9552e3733f62b3f4d42dcbd988dd78'
FREEZE_RECEIPT_COMMIT='f241fff8640909607cef7242c18e5422b51f8dd8'

GARDEN_SHA256='67b3cbd928af1c934cf2f68d317a5c80beda3b7e113371341a0ab5c8dfcc4047'
CONTRACT_SHA256='b64a7cfe6f39d12efe60dcae9efc170a52462907036c6489ae9d502420ef4647'
PARENT_SHA256='92765416ad8854376a779ef452f89497e2df77f225bf5a4eb5f74f4cd9004a6d'
PARENT_SEMANTICS_SHA256='99d5e3417550ad3f7b8223b29f25b3d8d8616ac425d4615c37c7f2f402668926'
FIRST_SOURCE_SHA256='fb25f6e4f4e78bed37c6e9400c76e1eb355d7f15d1fa513598778365aad08b29'
MODULE_SHA256='2c295c48bcd2de0f43a42787dcc612f78c7d40d528641e4fec890858d881c974'
EXAMPLE_SHA256='84828d28aa5a7de5240c0ae82a8e7c4ba0bcd86c3930c1365faa7ba937e828cc'
TEST_SHA256='f89d81b190b9670d5f3cf678fa85c89ec23679c7b38cec909b958657d0c775c3'
FIRST_RECEIPT_SHA256='d98251a299f19c807f93bfe272ce6623615e7294586fd49b613677ffb04f031f'
FREEZE_SHA256='1da425c1ff53273825a71b46850e0cd9e7d4cd5b77aa79eb65ef269aadd5a87b'
PARITY_RECEIPT_SHA256='82cdb8875783d34a903b7b599aeeb9501d73eba9ed0a2426040bd708aaf2665a'
GUARDIAN_DECISIONS_SHA256='b311b57b1043111f11ded42aceac1b25c624615cd2181a367230dd785a9a3b8e'
FIRST_EVIDENCE_SHA256='3435ea095019996cd5e3c3bf55810ae033ff07940e7080e728e5acb936f438eb'
EVIDENCE_SHA256='d27915015cabda1d11211968e0bde5655757599d8dc3313fbfc0506877e49694'
SOURCE_MANIFEST_SHA256='ee5e173f44fbf97c8f145bf22aa0e21db07da8203c06193ac2473598dc7acb06'
SEMANTICS_SHA256='e8268af20770dbf292fb39f92793b7b89d1651b2e88193e0cb6ee765dfc1f1ff'
WRAPPER_SHA256='ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008'
RESOLVER_SHA256='a7e37545490745a58731933b0e07db69843cfeea30e739ed554b82d099ec3d84'
COMPILER_SHA256='6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
GUARDIAN_BUILD_SHA256='af7c1098143d0aad108684646df4c72fecca03404557f5494206713486ca09b6'
GUARDIAN_POLICY_SHA256='64bb0118793fe46dcb392abc1a9212eb15bd55047461576a3ef1a6cefa3f17da'
GUARDIAN_MAIN_SHA256='99b6fe7e1c687c3a4e76cfe1585e4826e753f473dff8676dd287eb2f9e0021bc'
GUARDIAN_SELFTEST_SHA256='c9c7f839fb262dbf616716e3c5f0601bb03cfbcbbcfe2fbe09bd0b39894e2a9f'
PARENT_GATE_SHA256='2785013560cd5f3e699d1eff2b78d4c7e0c88c460f8e7ccda37f182371f7745f'
TOOLCHAIN_SHA256='a4d9e4290d4373baa095cdcdd2f6582587323fa392166815ec6676c6178c3590'
HARDWARE_SHA256='ba649e3df628e482654987fc29b86489bfdfc9f69f6bfdc141f3a8d238ee33cd'
COMMAND_SHA256='eb080abc37ed8f08945e003eed2dc7e8852dcafd0fe587eb0187cecd40479178'
TEST_COMMAND_SHA256='8b13511b443cb476ed2111bf501abcaebb716ee5f399d69fe4516ee852f13405'
RESULT_CONTRACT_SHA256='895c2e358892408677042bc7f0a865a0b1c70d70fd343ea7985da52f47600f71'
TEST_OUTPUT_SHA256='66b503c58371f973cd93d85b3f1d940c100b326378deeedf04197ad050c5013b'

FIRST_PREEXEC_FRAME_SHA256='57f2184815d6b85284a789a57df4f849a49e283c7469826002b230575c6a18a4'
FIRST_TEST_PREEXEC_FRAME_SHA256='7aab2076974f0a0335bdbcc3d4c88433af8e0ed123a3434a2c9eac906df8ff0d'
FREEZE_FRAME_SHA256='e5aa62cf6ed30096428a376adf56c74f70f5d97ac6e53324ac24558c755a44a2'
TEST_FREEZE_FRAME_SHA256='4790944a75d8e334e83f62718aa0a8791ebf989de74488babaae9a3384c2ea6e'
POLICY_MISSING_FRAME_SHA256='646534a9f2dcbb3efd8d85815f8d8a19ab7f95351e3533217c506ce89b477296'
POLICY_TIMEOUT_FRAME_SHA256='a9befd174ce73426420e9b3af686fc30e325b0d4b3ad53004af1c9982f0aae64'
PYTHON_FRAME_SHA256='60c8b16c97abc1b1eb7d6f1221a06420d3b1ad76f00153d1082a076d6d6627cb'
RUST_FRAME_SHA256='6eb56497a8ab2dba1977ae4e4a61adc4ffd74aedd17314a292f0fc9781f136c6'
LLM_PROMOTION_FRAME_SHA256='8b7f687c9ad0e5c8c467b02cb7992bbd9c293d4ce0f4accdc23582bfbb31cb5f'
CPP_AUTHORITY_FRAME_SHA256='4411128b50ad9385ef469266c1e58040ec6162f90f50d21420a64fffd57961f8'
PARITY_PREFREEZE_FRAME_SHA256='1e3a167c4b1a6d19be8db443fc4534c7a65ab5088ac8e765c004ff3eac66379a'
PARITY_OPEN_FRAME_SHA256='745f45d2ac041569e4b3ce1ff51fe88c16a28dadc98599c76ef860bba8b904c6'
PYTHON_TOOLCHAIN_SHA256='497ce0938df96d9bf3c159472d251a946c2d6bd832220937d1b885f7759b05ba'
PYTHON_COMMAND_SHA256='72566473f0019fb50e65175a0ac019af5ad2f495ccfc30b664b7735791e968fc'
RUST_TOOLCHAIN_SHA256='b58640570ba9ffcdb2c2d241e4ce8ece9c7d75c6b1e59e308dee3e5f0e10b56d'
RUST_COMMAND_SHA256='0421c7020d972449b7a1f0492945c18a10be46f6a16ade7f8efad157e2e01b00'
POLICY_MISSING=0
POLICY_PRESENT=1
POLICY_TIMEOUT=2
# Eight zero limbs encode an intentionally absent result, waiver, or freeze.
ZERO='0 0 0 0 0 0 0 0'

fail() {
  printf 'pireus cubic operator forge: FAIL: %s\n' "$*" >&2
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
    grep -Fqx -- \
      'SOUNIO_AUTHORITY schema=pireus-cubic-operator-forge.v4 role=SEMANTIC_AUTHORITY stage=SOUNIO_EXECUTABLE' \
      "${path}" &&
    grep -Fqx -- 'PIREUS_CUBIC_GRAMMAR mutations=48' "${path}" &&
    grep -Fqx -- ' descriptor_failures=0' "${path}" &&
    grep -Fqx -- ' support_failures=0' "${path}" &&
    grep -Fqx -- ' witness_failures=0' "${path}" &&
    grep -Fqx -- ' bilinear_witness_checks=3145728' "${path}" &&
    grep -Fqx -- ' bilinear_witness_failures=0' "${path}" &&
    grep -Fqx -- ' pairwise_checks=1128' "${path}" &&
    grep -Fqx -- ' pairwise_failures=0' "${path}" &&
    [[ "$(grep -c '^PIREUS_CUBIC_CHILD id=' "${path}")" -eq 48 ]] &&
    grep -Fqx -- ' sign_failures=0' "${path}" &&
    grep -Fqx -- ' parent_delta_failures=0' "${path}" &&
    grep -Fqx -- ' semantic_cell_failures=0' "${path}" &&
    grep -Fqx -- ' fixture_failures=0' "${path}" &&
    grep -Fqx -- ' groups=1536' "${path}" &&
    grep -Fqx -- ' unresolved=1920' "${path}" &&
    grep -Fqx -- ' selected_child=-1' "${path}" &&
    grep -Fqx -- ' ranking_present=0' "${path}" &&
    grep -Fqx -- ' bilinear_grammar_novelty=1' "${path}" &&
    grep -Fqx -- ' claim_ready=0' "${path}" &&
    grep -Fqx -- ' total=35' "${path}" &&
    grep -Fqx -- ' forge=3894840050' "${path}" &&
    grep -Fqx -- ' failures=0' "${path}" &&
    grep -Fqx -- ' frozen_match=1' "${path}" &&
    grep -Fqx -- ' frozen_mismatch_code=0' "${path}" &&
    grep -Fqx -- ' valid=1' "${path}" &&
    [[ "$(wc -l < "${path}")" -eq 1732 ]] &&
    [[ "$(wc -c < "${path}")" -eq 23637 ]]
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
  local frame decision rc
  frame="$(forbidden_frame "${language}" "${toolchain}" "${command}")"
  [[ "$(sha_text "${frame}")" == "${frame_sha}" ]] ||
    fail "Guardian frame drift: ${label}"
  set +e
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  rc=$?
  set -e
  if [[ "${rc}" -eq 0 ]]; then
    fail "Guardian allowed ${label}; this gate refused to dispatch the process"
  fi
  [[ "${rc}" -eq 110 ]] || fail "${label} denial exit drift: ${rc}"
  [[ "${decision}" == \
    'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN' ]] ||
    fail "${label} denial drift: ${decision}"
  printf 'GUARDIAN_DECISION label=%s decision=%s process_launched=false\n' \
    "${label}" "${decision}"
}

run_sounio_authorized() {
  local label="$1" command_hash="$2" frame_sha="$3" source="$4" output="$5"
  authorize "${label}" "$(freeze_frame "${POLICY_PRESENT}" "${command_hash}")" \
    "${frame_sha}" 0 \
    'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN' >&2
  (
    cd "${ROOT}"
    SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run "${source}" >"${output}"
  )
}

require_hash "${ROOT}/${GARDEN_REL}" "${GARDEN_SHA256}"
require_hash "${ROOT}/${CONTRACT_REL}" "${CONTRACT_SHA256}"
require_hash "${ROOT}/${PARENT_REL}" "${PARENT_SHA256}"
require_hash "${ROOT}/${MODULE_REL}" "${MODULE_SHA256}"
require_hash "${ROOT}/${EXAMPLE_REL}" "${EXAMPLE_SHA256}"
require_hash "${ROOT}/${TEST_REL}" "${TEST_SHA256}"
require_hash "${ROOT}/${FIRST_RECEIPT_REL}" "${FIRST_RECEIPT_SHA256}"
require_hash "${ROOT}/${FREEZE_REL}" "${FREEZE_SHA256}"
require_hash "${ROOT}/${PARITY_RECEIPT_REL}" "${PARITY_RECEIPT_SHA256}"
require_hash "${ROOT}/${GUARDIAN_DECISIONS_REL}" \
  "${GUARDIAN_DECISIONS_SHA256}"
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
require_hash "${ROOT}/scripts/ci/pireus_operator_genome.sh" \
  "${PARENT_GATE_SHA256}"
[[ -x "${GUARDIAN}" ]] || fail 'native Loom Guardian unavailable'

git -C "${ROOT}" merge-base --is-ancestor "${GARDEN_COMMIT}" \
  "${EXECUTABLE_COMMIT}" || fail 'Garden does not precede Sounio executable'
git -C "${ROOT}" merge-base --is-ancestor "${EXECUTABLE_COMMIT}" \
  "${FIRST_EVIDENCE_COMMIT}" || fail 'first executable does not precede evidence'
git -C "${ROOT}" merge-base --is-ancestor "${FIRST_EVIDENCE_COMMIT}" \
  "${SEMANTIC_FREEZE_COMMIT}" || fail 'first evidence does not precede matcher'
git -C "${ROOT}" merge-base --is-ancestor "${SEMANTIC_FREEZE_COMMIT}" \
  "${FREEZE_RECEIPT_COMMIT}" || fail 'matcher does not precede freeze receipt'
git -C "${ROOT}" merge-base --is-ancestor "${FREEZE_RECEIPT_COMMIT}" HEAD ||
  fail 'freeze receipt is not an ancestor of HEAD'

[[ "$(git -C "${ROOT}" show "${EXECUTABLE_COMMIT}:${MODULE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${FIRST_SOURCE_SHA256}" ]] || fail 'first executable source hash drift'
if git -C "${ROOT}" show "${EXECUTABLE_COMMIT}:${MODULE_REL}" |
    grep -Fq 'pireus_cubic_operator_forge_matches_frozen_semantics'; then
  fail 'frozen matcher existed in first executable commit'
fi
for forbidden_golden in 'let negative_counts: [i64; 48] = [' \
    'frozen_population: [i64; 8]' 'frozen_forge: [i64; 8]'; do
  if git -C "${ROOT}" show "${EXECUTABLE_COMMIT}:${MODULE_REL}" |
      grep -Fq "${forbidden_golden}"; then
    fail "golden ${forbidden_golden} existed before first execution"
  fi
done
[[ "$(git -C "${ROOT}" show "${FIRST_EVIDENCE_COMMIT}:${EVIDENCE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${FIRST_EVIDENCE_SHA256}" ]] || fail 'first evidence object drift'
[[ "$(git -C "${ROOT}" show "${FIRST_EVIDENCE_COMMIT}:${EVIDENCE_REL}" | wc -l)" -eq 1730 ]] ||
  fail 'first evidence line count drift'
[[ "$(git -C "${ROOT}" show "${FIRST_EVIDENCE_COMMIT}:${EVIDENCE_REL}" | wc -c)" -eq 23597 ]] ||
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
[[ "$(sha_text 'SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pireus_cubic_operator_forge.sio')" == \
  "${COMMAND_SHA256}" ]] || fail 'authority command drift'
[[ "$(sha_text 'SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/stdlib/hardware/test_pireus_cubic_operator_forge.sio')" == \
  "${TEST_COMMAND_SHA256}" ]] || fail 'test command drift'
result_record="schema=pireus-cubic-operator-forge.v4 mutations=48 bilinear_witness_checks=3145728 pairwise_checks=1128 groups=1536 unresolved_obligations=1920 selected_child=-1 forge_sha256=${SEMANTICS_SHA256} parity_open=false claim_ready=false"
[[ "$(sha_text "${result_record}")" == "${RESULT_CONTRACT_SHA256}" ]] ||
  fail 'result contract receipt drift'
source_manifest_record="garden=${GARDEN_SHA256} first_source=${FIRST_SOURCE_SHA256} frozen_source=${MODULE_SHA256} example=${EXAMPLE_SHA256} test=${TEST_SHA256} first_receipt=${FIRST_RECEIPT_SHA256} parent_semantics=${PARENT_SEMANTICS_SHA256} first_result=${FIRST_EVIDENCE_SHA256} frozen_result=${EVIDENCE_SHA256}"
[[ "$(sha_text "${source_manifest_record}")" == "${SOURCE_MANIFEST_SHA256}" ]] ||
  fail 'source manifest receipt drift'
mutations=48
dimension=16
bilinear_phases=65536
groups_per_child=32
targets=4
obligations_per_target=10
[[ "$((mutations * dimension * dimension))" -eq 12288 ]] ||
  fail 'sign-cell cardinality equation drift'
[[ "$((mutations * bilinear_phases))" -eq 3145728 ]] ||
  fail 'bilinear-witness cardinality equation drift'
[[ "$((mutations * (mutations - 1) / 2))" -eq 1128 ]] ||
  fail 'pairwise cardinality equation drift'
[[ "$((mutations * groups_per_child))" -eq 1536 ]] ||
  fail 'group cardinality equation drift'
[[ "$((mutations * targets * obligations_per_target))" -eq 1920 ]] ||
  fail 'target-obligation cardinality equation drift'

require_line "${ROOT}/${FREEZE_REL}" 'stage=SEMANTICS_FROZEN'
require_line "${ROOT}/${FREEZE_REL}" \
  "source_manifest_record=${source_manifest_record}"
require_line "${ROOT}/${FREEZE_REL}" "source_manifest_sha256=${SOURCE_MANIFEST_SHA256}"
require_line "${ROOT}/${FREEZE_REL}" "semantics_sha256=${SEMANTICS_SHA256}"
require_line "${ROOT}/${FREEZE_REL}" 'expected_mutation_count=48'
require_line "${ROOT}/${FREEZE_REL}" 'expected_bilinear_witness_checks=3145728'
require_line "${ROOT}/${FREEZE_REL}" 'expected_pairwise_failures=0'
require_line "${ROOT}/${FREEZE_REL}" 'expected_unresolved_target_obligations=1920'
require_line "${ROOT}/${FREEZE_REL}" 'expected_selected_child=-1'
require_line "${ROOT}/${FREEZE_REL}" 'relative_bilinear_grammar_novelty=true'
require_line "${ROOT}/${FREEZE_REL}" 'relative_algebraic_novelty=false'
require_line "${ROOT}/${FREEZE_REL}" 'claim_ready=false'
require_line "${ROOT}/${CONTRACT_REL}" 'Status: `SEMANTICS_FROZEN`'
require_line "${ROOT}/${CONTRACT_REL}" \
  'This contract stops at `SEMANTICS_FROZEN`. No Lean, Koka, C++, Haskell, target'
require_line "${ROOT}/${CONTRACT_REL}" \
  'This is `relative_bilinear_grammar_novelty=true`.'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" 'stage=PARITY_OPEN'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" 'first_authority_transcript_lines=1730'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" 'authority_transcript_lines=1732'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" 'opening_language_code=2'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" 'opening_language_role_code=2'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" 'opening_receipt_kind=STAGED_TRANSITION_RECEIPT'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" 'formal_parity_status=OPEN_NOT_EXECUTED'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" 'effect_parity_status=OPEN_NOT_EXECUTED'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" 'material_parity_status=OPEN_NOT_EXECUTED'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" 'generated_children=48'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" 'selected_child=-1'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" 'target_obligations_unresolved=1920'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" \
  'target_names=Darwin_Xeon,Apple_Silicon,DGX_Spark,dual_AMD_Alveo_U250'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" 'parity_processes_launched=false'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" 'target_processes_launched=false'
require_line "${ROOT}/${PARITY_RECEIPT_REL}" 'claim_ready=false'
require_line "${ROOT}/${GUARDIAN_DECISIONS_REL}" \
  'decision_05=DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
require_line "${ROOT}/${GUARDIAN_DECISIONS_REL}" \
  'decision_05_process_launched=false'
require_line "${ROOT}/${GUARDIAN_DECISIONS_REL}" \
  'decision_06=DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
require_line "${ROOT}/${GUARDIAN_DECISIONS_REL}" \
  'decision_06_process_launched=false'
require_line "${ROOT}/${GUARDIAN_DECISIONS_REL}" \
  'decision_07=DENY code=101 reason=policy-missing next_stage=SOUNIO_EXECUTABLE'
require_line "${ROOT}/${GUARDIAN_DECISIONS_REL}" \
  'decision_08=DENY code=102 reason=policy-timeout next_stage=SOUNIO_EXECUTABLE'
require_line "${ROOT}/${GUARDIAN_DECISIONS_REL}" \
  'decision_09=DENY code=112 reason=wrong-stage next_stage=SOUNIO_EXECUTABLE'
require_line "${ROOT}/${GUARDIAN_DECISIONS_REL}" \
  'decision_10=DENY code=119 reason=review-promoted-to-authority next_stage=SEMANTICS_FROZEN'
require_line "${ROOT}/${GUARDIAN_DECISIONS_REL}" \
  'decision_11=DENY code=113 reason=semantic-authority-required next_stage=SEMANTICS_FROZEN'
require_line "${ROOT}/${GUARDIAN_DECISIONS_REL}" \
  'decision_12=ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
require_line "${ROOT}/${GUARDIAN_DECISIONS_REL}" \
  'canonical_semantic_processes_launched=4'
require_line "${ROOT}/${GUARDIAN_DECISIONS_REL}" \
  'parity_processes_launched=0'
require_line "${ROOT}/stdlib/coordination/loom_language_authority.sio" \
  '    if language == 2 { return 2 }'
require_line "${ROOT}/stdlib/coordination/loom_language_authority.sio" \
  '    if action == 4 && stage == 3 { return 4 }'
transcript_admitted "${ROOT}/${EVIDENCE_REL}" ||
  fail 'canonical frozen authority transcript was not admitted'

"${ROOT}/scripts/ci/pireus_operator_genome.sh" >/dev/null
"${ROOT}/scripts/ci/sounio_loom_language_authority_selftest.sh" >/dev/null
authorize FIRST_EXECUTION "$(preexec_frame "${COMMAND_SHA256}")" \
  "${FIRST_PREEXEC_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE'
authorize FIRST_TEST "$(preexec_frame "${TEST_COMMAND_SHA256}")" \
  "${FIRST_TEST_PREEXEC_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE'
authorize FREEZE "$(freeze_frame "${POLICY_PRESENT}" "${COMMAND_SHA256}")" \
  "${FREEZE_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'
authorize TEST_FREEZE "$(freeze_frame "${POLICY_PRESENT}" "${TEST_COMMAND_SHA256}")" \
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
authorize LLM_PROMOTION "$(llm_promotion_frame)" \
  "${LLM_PROMOTION_FRAME_SHA256}" 119 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=119 reason=review-promoted-to-authority next_stage=SEMANTICS_FROZEN'
authorize CPP_AUTHORITY "$(cpp_authority_frame)" \
  "${CPP_AUTHORITY_FRAME_SHA256}" 113 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=SEMANTICS_FROZEN'

umask 077
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/pireus-cubic-forge-v4.XXXXXX")"
cleanup() { rm -rf "${TMP_ROOT}"; }
trap cleanup EXIT
(
  cd "${ROOT}"
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check "${EXAMPLE_REL}" >/dev/null
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check "${TEST_REL}" >/dev/null
)
run_sounio_authorized REPLAY_FREEZE "${COMMAND_SHA256}" \
  "${FREEZE_FRAME_SHA256}" "${EXAMPLE_REL}" "${TMP_ROOT}/authority.txt"
run_sounio_authorized REPLAY_TEST_FREEZE "${TEST_COMMAND_SHA256}" \
  "${TEST_FREEZE_FRAME_SHA256}" "${TEST_REL}" "${TMP_ROOT}/test.txt"
cmp -s "${TMP_ROOT}/authority.txt" "${ROOT}/${EVIDENCE_REL}" ||
  fail 'authority transcript replay drift'
transcript_admitted "${TMP_ROOT}/authority.txt" ||
  fail 'authority transcript replay was not admitted'
require_hash "${TMP_ROOT}/test.txt" "${TEST_OUTPUT_SHA256}"
require_line "${TMP_ROOT}/test.txt" 'PIREUS_CUBIC_OPERATOR_FORGE_EXECUTABLE_OK'

git -C "${ROOT}" show "${FIRST_EVIDENCE_COMMIT}:${EVIDENCE_REL}" \
  >"${TMP_ROOT}/first-authority.txt"
sed -e '/^ frozen_match=1$/d' -e '/^ frozen_mismatch_code=0$/d' \
  "${ROOT}/${EVIDENCE_REL}" >"${TMP_ROOT}/frozen-without-matcher.txt"
cmp -s "${TMP_ROOT}/first-authority.txt" \
  "${TMP_ROOT}/frozen-without-matcher.txt" ||
  fail 'first and frozen transcripts differ beyond the two matcher fields'

for sabotage in support bilinear selection digest freeze claim; do
  candidate="${TMP_ROOT}/tamper-${sabotage}.txt"
  cp "${ROOT}/${EVIDENCE_REL}" "${candidate}"
  case "${sabotage}" in
    support) sed -i '0,/^ support=32$/s// support=31/' "${candidate}" ;;
    bilinear) sed -i 's/^ bilinear_witness_checks=3145728$/ bilinear_witness_checks=3145727/' "${candidate}" ;;
    selection) sed -i 's/^ selected_child=-1$/ selected_child=0/' "${candidate}" ;;
    digest) sed -i 's/^ forge=3894840050$/ forge=3894840051/' "${candidate}" ;;
    freeze) sed -i 's/^ frozen_match=1$/ frozen_match=0/' "${candidate}" ;;
    claim) sed -i 's/^ claim_ready=0$/ claim_ready=1/' "${candidate}" ;;
  esac
  cmp -s "${candidate}" "${ROOT}/${EVIDENCE_REL}" &&
    fail "${sabotage} sabotage did not mutate transcript"
  transcript_admitted "${candidate}" &&
    fail "${sabotage}-tampered transcript was admitted"
done

authorize PARITY_OPEN "$(parity_frame 3)" \
  "${PARITY_OPEN_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'

printf '%s\n' \
  'pireus cubic operator forge: STAGE_REACHED_NOT_A_CLAIM stage=PARITY_OPEN language=Sounio children=48 bilinear_witness_checks=3145728 collisions=0 targets=4 unresolved=1920 selected_child=-1 formal=OPEN_NOT_EXECUTED effect=OPEN_NOT_EXECUTED material=OPEN_NOT_EXECUTED claim_ready=false python_process_launched=false rust_process_launched=false'
