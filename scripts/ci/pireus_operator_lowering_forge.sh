#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"

GARDEN_REL='tools/pireus/GARDEN_PIREUS_OPERATOR_LOWERING_FORGE_V6.md'
CONTRACT_REL='tools/pireus/PIREUS_OPERATOR_LOWERING_FORGE_CONTRACT_V6.md'
MODULE_REL='stdlib/hardware/pireus/operator_lowering_forge.sio'
EXAMPLE_REL='examples/pireus_operator_lowering_forge.sio'
TEST_REL='tests/stdlib/hardware/test_pireus_operator_lowering_forge.sio'
FIRST_RECEIPT_REL='tools/pireus/operator_lowering_forge.first.v6'
FIRST_DECISIONS_REL='tools/pireus/operator_lowering_forge.guardian-decisions.v6'
FREEZE_REL='tools/pireus/operator_lowering_forge.freeze.v6'
FREEZE_DECISIONS_REL='tools/pireus/operator_lowering_forge.freeze-decisions.v6'
PARITY_REL='tools/pireus/operator_lowering_forge.parity-open.v6'
FIRST_EVIDENCE_REL='tools/pireus/evidence/operator_lowering_forge_v6.txt'
FIRST_TEST_REL='tools/pireus/evidence/operator_lowering_forge_v6.test.txt'
FROZEN_EVIDENCE_REL='tools/pireus/evidence/operator_lowering_forge_v6.frozen.txt'
FROZEN_TEST_REL='tools/pireus/evidence/operator_lowering_forge_v6.test.frozen.txt'

GARDEN_COMMIT='92232819d51a7e5e20fa1bd04d377b91b5b59780'
EXECUTABLE_COMMIT='66dd1e871a03167499fdf347a9cdb053edd9b528'
FIRST_EVIDENCE_COMMIT='6d303318bfda3a3504d9c3accf8171d277ac732d'
MATCHER_COMMIT='1247143a16e3b4f88dc68ad9afa33b23d61eea51'
FREEZE_RECEIPT_COMMIT='bf4d15b67ae04ae4b1ba95d82b8b9d04c3751e7d'
FREEZE_GATE_COMMIT='b186953b0d238cea26f3842c56292cabb1a8d538'
PARITY_RECEIPT_COMMIT='c15e34c51cd40c696ff971dc26b59631abe0263d'

GARDEN_SHA256='42025f4a916441b87d15579f67a55b1da2f4fbc99a19c2619b04278cc325ae8e'
FIRST_SOURCE_SHA256='e71eae6e7673f11b0b2fe843c3b46fc46f3dc3ccf187e5c7fef4f9089aa2a078'
FIRST_EXAMPLE_SHA256='ab4318a6ab08ce066846c43e81a732ac174f1839047e1188a814e7c38be4da0a'
FIRST_TEST_SOURCE_SHA256='aa7481591f62b5fbd4e0e75ba10d62c06078a39547e35cfa65e44a49baa284c3'
MODULE_SHA256='178663aa64bc44938afbe88874268d8078ee1d56e312add965d1470bb3b42ae0'
EXAMPLE_SHA256='1119840ed4254d59f318c4743f01b5397ecfe5b2a05c27bf43799528b827eee7'
TEST_SHA256='4830f3ca7477e740ec2fdbe4da8e3f825c632820b0787989b5df659f66289b97'
CONTRACT_SHA256='f3510dce62daf85916105a3bb872598ca4b9fd342689c67a2b2e7eb8c28a1aa5'
FIRST_RECEIPT_SHA256='12ab83b513efb285951078eeff3371dc02bbdc3fb24241dfadd42cb7ba7cbf12'
FIRST_DECISIONS_SHA256='b902fee80a5c5a7b4d14b30b36167cafdca4e2e76de8774dd72207071c77e8d4'
FREEZE_SHA256='973d620f30337378b760aa185ddbe9897bdd82ce18ee9e212756f519d1ed7181'
FREEZE_DECISIONS_SHA256='0a804686d536f6612c1ebbb4312fd52984d599360b4d05172502c1a186b4188c'
PARITY_SHA256='4dbd89c5a18a2771bda46674b4ad93849e9f0ea160c7c9f42ce511307c7a6eba'
FIRST_EVIDENCE_SHA256='f7dd2398e3c0568f11e1cca5d2712fbe67169771bc2ade53171215f60197e689'
FIRST_TEST_SHA256='38a86df1295851f07a2ebc4550b1d298fb807916494636662e020563db70abb3'
FROZEN_EVIDENCE_SHA256='a7aae82ea8f57ab770036bc384371717d2d4f3eda9ca34aa78d23dd1132ca9b4'
FROZEN_TEST_SHA256='38a86df1295851f07a2ebc4550b1d298fb807916494636662e020563db70abb3'
SOURCE_MANIFEST_SHA256='10dfa6cb0849f3128a57415301a0ba35f1a27fa9830d21ca6b289ef3a2b1e926'
SEMANTICS_SHA256='bd69d1d890506ebf90ff14fe5e3f2b653d5651b968d2fca7fe8ef9298a0c26c1'
PARENT_SEMANTICS_SHA256='9dde3079369c2c90f66805c18abf93a302f5a1d5facf909e39292283ed65bb21'
PARENT_FREEZE_SHA256='640a271bbe1966a3993e72be8fe019b1152530372cfb3ab91ede92011c0fc8c7'
PARENT_GATE_SHA256='1bc9f27cea5a9f4e36a213efae17753b172db101e218a5982c6c3c674d70e29f'

WRAPPER_SHA256='ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008'
RESOLVER_SHA256='a7e37545490745a58731933b0e07db69843cfeea30e739ed554b82d099ec3d84'
COMPILER_SHA256='6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
GUARDIAN_BUILD_SHA256='af7c1098143d0aad108684646df4c72fecca03404557f5494206713486ca09b6'
GUARDIAN_POLICY_SHA256='64bb0118793fe46dcb392abc1a9212eb15bd55047461576a3ef1a6cefa3f17da'
GUARDIAN_MAIN_SHA256='99b6fe7e1c687c3a4e76cfe1585e4826e753f473dff8676dd287eb2f9e0021bc'
GUARDIAN_SELFTEST_SHA256='c9c7f839fb262dbf616716e3c5f0601bb03cfbcbbcfe2fbe09bd0b39894e2a9f'
TOOLCHAIN_SHA256='6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2'
HARDWARE_SHA256='6c0cad13fd376aea694c4a7a73e603194713a938d6198c8ebddf16f3a1a75689'
COMMAND_SHA256='19cd6147b3fcd9a7345f0c5609ec5c82d8a4158189ebe9e9f675156078dd0025'
TEST_COMMAND_SHA256='1d05d41548907e05d3b063d397b3617ae93df612cdc9ceb915c64d60568b9d0a'
CI_COMMAND_SHA256='21b767ceb93ea5a90fddc91f0af8f7ab01bc140182751fbf732391a311656279'
PYTHON_TOOLCHAIN_SHA256='c7dc38f3c922874a68445613786420f394fd6d55920a4e987d6cec975928fb5f'
PYTHON_COMMAND_SHA256='ff51bdf117d70b7558edd406754f0c55e81cd99e7070e64be178b06b396877c0'
RUST_TOOLCHAIN_SHA256='478b7abcb1fc9eae176fbbe999eaf2d0798d5cc6ffe51700b90436b41a655569'
RUST_COMMAND_SHA256='084f0452e053590db48aa5089cf963223e4444bb8ca920a2eecc5c253be005a2'

POLICY_MISSING=0
POLICY_PRESENT=1
POLICY_TIMEOUT=2
POLICY_ERROR=3
ZERO='0 0 0 0 0 0 0 0'

fail() {
  printf 'pireus operator lowering forge: FAIL: %s\n' "$*" >&2
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

sounio_preexec_frame() {
  local command_hash="$1"
  printf '9020 1 2 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" "${ZERO}" \
    "$(sha_limbs "${PARENT_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${command_hash}")" "${ZERO}" "${ZERO}"
}

freeze_frame() {
  local policy="$1" command_hash="$2" result_hash="$3"
  printf '9020 2 3 1 1 %s 1 1 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${policy}" "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PARENT_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${command_hash}")" \
    "$(sha_limbs "${result_hash}")" "${ZERO}"
}

forbidden_frame() {
  local language="$1" toolchain_hash="$2" command_hash="$3"
  printf '9020 3 4 %s 7 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${language}" "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${toolchain_hash}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${command_hash}")" "${ZERO}" "${ZERO}"
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

llm_promotion_frame() {
  printf '9020 3 5 6 6 1 0 0 0 1 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${COMMAND_SHA256}")" "${ZERO}" "${ZERO}"
}

cpp_authority_frame() {
  local semantic_write="$1" expected_write="$2"
  printf '9020 3 4 4 4 1 %s %s 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${semantic_write}" "${expected_write}" \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${COMMAND_SHA256}")" \
    "$(sha_limbs "${FROZEN_EVIDENCE_SHA256}")" "${ZERO}"
}

ci_frame() {
  printf '9020 4 11 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PARENT_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${CI_COMMAND_SHA256}")" \
    "$(sha_limbs "${FROZEN_EVIDENCE_SHA256}")" "${ZERO}"
}

authorize() {
  local label="$1" frame="$2" expected_rc="$3" expected="$4"
  local decision rc
  set +e
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  rc=$?
  set -e
  [[ "${rc}" -eq "${expected_rc}" ]] ||
    fail "Guardian exit drift for ${label}: expected ${expected_rc}, got ${rc}"
  [[ "${decision}" == "${expected}" ]] ||
    fail "Guardian decision drift for ${label}: ${decision}"
  printf 'GUARDIAN_DECISION label=%s frame_sha256=%s decision=%s\n' \
    "${label}" "$(sha_text "${frame}")" "${decision}"
}

transcript_admitted() {
  local path="$1" expected_sha="$2" expected_lines="$3"
  local expected_bytes="$4" frozen="$5"
  [[ "$(sha_file "${path}")" == "${expected_sha}" ]] &&
    [[ "$(wc -l < "${path}")" -eq "${expected_lines}" ]] &&
    [[ "$(wc -c < "${path}")" -eq "${expected_bytes}" ]] &&
    grep -Fqx -- \
      'SOUNIO_AUTHORITY schema=pireus-operator-lowering-forge.v6 role=SEMANTIC_AUTHORITY stage=SOUNIO_EXECUTABLE' \
      "${path}" &&
    grep -Fqx -- ' parent_operator_classes=14' "${path}" &&
    grep -Fqx -- ' parent_selected_child=-1' "${path}" &&
    grep -Fqx -- 'PIREUS_OLF_ATLAS candidates=1120' "${path}" &&
    grep -Fqx -- ' program_classes=560' "${path}" &&
    grep -Fqx -- ' machine_classes=4' "${path}" &&
    grep -Fqx -- ' separation_checks=626080' "${path}" &&
    grep -Fqx -- 'PIREUS_OLF_RESIDUALS count=1120' "${path}" &&
    grep -Fqx -- ' lowering_seeds=560' "${path}" &&
    grep -Fqx -- ' primitive_seeds=420' "${path}" &&
    grep -Fqx -- ' fabric_seeds=140' "${path}" &&
    grep -Fqx -- ' operator_seeds=0' "${path}" &&
    grep -Fqx -- ' discharged_obligations=3360' "${path}" &&
    grep -Fqx -- ' unresolved_obligations=10080' "${path}" &&
    grep -Fqx -- ' admitted_lowerings=0' "${path}" &&
    grep -Fqx -- ' selected=-1' "${path}" &&
    grep -Fqx -- ' ranking=0' "${path}" &&
    grep -Fqx -- ' material_machine_q=0' "${path}" &&
    grep -Fqx -- ' semantics_frozen=0' "${path}" &&
    grep -Fqx -- ' parity_open=0' "${path}" &&
    grep -Fqx -- ' claim_ready=0' "${path}" &&
    grep -Fqx -- 'PIREUS_OLF_NEGATIVES passed=21' "${path}" &&
    grep -Fqx -- ' total=21' "${path}" &&
    grep -Fqx -- 'PIREUS_OLF_SUMMARY error=0' "${path}" &&
    grep -Fqx -- ' failures=0' "${path}" &&
    grep -Fqx -- ' valid=1' "${path}" || return 1
  [[ "$(grep -c '^PIREUS_OLF_CELL id=' "${path}")" -eq 1120 ]] &&
    [[ "$(grep -c '^PIREUS_OLF_OPERATOR_CLASS id=' "${path}")" -eq 14 ]] &&
    [[ "$(grep -c '^PIREUS_OLF_TARGET index=' "${path}")" -eq 4 ]] || return 1
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
  "${FIRST_RECEIPT_REL}:${FIRST_RECEIPT_SHA256}" \
  "${FIRST_DECISIONS_REL}:${FIRST_DECISIONS_SHA256}" \
  "${FREEZE_REL}:${FREEZE_SHA256}" \
  "${FREEZE_DECISIONS_REL}:${FREEZE_DECISIONS_SHA256}" \
  "${PARITY_REL}:${PARITY_SHA256}" \
  "${FIRST_EVIDENCE_REL}:${FIRST_EVIDENCE_SHA256}" \
  "${FIRST_TEST_REL}:${FIRST_TEST_SHA256}" \
  "${FROZEN_EVIDENCE_REL}:${FROZEN_EVIDENCE_SHA256}" \
  "${FROZEN_TEST_REL}:${FROZEN_TEST_SHA256}"; do
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
require_hash "${ROOT}/scripts/ci/pireus_quotient_novelty_forge.sh" \
  "${PARENT_GATE_SHA256}"
require_hash "${ROOT}/tools/pireus/quotient_novelty_forge.freeze.v5" \
  "${PARENT_FREEZE_SHA256}"
[[ -x "${GUARDIAN}" ]] || fail 'native Loom Guardian unavailable'

chronology=(
  "${GARDEN_COMMIT}:${EXECUTABLE_COMMIT}"
  "${EXECUTABLE_COMMIT}:${FIRST_EVIDENCE_COMMIT}"
  "${FIRST_EVIDENCE_COMMIT}:${MATCHER_COMMIT}"
  "${MATCHER_COMMIT}:${FREEZE_RECEIPT_COMMIT}"
  "${FREEZE_RECEIPT_COMMIT}:${FREEZE_GATE_COMMIT}"
  "${FREEZE_GATE_COMMIT}:${PARITY_RECEIPT_COMMIT}"
  "${PARITY_RECEIPT_COMMIT}:HEAD"
)
for edge in "${chronology[@]}"; do
  git -C "${ROOT}" merge-base --is-ancestor "${edge%%:*}" "${edge#*:}" ||
    fail "authority chronology drift: ${edge}"
done

[[ "$(git -C "${ROOT}" show "${EXECUTABLE_COMMIT}:${MODULE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${FIRST_SOURCE_SHA256}" ]] || fail 'first executable source hash drift'
[[ "$(git -C "${ROOT}" show "${EXECUTABLE_COMMIT}:${EXAMPLE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${FIRST_EXAMPLE_SHA256}" ]] || fail 'first executable example hash drift'
[[ "$(git -C "${ROOT}" show "${EXECUTABLE_COMMIT}:${TEST_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${FIRST_TEST_SOURCE_SHA256}" ]] || fail 'first executable test hash drift'
for historical_rel in "${MODULE_REL}" "${EXAMPLE_REL}" "${TEST_REL}"; do
  if git -C "${ROOT}" show "${EXECUTABLE_COMMIT}:${historical_rel}" |
      grep -Fq 'pireus_operator_lowering_frozen_mismatch_code'; then
    fail "frozen matcher existed in first executable: ${historical_rel}"
  fi
  if git -C "${ROOT}" show "${EXECUTABLE_COMMIT}:${historical_rel}" |
      grep -Fq 'frozen_match='; then
    fail "frozen result existed in first executable: ${historical_rel}"
  fi
done
[[ "$(git -C "${ROOT}" show "${FIRST_EVIDENCE_COMMIT}:${FIRST_EVIDENCE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${FIRST_EVIDENCE_SHA256}" ]] || fail 'first Sounio result object drift'
[[ "$(git -C "${ROOT}" show "${MATCHER_COMMIT}:${MODULE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${MODULE_SHA256}" ]] || fail 'frozen matcher source object drift'
[[ "$(git -C "${ROOT}" show "${FREEZE_RECEIPT_COMMIT}:${FREEZE_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${FREEZE_SHA256}" ]] || fail 'freeze receipt object drift'
[[ "$(git -C "${ROOT}" show "${PARITY_RECEIPT_COMMIT}:${PARITY_REL}" | sha256sum | cut -d' ' -f1)" == \
  "${PARITY_SHA256}" ]] || fail 'parity-open receipt object drift'
grep -Fq 'pireus_operator_lowering_frozen_mismatch_code' \
  "${ROOT}/${MODULE_REL}" || fail 'current module does not contain frozen matcher'
grep -Fq ' frozen_match=' "${ROOT}/${EXAMPLE_REL}" ||
  fail 'current example does not expose frozen matcher result'
grep -Fq 'pireus_operator_lowering_frozen_mismatch_code' \
  "${ROOT}/${TEST_REL}" || fail 'current structural test does not execute matcher'

source_manifest="$(printf '%s\n' \
  'schema=pireus-operator-lowering-forge.source-manifest.v6' \
  "garden_sha256=${GARDEN_SHA256}" \
  "module_sha256=${MODULE_SHA256}" \
  "example_sha256=${EXAMPLE_SHA256}" \
  "test_sha256=${TEST_SHA256}" \
  "contract_sha256=${CONTRACT_SHA256}" \
  "first_receipt_sha256=${FIRST_RECEIPT_SHA256}" \
  "first_evidence_sha256=${FIRST_EVIDENCE_SHA256}" \
  "first_test_sha256=${FIRST_TEST_SHA256}")"
[[ "$(sha_text "${source_manifest}")" == "${SOURCE_MANIFEST_SHA256}" ]] ||
  fail 'source manifest drift'
semantics_material="$(printf '%s\n' \
  'schema=pireus-operator-lowering-forge.semantics.v6' \
  "source_manifest_sha256=${SOURCE_MANIFEST_SHA256}" \
  "first_result_sha256=${FIRST_EVIDENCE_SHA256}" \
  'forge_digest=2539393129:4020369131:3147403558:2306440881:94983304:453189920:2257839762:3786373918' \
  'candidate_cells=1120' \
  'program_classes=560' \
  'machine_envelope_classes=4' \
  'unresolved_obligations=10080' \
  'admitted_lowerings=0' \
  'selected_candidate=-1' \
  'material_machine_quotient=NOT_COMPUTED' \
  'claim_ready=false')"
[[ "$(sha_text "${semantics_material}")" == "${SEMANTICS_SHA256}" ]] ||
  fail 'semantic receipt drift'

require_line "${ROOT}/${FREEZE_REL}" 'status=SEMANTICS_FROZEN'
require_line "${ROOT}/${FREEZE_REL}" "source_manifest_sha256=${SOURCE_MANIFEST_SHA256}"
require_line "${ROOT}/${FREEZE_REL}" "semantics_sha256=${SEMANTICS_SHA256}"
require_line "${ROOT}/${FREEZE_REL}" "parent_semantics_sha256=${PARENT_SEMANTICS_SHA256}"
require_line "${ROOT}/${FREEZE_REL}" 'formal_parity_status=NOT_OPENED'
require_line "${ROOT}/${FREEZE_REL}" 'effect_parity_status=NOT_OPENED'
require_line "${ROOT}/${FREEZE_REL}" 'material_parity_status=NOT_OPENED'
require_line "${ROOT}/${FREEZE_REL}" 'unresolved_obligation_taxonomy_complete=false'
require_line "${ROOT}/${FREEZE_REL}" 'scope_material_machine_quotient=NOT_COMPUTED'
require_line "${ROOT}/${FREEZE_REL}" 'historical_novelty=false'
require_line "${ROOT}/${FREEZE_REL}" 'claim_ready=false'
require_line "${ROOT}/${FREEZE_DECISIONS_REL}" \
  'decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'
require_line "${ROOT}/${PARITY_REL}" 'status=OPEN_NOT_EXECUTED'
require_line "${ROOT}/${PARITY_REL}" 'stage=PARITY_OPEN'
require_line "${ROOT}/${PARITY_REL}" \
  'opening_decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
require_line "${ROOT}/${PARITY_REL}" 'lean_status=OPEN_NOT_EXECUTED'
require_line "${ROOT}/${PARITY_REL}" 'koka_status=OPEN_NOT_EXECUTED'
require_line "${ROOT}/${PARITY_REL}" 'cpp_status=OPEN_NOT_EXECUTED'
require_line "${ROOT}/${PARITY_REL}" 'parity_processes_launched=0'
require_line "${ROOT}/${PARITY_REL}" 'claim_ready=false'
require_line "${ROOT}/${FIRST_DECISIONS_REL}" \
  'GUARDIAN_DISPATCH label=PYTHON process_launched=false'
require_line "${ROOT}/${FIRST_DECISIONS_REL}" \
  'GUARDIAN_DISPATCH label=RUST process_launched=false'

[[ $((14 * 4 * 5 * 2 * 2)) -eq 1120 ]] || fail 'candidate product drift'
[[ $((1120 / 2)) -eq 560 ]] || fail 'program quotient cardinality drift'
[[ $(((1120 * 1119 / 2) - 560)) -eq 626080 ]] ||
  fail 'program separation cardinality drift'
[[ $((560 + 420 + 140)) -eq 1120 ]] || fail 'residual seed partition drift'
[[ $((1120 * 3)) -eq 3360 ]] || fail 'discharged obligation drift'
[[ $((1120 * 9)) -eq 10080 ]] || fail 'unresolved obligation drift'

transcript_admitted "${ROOT}/${FIRST_EVIDENCE_REL}" \
  "${FIRST_EVIDENCE_SHA256}" 19333 317144 false ||
  fail 'first Sounio transcript was not admitted'
transcript_admitted "${ROOT}/${FROZEN_EVIDENCE_REL}" \
  "${FROZEN_EVIDENCE_SHA256}" 19337 317186 true ||
  fail 'frozen Sounio transcript was not admitted'
cmp -n 317144 "${ROOT}/${FIRST_EVIDENCE_REL}" \
  "${ROOT}/${FROZEN_EVIDENCE_REL}" >/dev/null ||
  fail 'frozen matcher changed first-result bytes'
[[ "$(tail -n 4 "${ROOT}/${FROZEN_EVIDENCE_REL}")" == \
  $' frozen_match=1\n\n frozen_mismatch_code=0' ]] ||
  fail 'frozen transcript causal suffix drift'
require_line "${ROOT}/${FROZEN_TEST_REL}" \
  'pireus operator lowering forge structural failures=0'

tmp_dir="$(mktemp -d /tmp/pireus-olf-gate.XXXXXX)"
trap 'rm -rf "${tmp_dir}"' EXIT
tampered_transcript="${tmp_dir}/transcript-tamper.txt"
tampered_digest="${tmp_dir}/digest-tamper.txt"
tampered_debt="${tmp_dir}/debt-tamper.txt"
tampered_source="${tmp_dir}/source-tamper.sio"
main_output="${tmp_dir}/main.txt"
test_output="${tmp_dir}/test.txt"
sed '0,/^ admitted_lowerings=0$/s// admitted_lowerings=1/' \
  "${ROOT}/${FROZEN_EVIDENCE_REL}" >"${tampered_transcript}"
if transcript_admitted "${tampered_transcript}" \
    "${FROZEN_EVIDENCE_SHA256}" 19337 317186 true; then
  fail 'tampered result transcript was admitted'
fi
sed '0,/^:3786373918$/s//:3786373919/' \
  "${ROOT}/${FROZEN_EVIDENCE_REL}" >"${tampered_digest}"
if transcript_admitted "${tampered_digest}" \
    "${FROZEN_EVIDENCE_SHA256}" 19337 317186 true; then
  fail 'tampered forge digest was admitted'
fi
sed '0,/^ unresolved_obligations=10080$/s// unresolved_obligations=10079/' \
  "${ROOT}/${FROZEN_EVIDENCE_REL}" >"${tampered_debt}"
if transcript_admitted "${tampered_debt}" \
    "${FROZEN_EVIDENCE_SHA256}" 19337 317186 true; then
  fail 'tampered residual obligation ledger was admitted'
fi
cp "${ROOT}/${MODULE_REL}" "${tampered_source}"
printf '\n' >>"${tampered_source}"
[[ "$(sha_file "${tampered_source}")" != "${MODULE_SHA256}" ]] ||
  fail 'source tamper control did not change hash'

guardian_selftest_output="$("${ROOT}/scripts/ci/sounio_loom_language_authority_selftest.sh")"
printf '%s\n' "${guardian_selftest_output}" | grep -Fqx -- \
  'sounio-loom-language-authority-selftest: PASS language=Sounio cases=33 python=refused rust=refused policy_missing=refused llm_promotion=refused parent_laundering=refused ocaml_realization=admitted ocaml_prefreeze=refused ocaml_parent_laundering=refused ocaml_guardian=admitted ocaml_parity=refused cpp_bootstrap=admitted malformed=refused sabotage_python_rule=admits' ||
  fail 'Guardian selftest terminal marker drift'

authorize CI_PREEXEC "$(ci_frame)" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
authorize FROZEN_REPLAY_PREEXEC "$(sounio_preexec_frame "${COMMAND_SHA256}")" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE'
authorize FROZEN_TEST_PREEXEC "$(sounio_preexec_frame "${TEST_COMMAND_SHA256}")" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE'
authorize FREEZE_SEAL \
  "$(freeze_frame "${POLICY_PRESENT}" "${COMMAND_SHA256}" "${FROZEN_EVIDENCE_SHA256}")" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'
authorize TEST_FREEZE_SEAL \
  "$(freeze_frame "${POLICY_PRESENT}" "${TEST_COMMAND_SHA256}" "${FROZEN_TEST_SHA256}")" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'
authorize PYTHON_ORACLE \
  "$(forbidden_frame 7 "${PYTHON_TOOLCHAIN_SHA256}" "${PYTHON_COMMAND_SHA256}")" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
printf 'GUARDIAN_DISPATCH label=PYTHON_ORACLE process_launched=false\n'
authorize RUST_ORACLE \
  "$(forbidden_frame 8 "${RUST_TOOLCHAIN_SHA256}" "${RUST_COMMAND_SHA256}")" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
printf 'GUARDIAN_DISPATCH label=RUST_ORACLE process_launched=false\n'
authorize POLICY_MISSING \
  "$(freeze_frame "${POLICY_MISSING}" "${COMMAND_SHA256}" "${FROZEN_EVIDENCE_SHA256}")" 101 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=101 reason=policy-missing next_stage=SOUNIO_EXECUTABLE'
authorize POLICY_TIMEOUT \
  "$(freeze_frame "${POLICY_TIMEOUT}" "${COMMAND_SHA256}" "${FROZEN_EVIDENCE_SHA256}")" 102 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=102 reason=policy-timeout next_stage=SOUNIO_EXECUTABLE'
authorize POLICY_ERROR \
  "$(freeze_frame "${POLICY_ERROR}" "${COMMAND_SHA256}" "${FROZEN_EVIDENCE_SHA256}")" 103 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=103 reason=policy-error next_stage=SOUNIO_EXECUTABLE'
authorize CPP_SEMANTIC_WRITE "$(cpp_authority_frame 1 0)" 113 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=SEMANTICS_FROZEN'
authorize CPP_EXPECTED_WRITE "$(cpp_authority_frame 0 1)" 114 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=114 reason=expected-result-authority-required next_stage=SEMANTICS_FROZEN'
authorize LLM_PROMOTION "$(llm_promotion_frame)" 119 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=119 reason=review-promoted-to-authority next_stage=SEMANTICS_FROZEN'
authorize PARITY_PREFREEZE "$(parity_frame 2)" 112 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=112 reason=wrong-stage next_stage=SOUNIO_EXECUTABLE'
authorize PARITY_OPEN "$(parity_frame 3)" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'

parent_gate_output="$("${ROOT}/scripts/ci/pireus_quotient_novelty_forge.sh")"
printf '%s\n' "${parent_gate_output}" | grep -Fqx -- \
  'pireus quotient novelty forge: STAGE_REACHED_NOT_A_CLAIM stage=PARITY_OPEN language=Sounio admitted_actions=12 q0_classes=48 q1_classes=48 q2_classes=14 targets=4 unresolved=1920 selected_child=-1 formal=OPEN_NOT_EXECUTED effect=OPEN_NOT_EXECUTED material=OPEN_NOT_EXECUTED claim_ready=false python_process_launched=false rust_process_launched=false' ||
  fail 'parent gate terminal marker drift'

(
  cd "${ROOT}"
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
    examples/pireus_operator_lowering_forge.sio >"${main_output}"
)
(
  cd "${ROOT}"
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
    tests/stdlib/hardware/test_pireus_operator_lowering_forge.sio >"${test_output}"
)
[[ "$(sha_file "${main_output}")" == "${FROZEN_EVIDENCE_SHA256}" ]] ||
  fail 'live frozen transcript drift'
[[ "$(sha_file "${test_output}")" == "${FROZEN_TEST_SHA256}" ]] ||
  fail 'live structural test drift'
transcript_admitted "${main_output}" \
  "${FROZEN_EVIDENCE_SHA256}" 19337 317186 true ||
  fail 'live transcript did not satisfy frozen admission predicate'
require_line "${test_output}" \
  'pireus operator lowering forge structural failures=0'

printf '%s\n' \
  'pireus operator lowering forge: STAGE_REACHED_NOT_A_CLAIM stage=PARITY_OPEN freeze_scope=BOUNDED_ATLAS_AND_RESIDUAL_TAXONOMY_NOT_LOWERING_SUCCESS language=Sounio operator_classes=14 candidates=1120 program_classes=560 target_envelopes=4 residuals=1120 unresolved=10080 admitted_lowerings=0 selected_candidate=-1 formal=OPEN_NOT_EXECUTED effect=OPEN_NOT_EXECUTED material=OPEN_NOT_EXECUTED claim_ready=false python_process_launched=false rust_process_launched=false'
