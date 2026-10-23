#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"

PARENT_REL='stdlib/hardware/pireus/u250_dual_card_admission.sio'
PARENT_FREEZE_REL='tools/pireus/u250_dual_card_admission.freeze.v0'
MODULE_REL='stdlib/hardware/pireus/u250_material_ingestion.sio'
EXAMPLE_REL='examples/pireus_u250_material_ingestion.sio'
TEST_REL='tests/stdlib/hardware/test_pireus_u250_material_ingestion.sio'
TAMPERED_REL='tests/fixtures/pireus_u250_material_tampered_v1.txt'
CPP_REL='tools/pireus/u250_material_probe.cpp'
RAW_REL='docs/research/evidence/pireus_u250_dl380_material_probe_20260828.txt'
RECEIPT_REL='docs/research/evidence/pireus_u250_dl380_material_parity_20260828.txt'
FREEZE_REL='tools/pireus/u250_material_ingestion.freeze.v1'
EVIDENCE_REL='tools/pireus/evidence/u250_material_ingestion_v1.txt'

PARENT="${ROOT}/${PARENT_REL}"
PARENT_FREEZE="${ROOT}/${PARENT_FREEZE_REL}"
MODULE="${ROOT}/${MODULE_REL}"
EXAMPLE="${ROOT}/${EXAMPLE_REL}"
TEST="${ROOT}/${TEST_REL}"
TAMPERED="${ROOT}/${TAMPERED_REL}"
CPP="${ROOT}/${CPP_REL}"
RAW="${ROOT}/${RAW_REL}"
RECEIPT="${ROOT}/${RECEIPT_REL}"
FREEZE="${ROOT}/${FREEZE_REL}"
EVIDENCE="${ROOT}/${EVIDENCE_REL}"

PARENT_EXECUTABLE_COMMIT='8cff423c79a2140deaf287a330a452c7b36c38c9'
PARENT_FREEZE_COMMIT='d526b70493c2eedc8ebd07f8a90d397297d53b05'
MATERIAL_COMMIT='6bccde236bffdcd85df54721b26bd984b90889a3'
PARENT_SHA256='bf952aa999dad0e74871a0fc78dd6fe67479840a8f334de1c639ceaabd37eafb'
PARENT_SEMANTICS_SHA256='9f0fe0bd01baadec0c60b370bf9dd616a6d2063f1f22b7cdf131f2bc9b6f5586'
PARENT_FREEZE_SHA256='db90647e5ce23029699c2c75232ac8e84ccd9818ec597083f6ce56739843f64a'
MODULE_SHA256='dd24f9da944ecf5427491c5040442bb4f5fd1bd21a3c2394cbcdd585bc2469c2'
EXAMPLE_SHA256='728bfdf7851dbb2d4e526bd6f1b3d3ac9232428f06cd3c4f237348aec10f0a32'
TEST_SHA256='2bb9babbd1b7e839637a341108a0b65d2609924be1a0e8688c856f183a1ffaf8'
TAMPERED_SHA256='711c21a8b60e9c2717ca819b847b41779eafaab0d8f96924122146b76561164f'
CPP_SHA256='13be4f5284667b9ee76da4be4c71547352335076e6efb5700c8ee9c9f26aad80'
CPP_BINARY_SHA256='0a734aee5148432502e336c987d62dd0fedc86ac6e179afeb44d933fd4896e34'
RAW_SHA256='6bea3b962c519dfe9a9878c008a6300b67b920f0a2b51ba9d89dbf180661e7df'
RECEIPT_SHA256='9889567b684fcc0213ed38a44041e8475c4c9a71722b7baa1c6c064e1f1d0d7a'
FREEZE_SHA256='c4e2a0e0c1a4582f1192c185dc3d08ef837e3be19ac5ba982fa8a3327924f7d6'
EVIDENCE_SHA256='e4e1733e854da0ca7ccecb500e41ac1718a03f86737a0cbce43a1956026267fe'
SOURCE_MANIFEST_SHA256='039963d18f3a8c5095f9a5ab7263191506d6c1773c014e0f9de30d0a21ec7295'
SEMANTICS_SHA256='536312cdd0d75fca14ae38d3322ceec2ce931d16853a0842c257e45f087a6794'
WRAPPER_SHA256='ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008'
COMPILER_SHA256='6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2'
SOUNIO_TOOLCHAIN_SHA256='2ce5194cdc517de8d7a0063e09d4f4e7b6b5701a23fa1031a3ec9e8f56486b6e'
SOUNIO_HARDWARE_SHA256='46262c5d0fc8df5734998677c2ad063c686fa2d1120fb8dc18dc5b382c7c4805'
SOUNIO_COMMAND_SHA256='8929deb4cb42b381ab2badda6ce7f947d090303873c185cf02ce6dee3cf181df'
SOUNIO_RESULT_SHA256='c473e8d5b042bd407e04cd5795ea047b7cec810d99067955e4e3a96865e40130'
EXAMPLE_OUTPUT_SHA256='0d397f6494ef61959e3b1302ef7ba4a2aea6dc5ca5c226dbdf6e137223221493'
TEST_OUTPUT_SHA256='9b5a3b81b3b2cae03428cf99fd68324d36e65418fe235eb31185099e040ac3c6'
MATERIAL_TOOLCHAIN_SHA256='2948c4213fd27288a16e3b167c53b3cdb0a3fb9aa450550d9048ef9dc672ff18'
MATERIAL_HARDWARE_SHA256='3a58d573a921a705d01c8f510a54c9f44093cb8fcbd902b05dc0319ceea5f8a6'
MATERIAL_COMMAND_SHA256='7d5289f46cc31abcdb43edcf9183c24c7589268a20f3907819785fca38d556f0'
MATERIAL_PREEXEC_FRAME_SHA256='d92bad5162f30de59240755bf8e7ed85c92152ac81ed619307bdb17a44573ae5'
MATERIAL_SEAL_FRAME_SHA256='0cf0adaf72771d415ccd83ddf5b63e1a4d34cf525022afdd712451a3667d2038'
SOUNIO_SEAL_FRAME_SHA256='a34cd326191d22f47c8e8928bf21def0c1ae371dfa12ab7792d396a77019627b'
PYTHON_FRAME_SHA256='d6549830ba82755e991c92e17cd8dadbb910fc802ac68bdba11f52c4df27c391'
CLAIM_FRAME_SHA256='ce5221e64a6a7c367120d2020ab805da7d7323dfbdb3cf18b13bacdac4bdc822'
PYTHON_TOOLCHAIN_SHA256='5c8cfd947420cd48743adb75469089a210d7782421a4e9e46bfc4c40021fb7cf'
PYTHON_COMMAND_SHA256='d8180e4cc4df008b5f996d14096b846c30bbbff1c24ed1a3343dc7413695f8ca'
ZERO='0 0 0 0 0 0 0 0'

fail() {
  printf 'pireus U250 material parity: FAIL: %s\n' "$*" >&2
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

material_preexec_frame() {
  printf '9020 3 4 4 4 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${PARENT_SHA256}")" \
    "$(sha_limbs "${PARENT_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PARENT_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${MATERIAL_TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${MATERIAL_HARDWARE_SHA256}")" \
    "$(sha_limbs "${MATERIAL_COMMAND_SHA256}")" "${ZERO}" "${ZERO}"
}

material_seal_frame() {
  printf '9020 4 8 4 4 1 0 0 1 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${PARENT_SHA256}")" \
    "$(sha_limbs "${PARENT_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PARENT_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${MATERIAL_TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${MATERIAL_HARDWARE_SHA256}")" \
    "$(sha_limbs "${MATERIAL_COMMAND_SHA256}")" \
    "$(sha_limbs "${RAW_SHA256}")" "${ZERO}"
}

sounio_seal_frame() {
  printf '9020 4 8 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PARENT_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SOUNIO_TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${SOUNIO_HARDWARE_SHA256}")" \
    "$(sha_limbs "${SOUNIO_COMMAND_SHA256}")" \
    "$(sha_limbs "${SOUNIO_RESULT_SHA256}")" "${ZERO}"
}

python_frame() {
  printf '9020 4 4 7 7 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PYTHON_TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${SOUNIO_HARDWARE_SHA256}")" \
    "$(sha_limbs "${PYTHON_COMMAND_SHA256}")" "${ZERO}" "${ZERO}"
}

claim_frame() {
  printf '9020 4 7 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PARENT_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SOUNIO_TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${SOUNIO_HARDWARE_SHA256}")" \
    "$(sha_limbs "${SOUNIO_COMMAND_SHA256}")" \
    "$(sha_limbs "${SOUNIO_RESULT_SHA256}")" "${ZERO}"
}

authorize() {
  local frame="$1" expected_sha="$2" expected_rc="$3" expected="$4"
  local decision rc
  [[ "$(sha_text "${frame}")" == "${expected_sha}" ]] ||
    fail "Guardian frame drift: ${expected_sha}"
  set +e
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  rc=$?
  set -e
  [[ "${rc}" -eq "${expected_rc}" ]] ||
    fail "Guardian exit drift: expected ${expected_rc}, got ${rc}"
  [[ "${decision}" == "${expected}" ]] ||
    fail "Guardian decision drift: ${decision}"
}

[[ -x "${GUARDIAN}" ]] || fail 'native Loom Guardian unavailable'
require_hash "${PARENT}" "${PARENT_SHA256}"
require_hash "${PARENT_FREEZE}" "${PARENT_FREEZE_SHA256}"
require_hash "${MODULE}" "${MODULE_SHA256}"
require_hash "${EXAMPLE}" "${EXAMPLE_SHA256}"
require_hash "${TEST}" "${TEST_SHA256}"
require_hash "${TAMPERED}" "${TAMPERED_SHA256}"
require_hash "${CPP}" "${CPP_SHA256}"
require_hash "${RAW}" "${RAW_SHA256}"
require_hash "${RECEIPT}" "${RECEIPT_SHA256}"
require_hash "${FREEZE}" "${FREEZE_SHA256}"
require_hash "${EVIDENCE}" "${EVIDENCE_SHA256}"
require_hash "${ROOT}/bin/souc" "${WRAPPER_SHA256}"
require_hash "${ROOT}/bin/souc-lean-single-x86_64" "${COMPILER_SHA256}"

git -C "${ROOT}" merge-base --is-ancestor "${PARENT_EXECUTABLE_COMMIT}" \
  "${PARENT_FREEZE_COMMIT}" || fail 'parent semantics were not frozen'
git -C "${ROOT}" merge-base --is-ancestor "${PARENT_FREEZE_COMMIT}" \
  "${MATERIAL_COMMIT}" || fail 'material execution preceded the freeze'
git -C "${ROOT}" merge-base --is-ancestor "${MATERIAL_COMMIT}" HEAD ||
  fail 'material ingestion commit is not an ancestor of HEAD'

actual_manifest="$(
  cd "${ROOT}"
  sha256sum "${MODULE_REL}" "${EXAMPLE_REL}" "${TEST_REL}" \
    "${TAMPERED_REL}" | sha256sum | cut -d' ' -f1
)"
[[ "${actual_manifest}" == "${SOURCE_MANIFEST_SHA256}" ]] ||
  fail 'source manifest drift'
actual_semantics="$(cat "${MODULE}" "${EXAMPLE}" "${TEST}" "${TAMPERED}" |
  sha256sum | cut -d' ' -f1)"
[[ "${actual_semantics}" == "${SEMANTICS_SHA256}" ]] ||
  fail 'semantics bundle drift'

toolchain_record="$(printf '%s\n' \
  'engine=lean_single' 'wrapper_path=bin/souc' \
  "wrapper_sha256=${WRAPPER_SHA256}" \
  'compiler_path=bin/souc-lean-single-x86_64' \
  "compiler_sha256=${COMPILER_SHA256}")"
[[ "$(sha_text "${toolchain_record}")" == "${SOUNIO_TOOLCHAIN_SHA256}" ]] ||
  fail 'Sounio toolchain record drift'
hardware_record="$(printf '%s\n' \
  'hostname=sounio-workspace-control-0' 'os=Linux 7.0.2-5-pve' \
  'architecture=x86_64' 'cpu_model=INTEL(R) XEON(R) GOLD 6526Y' \
  'sockets=2' 'cores_per_socket=16' 'threads_per_core=2' \
  'logical_cpus=64')"
[[ "$(sha_text "${hardware_record}")" == "${SOUNIO_HARDWARE_SHA256}" ]] ||
  fail 'Sounio hardware record drift'
command_record="$(printf '%s\n' \
  'SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pireus_u250_material_ingestion.sio docs/research/evidence/pireus_u250_dl380_material_probe_20260828.txt docs/research/evidence/pireus_u250_dl380_material_parity_20260828.txt' \
  'SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/stdlib/hardware/test_pireus_u250_material_ingestion.sio docs/research/evidence/pireus_u250_dl380_material_probe_20260828.txt docs/research/evidence/pireus_u250_dl380_material_parity_20260828.txt tests/fixtures/pireus_u250_material_tampered_v1.txt')"
[[ "$(sha_text "${command_record}")" == "${SOUNIO_COMMAND_SHA256}" ]] ||
  fail 'Sounio command record drift'
result_record="$(printf '%s\n' \
  'stage=PARITY_OPEN' 'status=711191' 'status_name=INVENTORY_PARTIAL' \
  "raw_probe_sha256=${RAW_SHA256}" \
  "material_receipt_sha256=${RECEIPT_SHA256}" \
  'raw_probe_valid=true' 'parity_receipt_valid=true' \
  'classification_requested=true' 'classification_allowed=true' \
  'semantic_verdict_emitted_by_child=false' 'declared_card_count=2' \
  'material_slot_count=2' 'discovered_card_count=1' \
  'admitted_card_count=1' 'missing_card_count=1' \
  'second_slot=UNRESOLVED' 'inventory_complete=false' \
  'material_parity_ready=true' 'cost_present=false' \
  'speedup_present=false' 'kernel_correctness_present=false' \
  'parity_open=true' 'claim_ready=false' 'failures=0' \
  'tampered_receipt=REFUSED' 'tampered_probe=REFUSED' \
  'python_oracle=PREEXEC_REFUSED' 'python_process_launched=false')"
[[ "$(sha_text "${result_record}")" == "${SOUNIO_RESULT_SHA256}" ]] ||
  fail 'Sounio result record drift'

require_line "${RAW}" 'semantic_verdict_emitted=false'
require_line "${RAW}" 'classification_requested=false'
require_line "${RAW}" 'cost_present=false'
require_line "${RAW}" 'claim_ready=false'
require_line "${RECEIPT}" 'producing_language=C++'
require_line "${RECEIPT}" 'language_role=MATERIAL_PARITY'
require_line "${RECEIPT}" "result_sha256=${RAW_SHA256}"
require_line "${RECEIPT}" 'semantic_verdict_emitted=false'
require_line "${RECEIPT}" 'classification_requested=false'
require_line "${RECEIPT}" 'claim_ready=false'
require_line "${FREEZE}" 'stage=PARITY_OPEN'
require_line "${FREEZE}" "parent_semantics_sha256=${PARENT_SEMANTICS_SHA256}"
require_line "${FREEZE}" "semantics_sha256=${SEMANTICS_SHA256}"
require_line "${FREEZE}" 'expected_status_name=INVENTORY_PARTIAL'
require_line "${FREEZE}" 'expected_admitted_card_count=1'
require_line "${FREEZE}" 'expected_second_slot=UNRESOLVED'
require_line "${FREEZE}" 'expected_claim_ready=false'

authorize "$(material_preexec_frame)" "${MATERIAL_PREEXEC_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
authorize "$(material_seal_frame)" "${MATERIAL_SEAL_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'

work="$(mktemp -d "${TMPDIR:-/tmp}/pireus-u250-material-v1.XXXXXX")"
trap 'rm -rf "${work}"' EXIT
(
  cd "${ROOT}"
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run "${EXAMPLE_REL}" \
    "${RAW_REL}" "${RECEIPT_REL}"
) >"${work}/example.txt"
(
  cd "${ROOT}"
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run "${TEST_REL}" \
    "${RAW_REL}" "${RECEIPT_REL}" "${TAMPERED_REL}"
) >"${work}/test.txt"
require_hash "${work}/example.txt" "${EXAMPLE_OUTPUT_SHA256}"
require_hash "${work}/test.txt" "${TEST_OUTPUT_SHA256}"
require_line "${work}/example.txt" \
  'SOUNIO_SEMANTIC_AUTHORITY schema=pireus.u250-material-ingestion.v1 stage=PARITY_OPEN'
require_line "${work}/example.txt" ' status=711191'
require_line "${work}/example.txt" ' discovered=1'
require_line "${work}/example.txt" ' admitted=1'
require_line "${work}/example.txt" ' missing=1'
require_line "${work}/example.txt" ' parity_open=1'
require_line "${work}/example.txt" ' claim_ready=0'
require_line "${work}/test.txt" \
  'PIREUS_U250_MATERIAL_INGESTION_TEST_PASS sealed=1 status=INVENTORY_PARTIAL declared=2 discovered=1 admitted=1 missing=1 second_slot=UNRESOLVED tampered_receipt=REFUSED tampered_probe=REFUSED cost_present=0 speedup_present=0 kernel_correctness_present=0 claim_ready=0'

authorize "$(sounio_seal_frame)" "${SOUNIO_SEAL_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
authorize "$(python_frame)" "${PYTHON_FRAME_SHA256}" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=PARITY_OPEN'
authorize "$(claim_frame)" "${CLAIM_FRAME_SHA256}" 122 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=122 reason=parity-receipt-missing next_stage=PARITY_OPEN'

printf 'PIREUS_U250_MATERIAL_PARITY_GATE_PASS=true stage=PARITY_OPEN target=AMD_ALVEO_U250 declared=2 discovered=1 admitted=1 missing=1 second_slot=UNRESOLVED material_reexecution=false tampered=REFUSED python_oracle=E110 python_process_launched=false claim_promotion=E122 cost_present=false speedup_present=false kernel_correctness_present=false claim_ready=false\n'
