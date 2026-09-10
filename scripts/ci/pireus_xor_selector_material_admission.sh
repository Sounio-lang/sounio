#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"

PDF="/tmp/intel-sdm-vol-2c-326018-092.pdf"
XED="/tmp/pireus-xed-v2026.08.23-20260827/datafiles/avx512f/avx512-foundation-isa.xed.txt"
GARDEN="${ROOT}/docs/internal/garden/seeds/2026-08-27-pireus-xor-selector-material-admission.md"
MODULE="${ROOT}/stdlib/hardware/pireus/xor_selector_material_admission.sio"
EXAMPLE="${ROOT}/examples/pireus_xor_selector_material_admission.sio"
TEST="${ROOT}/tests/stdlib/hardware/test_pireus_xor_selector_material_admission.sio"
CONCEPT="${ROOT}/docs/internal/concepts/pireus-xor-selector-material-admission.md"
SEMANTICS="${ROOT}/docs/research/pireus_xor_selector_material_admission_semantics.md"
RECEIPT="${ROOT}/docs/research/receipts/pireus_xor_selector_material_admission_20260827.md"
EVIDENCE="${ROOT}/docs/research/evidence/pireus_xor_selector_material_admission_20260827.txt"
REGISTRY="${ROOT}/docs/internal/concepts/registry.tsv"

LOWERING_SOURCE="${ROOT}/stdlib/hardware/pireus/xor_lowering_legality.sio"
LOWERING_SEMANTICS="${ROOT}/docs/research/pireus_xor_lowering_legality_semantics.md"
LOWERING_RECEIPT="${ROOT}/docs/research/receipts/pireus_xor_lowering_legality_20260827.md"
DARWIN_RECEIPT="${ROOT}/docs/research/receipts/pireus_xor_lowering_darwin_xeon_material_parity_20260827.md"
DARWIN_EVIDENCE="${ROOT}/docs/research/evidence/pireus_xor_lowering_darwin_xeon_material_parity_20260827.txt"
APPLE_RECEIPT="${ROOT}/docs/research/receipts/pireus_apple_a64_tbl_material_parity_20260827.md"
APPLE_EVIDENCE="${ROOT}/docs/research/evidence/pireus_apple_a64_tbl_material_parity_20260827.txt"
DGX_RECEIPT="${ROOT}/docs/research/receipts/pireus_dgx_ptx_shfl_material_parity_20260827.md"
DGX_EVIDENCE="${ROOT}/docs/research/evidence/pireus_dgx_ptx_shfl_material_parity_20260827.txt"

GARDEN_COMMIT='b53115358687f2d660d3bc5596f07a37aa4929fb'
EXECUTABLE_COMMIT='fdd444afc5ba0e7529bfee532640dc0a665bfc3f'
SOURCE_SHA256='b9249fe24f5d08fb012631346164d826b8ee975130b0f298a809ad48f4843a66'
SEMANTICS_SHA256='17196cbc2c3fa286c9c2c6e48f042cd3b180d731ee41e0e492077b355ca34ea9'
PARENT_MANIFEST_SHA256='23eeef8d222c99674bc3a3f92ea5cb46772fc5d7a58ed74af36469a9f32ef712'
SOURCE_MANIFEST_SHA256='b3d65a5d278ceb034862af8315a861d556b6dcc0ad01b095196517e3e333b6a1'
TOOLCHAIN_SHA256='2ce5194cdc517de8d7a0063e09d4f4e7b6b5701a23fa1031a3ec9e8f56486b6e'
HARDWARE_SHA256='b6326139297e9ba59d82e208a2404b01e3d57445357a5e803c51c845dd388db0'
AUTHORITY_COMMAND_SHA256='bd6dc15675b133ac133114fffbf70e9d6eb3f851bf5c7864b296726bf5cdee97'
TEST_COMMAND_SHA256='d8d2c02d9a5867727a67556f622e82e13b5e4db6080e0d4996e69f3172d6c72e'
TAMPER_COMMAND_SHA256='2cf3d783d002e3034fbaeea0887c17e09f88d040d1dba6f6206dede430cd857b'
RESULT_SHA256='e8a0c579b064a63837058f6dd2c2d578ea22062444e5c18f769309afa838f176'
TEST_RESULT_SHA256='ed8719a3b76b982213811c753f2b5fa397029755a56f2a163ea0e69a33b47e2e'
ZERO='0 0 0 0 0 0 0 0'

fail() {
  printf 'pireus-xor-selector-material-admission: FAIL: %s\n' "$*" >&2
  exit 1
}

sha_file() {
  sha256sum "$1" | cut -d' ' -f1
}

sha_text() {
  printf '%s\n' "$1" | sha256sum | cut -d' ' -f1
}

require_hash() {
  local path="$1" expected="$2" actual
  [[ -f "${path}" ]] || fail "missing artifact: ${path}"
  actual="$(sha_file "${path}")"
  [[ "${actual}" == "${expected}" ]] ||
    fail "hash drift: ${path}: expected ${expected}, got ${actual}"
}

require_line() {
  local path="$1" line="$2"
  grep -Fqx -- "${line}" "${path}" || fail "missing exact line in ${path}: ${line}"
}

sha_limbs() {
  local hex="$1" out='' i part
  for ((i = 0; i < 8; i++)); do
    part="${hex:$((i * 8)):8}"
    out="${out}${out:+ }$((16#${part}))"
  done
  printf '%s' "${out}"
}

preaction_frame() {
  local command_sha="$1"
  printf '9020 1 2 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${SOURCE_SHA256}")" "${ZERO}" "${ZERO}" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${command_sha}")" "${ZERO}" "${ZERO}"
}

freeze_frame() {
  printf '9020 2 3 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${SOURCE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PARENT_MANIFEST_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${AUTHORITY_COMMAND_SHA256}")" \
    "$(sha_limbs "${RESULT_SHA256}")" "${ZERO}"
}

python_frame() {
  printf '9020 3 4 7 7 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${SOURCE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PARENT_MANIFEST_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${AUTHORITY_COMMAND_SHA256}")" \
    "$(sha_limbs "${RESULT_SHA256}")" "${ZERO}"
}

authorize() {
  local expected_frame_sha="$1" expected_decision="$2" frame="$3" decision
  [[ "$(sha_text "${frame}")" == "${expected_frame_sha}" ]] ||
    fail "Loom frame drift: expected ${expected_frame_sha}"
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  [[ "${decision}" == "${expected_decision}" ]] ||
    fail "Loom decision mismatch: ${decision}"
  printf 'loom_decision=%s frame_sha256=%s\n' "${decision}" "${expected_frame_sha}"
}

authority_command_record() {
  printf '%s' 'SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pireus_xor_selector_material_admission.sio /tmp/intel-sdm-vol-2c-326018-092.pdf /tmp/pireus-xed-v2026.08.23-20260827/datafiles/avx512f/avx512-foundation-isa.xed.txt docs/research/receipts/pireus_xor_lowering_darwin_xeon_material_parity_20260827.md docs/research/evidence/pireus_xor_lowering_darwin_xeon_material_parity_20260827.txt docs/research/receipts/pireus_apple_a64_tbl_material_parity_20260827.md docs/research/evidence/pireus_apple_a64_tbl_material_parity_20260827.txt docs/research/receipts/pireus_dgx_ptx_shfl_material_parity_20260827.md docs/research/evidence/pireus_dgx_ptx_shfl_material_parity_20260827.txt | tee /tmp/pireus-xor-selector-material-admission.authority.txt'
}

test_command_record() {
  printf '%s' 'SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/stdlib/hardware/test_pireus_xor_selector_material_admission.sio /tmp/intel-sdm-vol-2c-326018-092.pdf /tmp/pireus-xed-v2026.08.23-20260827/datafiles/avx512f/avx512-foundation-isa.xed.txt docs/research/receipts/pireus_xor_lowering_darwin_xeon_material_parity_20260827.md docs/research/evidence/pireus_xor_lowering_darwin_xeon_material_parity_20260827.txt docs/research/receipts/pireus_apple_a64_tbl_material_parity_20260827.md docs/research/evidence/pireus_apple_a64_tbl_material_parity_20260827.txt docs/research/receipts/pireus_dgx_ptx_shfl_material_parity_20260827.md docs/research/evidence/pireus_dgx_ptx_shfl_material_parity_20260827.txt'
}

run_authority() {
  (
    cd "${ROOT}"
    SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
      examples/pireus_xor_selector_material_admission.sio \
      "${PDF}" "${XED}" \
      docs/research/receipts/pireus_xor_lowering_darwin_xeon_material_parity_20260827.md \
      docs/research/evidence/pireus_xor_lowering_darwin_xeon_material_parity_20260827.txt \
      docs/research/receipts/pireus_apple_a64_tbl_material_parity_20260827.md \
      docs/research/evidence/pireus_apple_a64_tbl_material_parity_20260827.txt \
      docs/research/receipts/pireus_dgx_ptx_shfl_material_parity_20260827.md \
      docs/research/evidence/pireus_dgx_ptx_shfl_material_parity_20260827.txt \
      | tee /tmp/pireus-xor-selector-material-admission.authority.txt
  )
}

run_test() {
  (
    cd "${ROOT}"
    SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
      tests/stdlib/hardware/test_pireus_xor_selector_material_admission.sio \
      "${PDF}" "${XED}" \
      "${DARWIN_RECEIPT}" "${DARWIN_EVIDENCE}" \
      "${APPLE_RECEIPT}" "${APPLE_EVIDENCE}" \
      "${DGX_RECEIPT}" "${DGX_EVIDENCE}"
  )
}

[[ -x "${GUARDIAN}" ]] || fail "native Sounio Loom guardian unavailable: ${GUARDIAN}"

require_hash "${ROOT}/bin/souc" ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008
require_hash "${ROOT}/bin/souc-lean-single-x86_64" 6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2
require_hash "${PDF}" 939c9543ff98eefb80f5c5a517bf6f08e864497ea8e032334849f3e39a7b3b07
require_hash "${XED}" e9bc0522be4c1a3a3d938eab334c47e306fe759cccf376b9dfb2b9cf7aee0038

require_hash "${GARDEN}" 68b2844934cc1e7544794dd5fdb35d56387a58ad2a536075d81a6378feda34fe
require_hash "${MODULE}" "${SOURCE_SHA256}"
require_hash "${EXAMPLE}" dd7ecca3f8401ec7228033e5829ea8481ede8d5b8c3b3a5e514ab76a4caddb2e
require_hash "${TEST}" 66e64a9b023bb09e611f83210415040e4e1ad5d580cfa1abe4554db7fcb50b8e
require_hash "${CONCEPT}" e690e62601af45f12a4366fcbb2011d494e95c3f48bf7a2bcc6aee7f15bc1bb6
require_hash "${SEMANTICS}" "${SEMANTICS_SHA256}"
require_hash "${RECEIPT}" 2615448449a16faf1d826a6d42e0b0212036f485a3a3e815fc064c298070f979
require_hash "${EVIDENCE}" a59d975337fb4e0d825038e25ba4bf4b11105e28863fdf837d1cba60919ffc7e
require_hash "${REGISTRY}" f64c402492048c091774dbf1fefbf8abfbd2c23181d6393826d7c294bc433274

require_hash "${LOWERING_SOURCE}" 7087649a5cfdb41a884aa9a2e1b0b64bbe2d25da3ca3cd1d54d5b70429854edb
require_hash "${LOWERING_SEMANTICS}" 9e92f256c25a774979a1b02cb10d5d39a1ae702ccec3a273044f91930f03a970
require_hash "${LOWERING_RECEIPT}" daef832ee6370b656e93ae84c76ba6d17c98aaf5ad1dd86674dee27ba0f84346
require_hash "${DARWIN_RECEIPT}" 342d8ba8808c2a926bb2bbf0c09488f7b849967239c932687952ec6ae789a906
require_hash "${DARWIN_EVIDENCE}" ee37914bc738eb829f3589249f228e4a8312310fbffa0b00636cd0c9ed9a40d1
require_hash "${APPLE_RECEIPT}" c00a3d4e556688829efadbbf640ea858cfe9520dc04103fa745cf1a8101f7840
require_hash "${APPLE_EVIDENCE}" 2877bfd463b4d28dc3311b75c69bec2aa1c62b430d08314989187d44b32a781e
require_hash "${DGX_RECEIPT}" 3c10882eff43d3b197428839996c7a04c009c8f537d0c1451bdf3e8a13e2f385
require_hash "${DGX_EVIDENCE}" 2c6b6e448265a5566d17df9a674246ea62c05210e432e48e418d16358496853b

git -C "${ROOT}" cat-file -e "${GARDEN_COMMIT}^{commit}" || fail 'Garden commit missing'
git -C "${ROOT}" cat-file -e "${EXECUTABLE_COMMIT}^{commit}" || fail 'Sounio executable commit missing'
git -C "${ROOT}" merge-base --is-ancestor "${GARDEN_COMMIT}" "${EXECUTABLE_COMMIT}" ||
  fail 'Garden is not an ancestor of the first Sounio executable'
git -C "${ROOT}" merge-base --is-ancestor "${EXECUTABLE_COMMIT}" HEAD ||
  fail 'first Sounio executable is not an ancestor of HEAD'

PARENT_MANIFEST="$({
  cd "${ROOT}"
  sha256sum \
    docs/internal/garden/seeds/2026-08-27-pireus-xor-selector-material-admission.md \
    stdlib/hardware/pireus/xor_lowering_legality.sio \
    docs/research/pireus_xor_lowering_legality_semantics.md \
    docs/research/receipts/pireus_xor_lowering_legality_20260827.md \
    docs/research/receipts/pireus_xor_lowering_darwin_xeon_material_parity_20260827.md \
    docs/research/evidence/pireus_xor_lowering_darwin_xeon_material_parity_20260827.txt \
    docs/research/receipts/pireus_apple_a64_tbl_material_parity_20260827.md \
    docs/research/evidence/pireus_apple_a64_tbl_material_parity_20260827.txt \
    docs/research/receipts/pireus_dgx_ptx_shfl_material_parity_20260827.md \
    docs/research/evidence/pireus_dgx_ptx_shfl_material_parity_20260827.txt
})"
[[ "$(sha_text "${PARENT_MANIFEST}")" == "${PARENT_MANIFEST_SHA256}" ]] || fail 'parent manifest drift'

SOURCE_MANIFEST="$({
  cd "${ROOT}"
  sha256sum \
    stdlib/hardware/pireus/xor_selector_material_admission.sio \
    examples/pireus_xor_selector_material_admission.sio \
    tests/stdlib/hardware/test_pireus_xor_selector_material_admission.sio
})"
[[ "$(sha_text "${SOURCE_MANIFEST}")" == "${SOURCE_MANIFEST_SHA256}" ]] || fail 'source manifest drift'

TOOLCHAIN_RECORD="$(printf '%s\n' \
  'engine=lean_single' \
  'wrapper_path=bin/souc' \
  'wrapper_sha256=ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008' \
  'compiler_path=bin/souc-lean-single-x86_64' \
  'compiler_sha256=6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2')"
[[ "$(sha_text "${TOOLCHAIN_RECORD}")" == "${TOOLCHAIN_SHA256}" ]] || fail 'toolchain record drift'

[[ "$(uname -sr)" == 'Linux 7.0.2-5-pve' ]] || fail 'authority OS drift'
[[ "$(uname -m)" == 'x86_64' ]] || fail 'authority architecture drift'
grep -Fq 'CPU(s):                                  64' < <(lscpu) || fail 'logical CPU count drift'
grep -Fq 'Model name:                              INTEL(R) XEON(R) GOLD 6526Y' < <(lscpu) || fail 'CPU model drift'
grep -Fq 'Thread(s) per core:                      2' < <(lscpu) || fail 'thread topology drift'
grep -Fq 'Core(s) per socket:                      16' < <(lscpu) || fail 'core topology drift'
grep -Fq 'Socket(s):                               2' < <(lscpu) || fail 'socket topology drift'
HARDWARE_RECORD="$(printf '%s\n' \
  'os=Linux 7.0.2-5-pve' \
  'architecture=x86_64' \
  'cpu_model=INTEL(R) XEON(R) GOLD 6526Y' \
  'sockets=2' \
  'cores_per_socket=16' \
  'threads_per_core=2' \
  'logical_cpus=64')"
[[ "$(sha_text "${HARDWARE_RECORD}")" == "${HARDWARE_SHA256}" ]] || fail 'hardware record drift'

[[ "$(sha_text "$(authority_command_record)")" == "${AUTHORITY_COMMAND_SHA256}" ]] || fail 'authority command record drift'
[[ "$(sha_text "$(test_command_record)")" == "${TEST_COMMAND_SHA256}" ]] || fail 'test command record drift'

authorize dd04f8bfed2c784eed7235da05459d53d87c360d51e0c5330f6d9c648496f54d \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE' \
  "$(preaction_frame "${AUTHORITY_COMMAND_SHA256}")"

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pireus-selector-admission.XXXXXX")"
trap 'rm -rf "${WORK_DIR}"' EXIT
for run in 1 2 3; do
  authorize dd04f8bfed2c784eed7235da05459d53d87c360d51e0c5330f6d9c648496f54d \
    'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE' \
    "$(preaction_frame "${AUTHORITY_COMMAND_SHA256}")" >/dev/null
  run_authority >"${WORK_DIR}/authority-${run}.txt"
done
cmp -s "${WORK_DIR}/authority-1.txt" "${WORK_DIR}/authority-2.txt" || fail 'authority streams 1 and 2 differ'
cmp -s "${WORK_DIR}/authority-1.txt" "${WORK_DIR}/authority-3.txt" || fail 'authority streams 1 and 3 differ'
[[ "$(sha_file "${WORK_DIR}/authority-1.txt")" == "${RESULT_SHA256}" ]] || fail 'authority result drift'
[[ "$(wc -l <"${WORK_DIR}/authority-1.txt")" == 201 ]] || fail 'authority line count drift'
[[ "$(wc -c <"${WORK_DIR}/authority-1.txt")" == 3473 ]] || fail 'authority byte count drift'
require_line "${WORK_DIR}/authority-1.txt" 'PIREUS_XOR_ADMISSION_FILES count=6 matched=6'
grep -Fq ' receipts=3' "${WORK_DIR}/authority-1.txt" || fail 'receipt count missing'
grep -Fq ' admitted_nodes=7' "${WORK_DIR}/authority-1.txt" || fail 'admitted-node count missing'
grep -Fq ' unresolved_nodes=8' "${WORK_DIR}/authority-1.txt" || fail 'unresolved-node count missing'
grep -Fq ' refused_nodes=0' "${WORK_DIR}/authority-1.txt" || fail 'refused-node count missing'
require_line "${WORK_DIR}/authority-1.txt" 'PIREUS_XOR_ADMISSION_NEGATIVES passed=22'
grep -Fq ' failures=0' "${WORK_DIR}/authority-1.txt" || fail 'authority failures are nonzero'

authorize 5b2e37c08274ade11a14528b0a30dd7ade7929076f82c010b113da69dca240cd \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE' \
  "$(preaction_frame "${TEST_COMMAND_SHA256}")"
run_test >"${WORK_DIR}/test.txt"
require_line "${WORK_DIR}/test.txt" 'PIREUS_XOR_SELECTOR_MATERIAL_ADMISSION_EXECUTABLE_OK'
[[ "$(sha_file "${WORK_DIR}/test.txt")" == "${TEST_RESULT_SHA256}" ]] || fail 'dedicated test result drift'

TAMPERED_DARWIN='/tmp/pireus-xor-selector-material-admission.tampered-darwin.md'
cp "${DARWIN_RECEIPT}" "${TAMPERED_DARWIN}"
printf '\nTAMPERED_FIXTURE\n' >>"${TAMPERED_DARWIN}"
authorize 4b22f4248e944af127bb1b7d47d04b59e2bd3602e923d6aa0944dd57ea183b3f \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE' \
  "$(preaction_frame "${TAMPER_COMMAND_SHA256}")"
set +e
(
  cd "${ROOT}"
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
    examples/pireus_xor_selector_material_admission.sio \
    "${PDF}" "${XED}" "${TAMPERED_DARWIN}" "${DARWIN_EVIDENCE}" \
    "${APPLE_RECEIPT}" "${APPLE_EVIDENCE}" "${DGX_RECEIPT}" "${DGX_EVIDENCE}"
) >"${WORK_DIR}/tampered.txt"
TAMPER_RC=$?
set -e
[[ "${TAMPER_RC}" == 1 ]] || fail "tampered execution returned ${TAMPER_RC}"
require_line "${WORK_DIR}/tampered.txt" 'PIREUS_XOR_ADMISSION_FILES count=6 matched=5'
grep -Fq ' binding_error=1' "${WORK_DIR}/tampered.txt" || fail 'tampered binding error missing'
grep -Fq ' error=2' "${WORK_DIR}/tampered.txt" || fail 'tampered receipt error missing'

authorize 6c5f66fab86d98939fb8fae746e97dd3c7f0cec779b9c32f64aa0991b1070eef \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN' \
  "$(freeze_frame)"

PYTHON_FRAME="$(python_frame)"
[[ "$(sha_text "${PYTHON_FRAME}")" == e2fe5456c6284f019eba4b5a34ab4662746f6dcb1da397895f78b80bf7413ea6 ]] || fail 'Python frame drift'
set +e
PYTHON_DECISION="$(printf '%s\n' "${PYTHON_FRAME}" | "${GUARDIAN}")"
PYTHON_RC=$?
set -e
[[ "${PYTHON_RC}" == 110 ]] || fail "Python authority request returned ${PYTHON_RC}"
[[ "${PYTHON_DECISION}" == 'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN' ]] ||
  fail "Python authority decision drift: ${PYTHON_DECISION}"
[[ "$(sha_text "${PYTHON_DECISION}")" == 3e2b1112dc7ce41d6c752c48daca33e6ee400b93df1e3fafa795a5709b4aa2a3 ]] || fail 'Python decision hash drift'
printf 'loom_decision=%s frame_sha256=%s interpreter_launch_count=0\n' \
  "${PYTHON_DECISION}" e2fe5456c6284f019eba4b5a34ab4662746f6dcb1da397895f78b80bf7413ea6

require_line "${EVIDENCE}" 'stage=SEMANTICS_FROZEN'
require_line "${EVIDENCE}" 'material_file_match_count=6'
require_line "${EVIDENCE}" 'negative_witnesses_passed=22'
require_line "${EVIDENCE}" 'negative_witnesses_total=22'
require_line "${EVIDENCE}" 'generic_instruction_cost=false'
require_line "${EVIDENCE}" 'cross_isa_equivalence=false'
require_line "${EVIDENCE}" 'transform_authorized=false'
require_line "${EVIDENCE}" 'review_promoted=false'
require_line "${EVIDENCE}" 'parity_open=false'
require_line "${EVIDENCE}" 'claim_ready=false'
require_line "${EVIDENCE}" 'interpreter_launch_count=0'
require_line "${EVIDENCE}" 'rust_used=false'
require_line "${RECEIPT}" '`PARITY_OPEN=false` and `CLAIM_READY=false`.'

printf 'PIREUS_XOR_SELECTOR_MATERIAL_ADMISSION_PASS=true result_sha256=%s test_sha256=%s repeats=3 tamper=E2 python_oracle=E110 interpreter_launch_count=0 semantics_frozen=true parity_open=false claim_ready=false\n' \
  "${RESULT_SHA256}" "${TEST_RESULT_SHA256}"
