#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN_CANDIDATE="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
GUARDIAN="$(readlink -f "${GUARDIAN_CANDIDATE}")"

GATE_REL='scripts/ci/pireus_multiprobe_block_reuse_admission.sh'
FREEZE_REL='tools/pireus/multiprobe_block_certification.freeze.v14'
MODULE_REL='stdlib/hardware/pireus/multiprobe_block_reuse_admission.sio'
EXAMPLE_REL='examples/pireus_multiprobe_block_reuse_admission.sio'
TEST_REL='tests/stdlib/hardware/test_pireus_multiprobe_block_reuse_admission.sio'
EVIDENCE_REL='tools/pireus/evidence/multiprobe_block_reuse_admission_v14.txt'
TEST_EVIDENCE_REL='tools/pireus/evidence/multiprobe_block_reuse_admission_v14.test.txt'
EVIDENCE_PENDING_REL="${EVIDENCE_REL}.pending"
TEST_EVIDENCE_PENDING_REL="${TEST_EVIDENCE_REL}.pending"

FREEZE_COMMIT='d239261c807ad600a7d3c8f347e2588185cc4294'
SOURCE_BASE_COMMIT='c9c4e93a77b9a94f99ecb6da659402ea552f9f11'
SOURCE_BOUNDARY_COMMIT='0f22cb703fbc0308685ba28bf48d7de39162d4d6'
SOURCE_COMMIT='1fe3c1820af46f4c598dbee5494329334a326127'
POLICY_COMMIT='f3a4128388d47e091e9803d67c097a6976efeb02'
PRESEAL_GATE_COMMIT='PENDING'

FREEZE_SHA256='dab99f29f594de4c83ef15d932c4ddd0eaac388236c94aea8992822a3bd8b42f'
SOURCE_BASE_MODULE_SHA256='a6d3fb0445405811d50429ba87436efaf3e08a1103e4375dfbce7527efa4cfee'
MODULE_SHA256='4e34d2fa92662c682d2ee664ef5ab1a832b430e19b370763db27fe1b30a00a64'
EXAMPLE_SHA256='af6ed3880c3eb1e60ebb48373f6a5448f58fc2b733e071d2f9c086b0230fec2b'
TEST_SHA256='feff53c9b8617fb08f7836ec279cbeb74cc70251b5ae6b0bb22e2c5e1d467091'
SOURCE_MANIFEST_SHA256='d457672724ba7aa7c252607e7cc190f6e8d8e7243bd7ddc8d1e20ffa10fe238a'
PARENT_SOURCE_MANIFEST_SHA256='162beb0a344715c5674e33fb110dad48910a729c58f5756e79c6e892d3dcf768'
SEMANTICS_SHA256='f7b7e81c546bf54a2a92ec374f50465dfb2e0874d52b77fc6ac70484587dc20c'
PARENT_SEMANTICS_SHA256='f7b7e81c546bf54a2a92ec374f50465dfb2e0874d52b77fc6ac70484587dc20c'
EVIDENCE_SHA256='62f6ca1b835fd4d2dc063390555c25e33096c9f6dd9c5cb8830b8766a0a00fe5'
TEST_EVIDENCE_SHA256='96cfd99ff4903381875c0702a0f7d17b0178feed811f1c63f8fe73a14a72b846'
PRESEAL_GATE_SHA256='PENDING'
WRAPPER_SHA256='ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008'
RESOLVER_SHA256='a7e37545490745a58731933b0e07db69843cfeea30e739ed554b82d099ec3d84'
COMPILER_SHA256='6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2'
TOOLCHAIN_SHA256='5feb92bb4a13a9ec55bb3b76732eb8a5dfdcc28bcc38632813e0e6655f1eaed5'
HARDWARE_SHA256='23cfdd2fc963c4b7ef736e54c078d5e1e03b685aff2f9455a20b60bbea8abd3b'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
GUARDIAN_POLICY_SHA256='64bb0118793fe46dcb392abc1a9212eb15bd55047461576a3ef1a6cefa3f17da'

EXAMPLE_COMMAND="SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run ${EXAMPLE_REL} > ${EVIDENCE_PENDING_REL} && mv -- ${EVIDENCE_PENDING_REL} ${EVIDENCE_REL}"
TEST_COMMAND="SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run ${TEST_REL} > ${TEST_EVIDENCE_PENDING_REL} && mv -- ${TEST_EVIDENCE_PENDING_REL} ${TEST_EVIDENCE_REL}"
PYTHON_COMMAND="python3 -c 'open(\"/tmp/pireus-mbc-reuse-python-oracle\",\"w\").write(\"forbidden\")'"
RUST_COMMAND='rustc --version'
LLM_COMMAND='llm-review-promote-to-semantic-authority'
CPP_COMMAND='c++-parity-write-semantic-result'
ZERO='0 0 0 0 0 0 0 0'
FRAME_WORDS=82

fail() {
  printf 'pireus multiprobe block reuse admission: FAIL: %s\n' "$*" >&2
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

require_committed_hash() {
  local commit="$1" path="$2" expected="$3"
  [[ "$(git -C "${ROOT}" show "${commit}:${path}" | sha256sum | cut -d' ' -f1)" == "${expected}" ]] ||
    fail "committed hash drift: ${commit}:${path}"
}

require_line() {
  local path="$1" expected="$2"
  grep -Fqx -- "${expected}" "${path}" ||
    fail "missing exact line in ${path}: ${expected}"
}

sha_limbs() {
  local hex="$1" out='' i part
  [[ "${#hex}" -eq 64 && "${hex}" =~ ^[0-9a-f]{64}$ ]] ||
    fail "invalid SHA-256: ${hex}"
  for ((i=0; i<8; i++)); do
    part="${hex:$((i*8)):8}"
    out="${out}${out:+ }$((16#${part}))"
  done
  printf '%s' "${out}"
}

authority_frame() {
  local stage="$1" action="$2" language="$3" role="$4" policy="$5"
  local semantic_write="$6" expected_write="$7" parity_valid="$8"
  local review_promoted="$9" command_hash="${10}" result_hash="${11}"
  printf '9020 %s %s %s %s %s %s %s %s %s 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${stage}" "${action}" "${language}" "${role}" "${policy}" \
    "${semantic_write}" "${expected_write}" "${parity_valid}" \
    "${review_promoted}" "$(sha_limbs "${SOURCE_MANIFEST_SHA256}")" \
    "$(sha_limbs "${PARENT_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${command_hash}")" \
    "$(sha_limbs "${result_hash}")" "${ZERO}"
}

check_guardian() {
  local label="$1" frame="$2" expected_rc="$3" expected="$4"
  local decision rc frame_sha words
  words="$(wc -w <<<"${frame}" | tr -d ' ')"
  [[ "${words}" -eq "${FRAME_WORDS}" ]] ||
    fail "Guardian frame width drift for ${label}: ${words}"
  frame_sha="$(sha_text "${frame}")"
  set +e
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  rc=$?
  set -e
  [[ "${rc}" -eq "${expected_rc}" ]] ||
    fail "Guardian exit drift for ${label}: ${rc}"
  [[ "${decision}" == "${expected}" ]] ||
    fail "Guardian decision drift for ${label}: ${decision}"
  if [[ "${rc}" -eq 0 ]]; then
    printf 'GUARDIAN_DECISION label=%s frame_sha256=%s decision=%s process_launch_authorized=true process_launched=false\n' \
      "${label}" "${frame_sha}" "${decision}"
  else
    printf 'GUARDIAN_DECISION label=%s frame_sha256=%s decision=%s process_launch_authorized=false process_launched=false\n' \
      "${label}" "${frame_sha}" "${decision}"
  fi
}

[[ "${PRESEAL_GATE_COMMIT}" != PENDING ]] || fail 'gate commit is unsealed'
[[ "${PRESEAL_GATE_SHA256}" != PENDING ]] || fail 'gate hash is unsealed'
[[ -x "${GUARDIAN}" ]] || fail 'native Loom Guardian unavailable'

require_hash "${ROOT}/${FREEZE_REL}" "${FREEZE_SHA256}"
require_hash "${ROOT}/${MODULE_REL}" "${MODULE_SHA256}"
require_hash "${ROOT}/${EXAMPLE_REL}" "${EXAMPLE_SHA256}"
require_hash "${ROOT}/${TEST_REL}" "${TEST_SHA256}"
require_hash "${ROOT}/bin/souc" "${WRAPPER_SHA256}"
require_hash "${ROOT}/scripts/lib/resolve_souc.sh" "${RESOLVER_SHA256}"
require_hash "${ROOT}/bin/souc-lean-single-x86_64" "${COMPILER_SHA256}"
require_hash "${GUARDIAN}" "${GUARDIAN_SHA256}"
require_hash "${ROOT}/stdlib/coordination/loom_language_authority.sio" \
  "${GUARDIAN_POLICY_SHA256}"

git -C "${ROOT}" merge-base --is-ancestor "${FREEZE_COMMIT}" \
  "${SOURCE_BASE_COMMIT}" || fail 'frozen semantics do not precede reuse source'
git -C "${ROOT}" merge-base --is-ancestor "${SOURCE_BASE_COMMIT}" \
  "${SOURCE_BOUNDARY_COMMIT}" ||
  fail 'reuse source does not precede boundary correction'
git -C "${ROOT}" merge-base --is-ancestor "${SOURCE_BOUNDARY_COMMIT}" \
  "${SOURCE_COMMIT}" || fail 'boundary correction does not precede source repair'
git -C "${ROOT}" merge-base --is-ancestor "${SOURCE_COMMIT}" \
  "${PRESEAL_GATE_COMMIT}" || fail 'reuse source does not precede gate'
git -C "${ROOT}" merge-base --is-ancestor "${PRESEAL_GATE_COMMIT}" HEAD ||
  fail 'presealed gate is not an ancestor of HEAD'
git -C "${ROOT}" merge-base --is-ancestor "${POLICY_COMMIT}" \
  "${SOURCE_COMMIT}" || fail 'Guardian policy does not precede reuse source'

require_committed_hash "${FREEZE_COMMIT}" "${FREEZE_REL}" "${FREEZE_SHA256}"
require_committed_hash "${SOURCE_BASE_COMMIT}" "${MODULE_REL}" \
  "${SOURCE_BASE_MODULE_SHA256}"
require_committed_hash "${SOURCE_BASE_COMMIT}" "${TEST_REL}" "${TEST_SHA256}"
require_committed_hash "${SOURCE_BOUNDARY_COMMIT}" "${MODULE_REL}" \
  "${SOURCE_BASE_MODULE_SHA256}"
require_committed_hash "${SOURCE_BOUNDARY_COMMIT}" "${EXAMPLE_REL}" \
  "${EXAMPLE_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${MODULE_REL}" "${MODULE_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${EXAMPLE_REL}" "${EXAMPLE_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${TEST_REL}" "${TEST_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" 'bin/souc' "${WRAPPER_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" 'scripts/lib/resolve_souc.sh' "${RESOLVER_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" 'bin/souc-lean-single-x86_64' "${COMPILER_SHA256}"
require_committed_hash "${POLICY_COMMIT}" \
  'stdlib/coordination/loom_language_authority.sio' "${GUARDIAN_POLICY_SHA256}"
require_committed_hash "${PRESEAL_GATE_COMMIT}" "${GATE_REL}" \
  "${PRESEAL_GATE_SHA256}"

NORMALIZED_GATE_SHA256="$(sed \
  -e "s/^PRESEAL_GATE_COMMIT=.*/PRESEAL_GATE_COMMIT='PENDING'/" \
  -e "s/^PRESEAL_GATE_SHA256=.*/PRESEAL_GATE_SHA256='PENDING'/" \
  "${ROOT}/${GATE_REL}" | sha256sum | cut -d' ' -f1)"
[[ "${NORMALIZED_GATE_SHA256}" == "${PRESEAL_GATE_SHA256}" ]] ||
  fail 'live gate differs from preseal beyond its two seal fields'

SOURCE_BASE_DELTA="$(git -C "${ROOT}" diff-tree --no-commit-id --name-only -r "${SOURCE_BASE_COMMIT}" | LC_ALL=C sort)"
[[ "${SOURCE_BASE_DELTA}" == $'.claude/llm_offload_log.md\nexamples/pireus_multiprobe_block_reuse_admission.sio\nstdlib/hardware/pireus/multiprobe_block_reuse_admission.sio\ntests/stdlib/hardware/test_pireus_multiprobe_block_reuse_admission.sio' ]] ||
  fail 'reuse source base commit escaped its exact path allowlist'
SOURCE_BOUNDARY_DELTA="$(git -C "${ROOT}" diff-tree --no-commit-id --name-only -r "${SOURCE_BOUNDARY_COMMIT}" | LC_ALL=C sort)"
[[ "${SOURCE_BOUNDARY_DELTA}" == $'.claude/llm_offload_log.md\nexamples/pireus_multiprobe_block_reuse_admission.sio' ]] ||
  fail 'reuse boundary correction escaped its exact path allowlist'
SOURCE_DELTA="$(git -C "${ROOT}" diff-tree --no-commit-id --name-only -r "${SOURCE_COMMIT}" | LC_ALL=C sort)"
[[ "${SOURCE_DELTA}" == $'.claude/llm_offload_log.md\nstdlib/hardware/pireus/multiprobe_block_reuse_admission.sio' ]] ||
  fail 'reuse source repair escaped its exact path allowlist'
[[ "$(git -C "${ROOT}" diff-tree --no-commit-id --name-only -r "${PRESEAL_GATE_COMMIT}")" == "${GATE_REL}" ]] ||
  fail 'gate preseal commit escaped its exact path allowlist'

require_line "${ROOT}/${FREEZE_REL}" 'status=SEMANTICS_FROZEN'
require_line "${ROOT}/${FREEZE_REL}" 'stage=SEMANTICS_FROZEN'
require_line "${ROOT}/${FREEZE_REL}" \
  "source_manifest_sha256=${PARENT_SOURCE_MANIFEST_SHA256}"
require_line "${ROOT}/${FREEZE_REL}" "semantics_sha256=${SEMANTICS_SHA256}"
require_line "${ROOT}/${FREEZE_REL}" 'expected_result_digests=0'
require_line "${ROOT}/${FREEZE_REL}" 'completed_blocks=0'
require_line "${ROOT}/${FREEZE_REL}" 'reuse_hits=0'
require_line "${ROOT}/${FREEZE_REL}" 'certified_blocks=0'
require_line "${ROOT}/${FREEZE_REL}" 'parity_open=false'
require_line "${ROOT}/${FREEZE_REL}" 'claim_ready=false'

SOURCE_MANIFEST="$(cd "${ROOT}" && sha256sum \
  "${MODULE_REL}" "${EXAMPLE_REL}" "${TEST_REL}")"
[[ "$(sha_text "${SOURCE_MANIFEST}")" == "${SOURCE_MANIFEST_SHA256}" ]] ||
  fail 'reuse source manifest drift'

TOOLCHAIN_RECORD="$(printf '%s\n' \
  'engine=lean_single' \
  'wrapper=bin/souc' \
  "wrapper_sha256=${WRAPPER_SHA256}" \
  'resolver=scripts/lib/resolve_souc.sh' \
  "resolver_sha256=${RESOLVER_SHA256}" \
  'compiler=bin/souc-lean-single-x86_64' \
  "compiler_sha256=${COMPILER_SHA256}")"
[[ "$(sha_text "${TOOLCHAIN_RECORD}")" == "${TOOLCHAIN_SHA256}" ]] ||
  fail 'toolchain record drift'
CPU_MODEL="$(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -n 1)"
HARDWARE_RECORD="$(printf '%s\n' \
  "hostname=$(uname -n)" \
  "architecture=$(uname -m)" \
  "cpu_model=${CPU_MODEL}")"
[[ "$(sha_text "${HARDWARE_RECORD}")" == "${HARDWARE_SHA256}" ]] ||
  fail 'hardware record drift'
[[ "${CPU_MODEL}" == 'INTEL(R) XEON(R) GOLD 6526Y' ]] ||
  fail 'execution CPU identity drift'

[[ "${EXAMPLE_COMMAND}" == SOUNIO_SOUC_ENGINE=lean_single\ ./bin/souc\ run* ]] ||
  fail 'example command bypasses canonical wrapper'
[[ "${TEST_COMMAND}" == SOUNIO_SOUC_ENGINE=lean_single\ ./bin/souc\ run* ]] ||
  fail 'test command bypasses canonical wrapper'
[[ "${EXAMPLE_COMMAND}" != *souc-lean-single-x86_64* ]] ||
  fail 'example command invokes raw compiler directly'
[[ "${TEST_COMMAND}" != *souc-lean-single-x86_64* ]] ||
  fail 'test command invokes raw compiler directly'
[[ ! -e "${ROOT}/${EVIDENCE_REL}" ]] ||
  fail "reuse evidence already exists: ${EVIDENCE_REL}"
[[ ! -e "${ROOT}/${TEST_EVIDENCE_REL}" ]] ||
  fail "reuse test evidence already exists: ${TEST_EVIDENCE_REL}"
[[ ! -e "${ROOT}/${EVIDENCE_PENDING_REL}" ]] ||
  fail "reuse pending evidence already exists: ${EVIDENCE_PENDING_REL}"
[[ ! -e "${ROOT}/${TEST_EVIDENCE_PENDING_REL}" ]] ||
  fail "reuse pending test evidence already exists: ${TEST_EVIDENCE_PENDING_REL}"
[[ ! -e /tmp/pireus-mbc-reuse-python-oracle ]] ||
  fail 'forbidden Python oracle marker pre-exists'

GATE_COMPLETE=false
cleanup_partial_evidence() {
  if [[ "${GATE_COMPLETE}" != true ]]; then
    rm -f -- "${ROOT}/${EVIDENCE_PENDING_REL}" \
      "${ROOT}/${TEST_EVIDENCE_PENDING_REL}" \
      "${ROOT}/${EVIDENCE_REL}" "${ROOT}/${TEST_EVIDENCE_REL}"
  fi
}
trap cleanup_partial_evidence EXIT

EXAMPLE_COMMAND_SHA256="$(sha_text "${EXAMPLE_COMMAND}")"
TEST_COMMAND_SHA256="$(sha_text "${TEST_COMMAND}")"
PYTHON_COMMAND_SHA256="$(sha_text "${PYTHON_COMMAND}")"
RUST_COMMAND_SHA256="$(sha_text "${RUST_COMMAND}")"
LLM_COMMAND_SHA256="$(sha_text "${LLM_COMMAND}")"
CPP_COMMAND_SHA256="$(sha_text "${CPP_COMMAND}")"

check_guardian POLICY_MISSING \
  "$(authority_frame 3 8 1 1 0 0 0 0 0 "${EXAMPLE_COMMAND_SHA256}" "${EVIDENCE_SHA256}")" 101 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=101 reason=policy-missing next_stage=SEMANTICS_FROZEN'
check_guardian POLICY_TIMEOUT \
  "$(authority_frame 3 8 1 1 2 0 0 0 0 "${EXAMPLE_COMMAND_SHA256}" "${EVIDENCE_SHA256}")" 102 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=102 reason=policy-timeout next_stage=SEMANTICS_FROZEN'
check_guardian POLICY_ERROR \
  "$(authority_frame 3 8 1 1 3 0 0 0 0 "${EXAMPLE_COMMAND_SHA256}" "${EVIDENCE_SHA256}")" 103 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=103 reason=policy-error next_stage=SEMANTICS_FROZEN'
check_guardian PYTHON_ORACLE \
  "$(authority_frame 3 8 7 7 1 0 0 0 0 "${PYTHON_COMMAND_SHA256}" "${EVIDENCE_SHA256}")" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
check_guardian RUST_ORACLE \
  "$(authority_frame 3 8 8 7 1 0 0 0 0 "${RUST_COMMAND_SHA256}" "${EVIDENCE_SHA256}")" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
check_guardian LLM_PROMOTION \
  "$(authority_frame 3 5 6 6 1 0 0 0 1 "${LLM_COMMAND_SHA256}" "${EVIDENCE_SHA256}")" 119 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=119 reason=review-promoted-to-authority next_stage=SEMANTICS_FROZEN'
check_guardian CPP_SEMANTIC_WRITE \
  "$(authority_frame 3 8 4 4 1 1 0 0 0 "${CPP_COMMAND_SHA256}" "${EVIDENCE_SHA256}")" 113 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=SEMANTICS_FROZEN'

check_guardian SOUNIO_REUSE_FIXTURE \
  "$(authority_frame 3 8 1 1 1 0 0 0 0 "${EXAMPLE_COMMAND_SHA256}" "${EVIDENCE_SHA256}")" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'
set +e
(cd "${ROOT}" && bash -c "${EXAMPLE_COMMAND}")
EXAMPLE_RC=$?
set -e
printf 'PROCESS_RESULT label=SOUNIO_REUSE_FIXTURE process_launch_authorized=true process_launched=true exit_code=%s\n' \
  "${EXAMPLE_RC}"
[[ "${EXAMPLE_RC}" -eq 0 ]] || fail "Sounio reuse fixture exited ${EXAMPLE_RC}"

check_guardian SOUNIO_REUSE_FIXTURE_TEST \
  "$(authority_frame 3 8 1 1 1 0 0 0 0 "${TEST_COMMAND_SHA256}" "${TEST_EVIDENCE_SHA256}")" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'
set +e
(cd "${ROOT}" && bash -c "${TEST_COMMAND}")
TEST_RC=$?
set -e
printf 'PROCESS_RESULT label=SOUNIO_REUSE_FIXTURE_TEST process_launch_authorized=true process_launched=true exit_code=%s\n' \
  "${TEST_RC}"
[[ "${TEST_RC}" -eq 0 ]] || fail "Sounio reuse fixture test exited ${TEST_RC}"

[[ ! -e /tmp/pireus-mbc-reuse-python-oracle ]] ||
  fail 'forbidden Python oracle process created a marker'
require_hash "${ROOT}/${EVIDENCE_REL}" "${EVIDENCE_SHA256}"
require_hash "${ROOT}/${TEST_EVIDENCE_REL}" "${TEST_EVIDENCE_SHA256}"
[[ "$(wc -l < "${ROOT}/${EVIDENCE_REL}")" -eq 2 ]] ||
  fail 'reuse evidence line count drift'
[[ "$(wc -l < "${ROOT}/${TEST_EVIDENCE_REL}")" -eq 1 ]] ||
  fail 'reuse test evidence line count drift'
require_line "${ROOT}/${EVIDENCE_REL}" \
  'SOUNIO_AUTHORITY schema=pireus-multiprobe-block-reuse-admission.v14 role=SEMANTIC_AUTHORITY parent_semantics_sha256=f7b7e81c546bf54a2a92ec374f50465dfb2e0874d52b77fc6ac70484587dc20c stage=SEMANTICS_FROZEN'
require_line "${ROOT}/${EVIDENCE_REL}" \
  'PIREUS_MBC_REUSE_FIXTURE_ADMISSION fixture_only=1 fixture_eligible=1 work_replayed=0 actual_cache_hit=0 actual_reuse_admitted=0 material_receipt_admitted=0 negative_passed=19 negative_total=19 claim_ready=0'
require_line "${ROOT}/${TEST_EVIDENCE_REL}" \
  'PIREUS_MULTIPROBE_BLOCK_REUSE_ADMISSION_OK'

GATE_COMPLETE=true
printf 'PIREUS_MBC_REUSE_FIXTURE_ADMISSION_GATE_PASS=true stage=SEMANTICS_FROZEN parent_semantics_sha256=%s source_manifest_sha256=%s fixture_only=true fixture_eligible=true work_replayed=false actual_cache_hit=false actual_reuse_admitted=false material_receipt_admitted=false negative_passed=19 negative_total=19 claim_ready=false python_dispatch=E110 rust_dispatch=E110 llm_promotion=E119 cpp_semantic_write=E113 policy_missing=E101 policy_timeout=E102 policy_error=E103 engine=lean_single explicit_bootstrap_fallback=true raw_elf_process_launched=false\n' \
  "${SEMANTICS_SHA256}" "${SOURCE_MANIFEST_SHA256}"
