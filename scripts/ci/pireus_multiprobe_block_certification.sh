#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN_CANDIDATE="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
GUARDIAN="$(readlink -f "${GUARDIAN_CANDIDATE}")"

GARDEN_REL='tools/pireus/GARDEN_PIREUS_MULTIPROBE_BLOCK_CERTIFICATION_V14.md'
CONTRACT_REL='tools/pireus/PIREUS_MULTIPROBE_BLOCK_CERTIFICATION_CONTRACT_V14.md'
MODULE_REL='stdlib/hardware/pireus/multiprobe_block_certification.sio'
EXAMPLE_REL='examples/pireus_multiprobe_block_certification.sio'
TEST_REL='tests/stdlib/hardware/test_pireus_multiprobe_block_certification.sio'
PARENT_TRANSCRIPT_REL='tools/pireus/operator_orbit_canonicalization.first.v13'
FIRST_TRANSCRIPT_REL='tools/pireus/multiprobe_block_certification.first.v14'
TEST_EVIDENCE_REL='tools/pireus/evidence/multiprobe_block_certification_v14.test.txt'

GARDEN_COMMIT='db5a1ab31fb616e846a757bd52dd201a0cf5d703'
EXECUTABLE_COMMIT='2b5d42a9db2b84307784f48696fd5c27756ce49c'
PARENT_EVIDENCE_COMMIT='22fbabe81cf365c0b542d8a425ec4c081f31e390'
POLICY_COMMIT='f3a4128388d47e091e9803d67c097a6976efeb02'
GARDEN_SHA256='2bc8b791549a30c041d07801c712b8a859bbb9fd5933177802fa9eda3bcd2a9c'
CONTRACT_SHA256='1f7fb55f32b413683ce45b4bdfebc686600f14b4346ca780e29ec8bbf91c8a48'
MODULE_SHA256='d6d38280a2934956ce1721775c978911e1cce9c526c05fb11c75c7f00ed2ddac'
EXAMPLE_SHA256='71b9157c6867f1fe33f0d1a1709ef1412f3ee924937ff6501854490428988ca5'
TEST_SHA256='58974ff011fc680481d21c31cb299f20a2c24105bf07d3571ffc14b82dadc4ea'
PARENT_TRANSCRIPT_SHA256='16af63f5e0f8aa7e5c899f4c395404b83fb402f6bbdb5f20dea2a3d10ad2e19f'
WRAPPER_SHA256='ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008'
RESOLVER_SHA256='a7e37545490745a58731933b0e07db69843cfeea30e739ed554b82d099ec3d84'
COMPILER_SHA256='6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
GUARDIAN_POLICY_SHA256='64bb0118793fe46dcb392abc1a9212eb15bd55047461576a3ef1a6cefa3f17da'

MAIN_COMMAND="SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run ${EXAMPLE_REL} > ${FIRST_TRANSCRIPT_REL}"
TEST_COMMAND="SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run ${TEST_REL} > ${TEST_EVIDENCE_REL}"
PYTHON_COMMAND="python3 -c 'open(\"/tmp/pireus-mbc-python-oracle\",\"w\").write(\"forbidden\")'"
RUST_COMMAND='rustc --version'
PARITY_COMMAND='lake build SounioPireusMultiprobeBlockCertification'
ZERO='0 0 0 0 0 0 0 0'

fail() {
  printf 'pireus multiprobe block certification: FAIL: %s\n' "$*" >&2
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
  local command_hash="$6"
  printf '9020 %s %s %s %s %s 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${stage}" "${action}" "${language}" "${role}" "${policy}" \
    "$(sha_limbs "${MODULE_SHA256}")" "${ZERO}" "${ZERO}" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${command_hash}")" "${ZERO}" "${ZERO}"
}

check_guardian() {
  local label="$1" frame="$2" expected_rc="$3" expected="$4"
  local decision rc frame_sha
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
    printf 'GUARDIAN_DECISION label=%s frame_sha256=%s decision=%s process_launch_authorized=true\n' \
      "${label}" "${frame_sha}" "${decision}"
  else
    printf 'GUARDIAN_DECISION label=%s frame_sha256=%s decision=%s process_launched=false\n' \
      "${label}" "${frame_sha}" "${decision}"
  fi
}

[[ -x "${GUARDIAN}" ]] || fail 'native Loom Guardian unavailable'
require_hash "${ROOT}/${GARDEN_REL}" "${GARDEN_SHA256}"
require_hash "${ROOT}/${CONTRACT_REL}" "${CONTRACT_SHA256}"
require_hash "${ROOT}/${MODULE_REL}" "${MODULE_SHA256}"
require_hash "${ROOT}/${EXAMPLE_REL}" "${EXAMPLE_SHA256}"
require_hash "${ROOT}/${TEST_REL}" "${TEST_SHA256}"
require_hash "${ROOT}/${PARENT_TRANSCRIPT_REL}" "${PARENT_TRANSCRIPT_SHA256}"
require_hash "${ROOT}/bin/souc" "${WRAPPER_SHA256}"
require_hash "${ROOT}/scripts/lib/resolve_souc.sh" "${RESOLVER_SHA256}"
require_hash "${ROOT}/bin/souc-lean-single-x86_64" "${COMPILER_SHA256}"
require_hash "${GUARDIAN}" "${GUARDIAN_SHA256}"
require_hash "${ROOT}/stdlib/coordination/loom_language_authority.sio" \
  "${GUARDIAN_POLICY_SHA256}"

git -C "${ROOT}" merge-base --is-ancestor "${GARDEN_COMMIT}" \
  "${EXECUTABLE_COMMIT}" || fail 'Garden does not precede executable'
git -C "${ROOT}" merge-base --is-ancestor "${PARENT_EVIDENCE_COMMIT}" \
  "${EXECUTABLE_COMMIT}" || fail 'V13 parent evidence does not precede executable'
git -C "${ROOT}" merge-base --is-ancestor "${POLICY_COMMIT}" \
  "${EXECUTABLE_COMMIT}" || fail 'Guardian policy does not precede executable'
git -C "${ROOT}" merge-base --is-ancestor "${EXECUTABLE_COMMIT}" HEAD ||
  fail 'matcher-free executable commit is not an ancestor of HEAD'

[[ "$(git -C "${ROOT}" show "${GARDEN_COMMIT}:${GARDEN_REL}" | sha256sum | cut -d' ' -f1)" == "${GARDEN_SHA256}" ]] ||
  fail 'committed Garden hash drift'
[[ "$(git -C "${ROOT}" show "${GARDEN_COMMIT}:${CONTRACT_REL}" | sha256sum | cut -d' ' -f1)" == "${CONTRACT_SHA256}" ]] ||
  fail 'committed Garden contract hash drift'
[[ "$(git -C "${ROOT}" show "${EXECUTABLE_COMMIT}:${MODULE_REL}" | sha256sum | cut -d' ' -f1)" == "${MODULE_SHA256}" ]] ||
  fail 'committed matcher-free module hash drift'
[[ "$(git -C "${ROOT}" show "${EXECUTABLE_COMMIT}:${EXAMPLE_REL}" | sha256sum | cut -d' ' -f1)" == "${EXAMPLE_SHA256}" ]] ||
  fail 'committed matcher-free example hash drift'
[[ "$(git -C "${ROOT}" show "${EXECUTABLE_COMMIT}:${TEST_REL}" | sha256sum | cut -d' ' -f1)" == "${TEST_SHA256}" ]] ||
  fail 'committed matcher-free test hash drift'
[[ "$(git -C "${ROOT}" show "${PARENT_EVIDENCE_COMMIT}:${PARENT_TRANSCRIPT_REL}" | sha256sum | cut -d' ' -f1)" == "${PARENT_TRANSCRIPT_SHA256}" ]] ||
  fail 'committed V13 parent transcript hash drift'
[[ "$(git -C "${ROOT}" show "${POLICY_COMMIT}:stdlib/coordination/loom_language_authority.sio" | sha256sum | cut -d' ' -f1)" == "${GUARDIAN_POLICY_SHA256}" ]] ||
  fail 'committed Guardian policy hash drift'
if git -C "${ROOT}" grep -q -E \
  'matches_frozen|frozen_mismatch|expected_(strategy|probe|partition|plan|address)' \
  "${EXECUTABLE_COMMIT}" -- "${MODULE_REL}" "${EXAMPLE_REL}" "${TEST_REL}"; then
  fail 'V14 matcher leaked into matcher-free executable commit'
fi

[[ ! -e "${ROOT}/${FIRST_TRANSCRIPT_REL}" ]] ||
  fail "first transcript already exists: ${FIRST_TRANSCRIPT_REL}"
[[ ! -e "${ROOT}/${TEST_EVIDENCE_REL}" ]] ||
  fail "test evidence already exists: ${TEST_EVIDENCE_REL}"
[[ ! -e /tmp/pireus-mbc-python-oracle ]] ||
  fail 'forbidden Python oracle marker pre-exists'

TOOLCHAIN_RECORD="$(printf '%s\n' \
  'engine=lean_single' \
  'wrapper=bin/souc' \
  "wrapper_sha256=${WRAPPER_SHA256}" \
  'resolver=scripts/lib/resolve_souc.sh' \
  "resolver_sha256=${RESOLVER_SHA256}" \
  'compiler=bin/souc-lean-single-x86_64' \
  "compiler_sha256=${COMPILER_SHA256}")"
TOOLCHAIN_SHA256="$(sha_text "${TOOLCHAIN_RECORD}")"
CPU_MODEL="$(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -n 1)"
HARDWARE_RECORD="$(printf '%s\n' \
  "hostname=$(uname -n)" \
  "architecture=$(uname -m)" \
  "cpu_model=${CPU_MODEL}")"
HARDWARE_SHA256="$(sha_text "${HARDWARE_RECORD}")"
[[ -n "${CPU_MODEL}" ]] || fail 'hardware CPU model unavailable'

MAIN_COMMAND_SHA256="$(sha_text "${MAIN_COMMAND}")"
TEST_COMMAND_SHA256="$(sha_text "${TEST_COMMAND}")"
PYTHON_COMMAND_SHA256="$(sha_text "${PYTHON_COMMAND}")"
RUST_COMMAND_SHA256="$(sha_text "${RUST_COMMAND}")"
PARITY_COMMAND_SHA256="$(sha_text "${PARITY_COMMAND}")"

check_guardian POLICY_MISSING \
  "$(authority_frame 1 2 1 1 0 "${MAIN_COMMAND_SHA256}")" 101 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=101 reason=policy-missing next_stage=GARDEN'
check_guardian POLICY_TIMEOUT \
  "$(authority_frame 1 2 1 1 2 "${MAIN_COMMAND_SHA256}")" 102 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=102 reason=policy-timeout next_stage=GARDEN'
check_guardian POLICY_ERROR \
  "$(authority_frame 1 2 1 1 3 "${MAIN_COMMAND_SHA256}")" 103 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=103 reason=policy-error next_stage=GARDEN'
check_guardian PYTHON_ORACLE \
  "$(authority_frame 1 2 7 7 1 "${PYTHON_COMMAND_SHA256}")" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=GARDEN'
check_guardian RUST_ORACLE \
  "$(authority_frame 1 2 8 7 1 "${RUST_COMMAND_SHA256}")" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=GARDEN'
check_guardian PARITY_PREFREEZE \
  "$(authority_frame 2 4 2 2 1 "${PARITY_COMMAND_SHA256}")" 112 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=112 reason=wrong-stage next_stage=SOUNIO_EXECUTABLE'
check_guardian SOUNIO_FIRST_RUN \
  "$(authority_frame 1 2 1 1 1 "${MAIN_COMMAND_SHA256}")" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE'

(
  cd "${ROOT}"
  bash -c "${MAIN_COMMAND}"
)

check_guardian SOUNIO_STRUCTURAL_TEST \
  "$(authority_frame 1 2 1 1 1 "${TEST_COMMAND_SHA256}")" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE'

(
  cd "${ROOT}"
  bash -c "${TEST_COMMAND}"
)

[[ ! -e /tmp/pireus-mbc-python-oracle ]] ||
  fail 'forbidden Python oracle process created a marker'
require_line "${ROOT}/${FIRST_TRANSCRIPT_REL}" \
  'SOUNIO_AUTHORITY schema=pireus-multiprobe-block-certification.v14 role=SEMANTIC_AUTHORITY stage=SOUNIO_EXECUTABLE matcher_free=1 expected_results=0'
require_line "${ROOT}/${FIRST_TRANSCRIPT_REL}" \
  'PIREUS_MBC_PLAN work_domains=2048 work_addresses=2048 domain_failures=0 address_recomputations=2048 address_recomputation_failures=0 completed_blocks=0 reuse_hits=0 expected_results=0'
require_line "${ROOT}/${FIRST_TRANSCRIPT_REL}" \
  'PIREUS_MBC_BOUNDARY actual_blocks=0 reused=0 certified=0 purity_proved=0 material_bound_profile=0 collision_free=0 formal_parity=0 effect_parity=0 material_parity=0 denotational_parity=0 performance=0 speedup=0 cache_hit_rate=0 algorithmic=0 material_novelty=0 scientific=0 global=0 historical=0 priority=0 claim_ready=0'
require_line "${ROOT}/${FIRST_TRANSCRIPT_REL}" \
  'PIREUS_MBC_RESULT valid=1 error=0'
require_line "${ROOT}/${TEST_EVIDENCE_REL}" \
  'PIREUS_MULTIPROBE_BLOCK_CERTIFICATION_EXECUTABLE_OK'

[[ "$(wc -l < "${ROOT}/${FIRST_TRANSCRIPT_REL}")" -eq 9 ]] ||
  fail 'first transcript line count drift'
[[ "$(wc -l < "${ROOT}/${TEST_EVIDENCE_REL}")" -eq 1 ]] ||
  fail 'structural test evidence line count drift'
require_line "${ROOT}/${FIRST_TRANSCRIPT_REL}" \
  'PIREUS_MBC_LINEAGE garden_match=1 contract_match=1 parent_transcript_match=1'
require_line "${ROOT}/${FIRST_TRANSCRIPT_REL}" \
  'PIREUS_MBC_PARENT probes=32 canonical_cells=8192 duplicate_contents=0 parse_failures=0'
require_line "${ROOT}/${FIRST_TRANSCRIPT_REL}" \
  'PIREUS_MBC_PARTITION blocks=64 views=40320 failures=0 first_start=0 first_end=630 last_start=39690 last_end=40320'
NEGATIVE_LINE="$(grep '^PIREUS_MBC_NEGATIVES ' "${ROOT}/${FIRST_TRANSCRIPT_REL}")"
[[ "${NEGATIVE_LINE}" =~ ^PIREUS_MBC_NEGATIVES\ passed=([0-9]+)\ total=([0-9]+)$ ]] ||
  fail 'negative census shape drift'
[[ "${BASH_REMATCH[1]}" -gt 0 && "${BASH_REMATCH[1]}" == "${BASH_REMATCH[2]}" ]] ||
  fail 'negative census did not close'
grep -Eq '^PIREUS_MBC_DIGEST strategy=([0-9]+:){7}[0-9]+ probe_set=([0-9]+:){7}[0-9]+ partition=([0-9]+:){7}[0-9]+ plan=([0-9]+:){7}[0-9]+ first_address=([0-9]+:){7}[0-9]+ last_address=([0-9]+:){7}[0-9]+$' \
  "${ROOT}/${FIRST_TRANSCRIPT_REL}" || fail 'digest record shape drift'

printf 'PIREUS_MBC_FIRST_PLAN_GATE_PASS=true stage=SOUNIO_EXECUTABLE actual_blocks=0 reused=0 certified=0 claim_ready=false main_stdout_sha256=%s test_stdout_sha256=%s toolchain_sha256=%s hardware_sha256=%s python_dispatch=E110 rust_dispatch=E110 parity_prefreeze=E112 raw_elf_command_requested=false\n' \
  "$(sha_file "${ROOT}/${FIRST_TRANSCRIPT_REL}")" \
  "$(sha_file "${ROOT}/${TEST_EVIDENCE_REL}")" \
  "${TOOLCHAIN_SHA256}" "${HARDWARE_SHA256}"
