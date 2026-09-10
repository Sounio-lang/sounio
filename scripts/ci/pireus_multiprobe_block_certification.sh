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
FROZEN_MODULE_REL='stdlib/hardware/pireus/multiprobe_block_certification_frozen.sio'
FROZEN_REPLAY_REL='examples/pireus_multiprobe_block_certification_frozen_replay.sio'
FROZEN_TEST_REL='tests/stdlib/hardware/test_pireus_multiprobe_block_certification_frozen.sio'
PARENT_TRANSCRIPT_REL='tools/pireus/operator_orbit_canonicalization.first.v13'
FIRST_TRANSCRIPT_REL='tools/pireus/multiprobe_block_certification.first.v14'
FIRST_RECEIPT_REL='tools/pireus/evidence/multiprobe_block_certification_v14.first-execution.txt'
FIRST_TEST_REL='tools/pireus/evidence/multiprobe_block_certification_v14.test.txt'
SEMANTICS_CANDIDATE_REL='tools/pireus/multiprobe_block_certification.semantics-candidate.v14'
FROZEN_EVIDENCE_REL='tools/pireus/evidence/multiprobe_block_certification_v14.frozen.txt'
FROZEN_TEST_EVIDENCE_REL='tools/pireus/evidence/multiprobe_block_certification_v14.frozen-test.txt'

GARDEN_COMMIT='db5a1ab31fb616e846a757bd52dd201a0cf5d703'
EXECUTABLE_COMMIT='2b5d42a9db2b84307784f48696fd5c27756ce49c'
FIRST_EVIDENCE_COMMIT='2c03d8259d5a0d3004e54dc8549ad8b7d4a411ee'
MATCHER_COMMIT='62b8fb8352d1b2ae377860176b7bdb59ae63280e'
CANDIDATE_COMMIT='c31b7824c9681c1c028d7f3a96068d81ea665e7b'
PARENT_EVIDENCE_COMMIT='22fbabe81cf365c0b542d8a425ec4c081f31e390'
POLICY_COMMIT='f3a4128388d47e091e9803d67c097a6976efeb02'
PRESEAL_GATE_COMMIT='09f97afa52e224e5e94be373151568cf8888539d'

GARDEN_SHA256='2bc8b791549a30c041d07801c712b8a859bbb9fd5933177802fa9eda3bcd2a9c'
CONTRACT_SHA256='1f7fb55f32b413683ce45b4bdfebc686600f14b4346ca780e29ec8bbf91c8a48'
MODULE_SHA256='d6d38280a2934956ce1721775c978911e1cce9c526c05fb11c75c7f00ed2ddac'
EXAMPLE_SHA256='71b9157c6867f1fe33f0d1a1709ef1412f3ee924937ff6501854490428988ca5'
TEST_SHA256='58974ff011fc680481d21c31cb299f20a2c24105bf07d3571ffc14b82dadc4ea'
FROZEN_MODULE_SHA256='754d1c6b10be83faa220800e1158e884c49f23b0fcbe5451faf01ec7d6783c00'
FROZEN_REPLAY_SHA256='308b9a70e0ee321a637b3832b9edb0c2e040e5c7ba7385153bbf6a3b68b8fb44'
FROZEN_TEST_SHA256='4dd6d675161091e6bc731fcc30b1108105bbe9209bda9f332c15f4bba3678d88'
PARENT_TRANSCRIPT_SHA256='16af63f5e0f8aa7e5c899f4c395404b83fb402f6bbdb5f20dea2a3d10ad2e19f'
FIRST_TRANSCRIPT_SHA256='46f50ae65d018639b1aab0f598a175a2d61c3ea0ce4368da958796e7d49d8e93'
FIRST_RECEIPT_SHA256='a7478e7cc011c32dd06e229aff9eaa4f38a9f66d8bfc99725fde1cef7153836d'
FIRST_TEST_SHA256='0d430af317b34703ceee3b46afdb59db2ba38b41281250dda9656db04fcb80da'
SEMANTICS_CANDIDATE_SHA256='55c00352d60130097e24d80ce085515fffda34bcca52903943f33fb041ac4924'
SOURCE_MANIFEST_SHA256='162beb0a344715c5674e33fb110dad48910a729c58f5756e79c6e892d3dcf768'
SEMANTICS_SHA256='f7b7e81c546bf54a2a92ec374f50465dfb2e0874d52b77fc6ac70484587dc20c'
FROZEN_EVIDENCE_SHA256='12965d1402380c938efd564fc75685be13cc2e3564b9ceb12eb948ff19d232d8'
FROZEN_TEST_EVIDENCE_SHA256='9e6144c2bed92643c2f8b59798cb7eb00acfb9d7d83a9cf7037b1268f768d8a5'
PRESEAL_GATE_SHA256='d5a38ef510816fcfe61067ad886a3187a09fcf21f8bdc478ed2406fe738edcfa'
WRAPPER_SHA256='ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008'
RESOLVER_SHA256='a7e37545490745a58731933b0e07db69843cfeea30e739ed554b82d099ec3d84'
COMPILER_SHA256='6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
GUARDIAN_POLICY_SHA256='64bb0118793fe46dcb392abc1a9212eb15bd55047461576a3ef1a6cefa3f17da'
PARENT_SEMANTICS_SHA256='0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c'

FROZEN_COMMAND="SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run ${FROZEN_REPLAY_REL} > ${FROZEN_EVIDENCE_REL}"
FROZEN_TEST_COMMAND="SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run ${FROZEN_TEST_REL} > ${FROZEN_TEST_EVIDENCE_REL}"
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
  local command_hash="$6" result_hash="$7"
  printf '9020 %s %s %s %s %s 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${stage}" "${action}" "${language}" "${role}" "${policy}" \
    "$(sha_limbs "${SOURCE_MANIFEST_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PARENT_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${command_hash}")" \
    "$(sha_limbs "${result_hash}")" "${ZERO}"
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
    printf 'GUARDIAN_DECISION label=%s frame_sha256=%s decision=%s process_launch_authorized=true process_launched=false\n' \
      "${label}" "${frame_sha}" "${decision}"
  else
    printf 'GUARDIAN_DECISION label=%s frame_sha256=%s decision=%s process_launch_authorized=false process_launched=false\n' \
      "${label}" "${frame_sha}" "${decision}"
  fi
}

[[ -x "${GUARDIAN}" ]] || fail 'native Loom Guardian unavailable'
require_hash "${ROOT}/${GARDEN_REL}" "${GARDEN_SHA256}"
require_hash "${ROOT}/${CONTRACT_REL}" "${CONTRACT_SHA256}"
require_hash "${ROOT}/${MODULE_REL}" "${MODULE_SHA256}"
require_hash "${ROOT}/${EXAMPLE_REL}" "${EXAMPLE_SHA256}"
require_hash "${ROOT}/${TEST_REL}" "${TEST_SHA256}"
require_hash "${ROOT}/${FROZEN_MODULE_REL}" "${FROZEN_MODULE_SHA256}"
require_hash "${ROOT}/${FROZEN_REPLAY_REL}" "${FROZEN_REPLAY_SHA256}"
require_hash "${ROOT}/${FROZEN_TEST_REL}" "${FROZEN_TEST_SHA256}"
require_hash "${ROOT}/${PARENT_TRANSCRIPT_REL}" "${PARENT_TRANSCRIPT_SHA256}"
require_hash "${ROOT}/${FIRST_TRANSCRIPT_REL}" "${FIRST_TRANSCRIPT_SHA256}"
require_hash "${ROOT}/${FIRST_RECEIPT_REL}" "${FIRST_RECEIPT_SHA256}"
require_hash "${ROOT}/${FIRST_TEST_REL}" "${FIRST_TEST_SHA256}"
require_hash "${ROOT}/${SEMANTICS_CANDIDATE_REL}" "${SEMANTICS_CANDIDATE_SHA256}"
require_hash "${ROOT}/bin/souc" "${WRAPPER_SHA256}"
require_hash "${ROOT}/scripts/lib/resolve_souc.sh" "${RESOLVER_SHA256}"
require_hash "${ROOT}/bin/souc-lean-single-x86_64" "${COMPILER_SHA256}"
require_hash "${GUARDIAN}" "${GUARDIAN_SHA256}"
require_hash "${ROOT}/stdlib/coordination/loom_language_authority.sio" \
  "${GUARDIAN_POLICY_SHA256}"

git -C "${ROOT}" merge-base --is-ancestor "${GARDEN_COMMIT}" \
  "${EXECUTABLE_COMMIT}" || fail 'Garden does not precede executable'
git -C "${ROOT}" merge-base --is-ancestor "${EXECUTABLE_COMMIT}" \
  "${FIRST_EVIDENCE_COMMIT}" || fail 'executable does not precede first evidence'
git -C "${ROOT}" merge-base --is-ancestor "${FIRST_EVIDENCE_COMMIT}" \
  "${MATCHER_COMMIT}" || fail 'first evidence does not precede matcher'
git -C "${ROOT}" merge-base --is-ancestor "${MATCHER_COMMIT}" \
  "${CANDIDATE_COMMIT}" || fail 'matcher does not precede semantics candidate'
git -C "${ROOT}" merge-base --is-ancestor "${CANDIDATE_COMMIT}" HEAD ||
  fail 'semantics candidate is not an ancestor of HEAD'
git -C "${ROOT}" merge-base --is-ancestor "${PARENT_EVIDENCE_COMMIT}" \
  "${EXECUTABLE_COMMIT}" || fail 'V13 parent evidence does not precede executable'
git -C "${ROOT}" merge-base --is-ancestor "${POLICY_COMMIT}" \
  "${EXECUTABLE_COMMIT}" || fail 'Guardian policy does not precede executable'

require_committed_hash "${GARDEN_COMMIT}" "${GARDEN_REL}" "${GARDEN_SHA256}"
require_committed_hash "${GARDEN_COMMIT}" "${CONTRACT_REL}" "${CONTRACT_SHA256}"
require_committed_hash "${EXECUTABLE_COMMIT}" "${MODULE_REL}" "${MODULE_SHA256}"
require_committed_hash "${EXECUTABLE_COMMIT}" "${EXAMPLE_REL}" "${EXAMPLE_SHA256}"
require_committed_hash "${EXECUTABLE_COMMIT}" "${TEST_REL}" "${TEST_SHA256}"
require_committed_hash "${EXECUTABLE_COMMIT}" 'bin/souc' "${WRAPPER_SHA256}"
require_committed_hash "${EXECUTABLE_COMMIT}" 'scripts/lib/resolve_souc.sh' "${RESOLVER_SHA256}"
require_committed_hash "${EXECUTABLE_COMMIT}" 'bin/souc-lean-single-x86_64' "${COMPILER_SHA256}"
require_committed_hash "${FIRST_EVIDENCE_COMMIT}" "${FIRST_TRANSCRIPT_REL}" "${FIRST_TRANSCRIPT_SHA256}"
require_committed_hash "${FIRST_EVIDENCE_COMMIT}" "${FIRST_RECEIPT_REL}" "${FIRST_RECEIPT_SHA256}"
require_committed_hash "${FIRST_EVIDENCE_COMMIT}" "${FIRST_TEST_REL}" "${FIRST_TEST_SHA256}"
require_committed_hash "${MATCHER_COMMIT}" "${FROZEN_MODULE_REL}" "${FROZEN_MODULE_SHA256}"
require_committed_hash "${MATCHER_COMMIT}" "${FROZEN_REPLAY_REL}" "${FROZEN_REPLAY_SHA256}"
require_committed_hash "${MATCHER_COMMIT}" "${FROZEN_TEST_REL}" "${FROZEN_TEST_SHA256}"
require_committed_hash "${CANDIDATE_COMMIT}" "${SEMANTICS_CANDIDATE_REL}" "${SEMANTICS_CANDIDATE_SHA256}"
require_committed_hash "${PARENT_EVIDENCE_COMMIT}" "${PARENT_TRANSCRIPT_REL}" "${PARENT_TRANSCRIPT_SHA256}"
require_committed_hash "${POLICY_COMMIT}" 'stdlib/coordination/loom_language_authority.sio' "${GUARDIAN_POLICY_SHA256}"
require_committed_hash "${PRESEAL_GATE_COMMIT}" 'scripts/ci/pireus_multiprobe_block_certification.sh' "${PRESEAL_GATE_SHA256}"

if git -C "${ROOT}" grep -q -E \
  'matches_frozen|frozen_mismatch|expected_(strategy|probe|partition|plan|address)' \
  "${EXECUTABLE_COMMIT}" -- "${MODULE_REL}" "${EXAMPLE_REL}" "${TEST_REL}"; then
  fail 'V14 matcher leaked into matcher-free executable commit'
fi
MATCHER_DELTA="$(git -C "${ROOT}" diff-tree --no-commit-id --name-only -r "${MATCHER_COMMIT}" | LC_ALL=C sort)"
[[ "${MATCHER_DELTA}" == $'.claude/llm_offload_log.md\nexamples/pireus_multiprobe_block_certification_frozen_replay.sio\nstdlib/hardware/pireus/multiprobe_block_certification_frozen.sio\ntests/stdlib/hardware/test_pireus_multiprobe_block_certification_frozen.sio' ]] ||
  fail 'matcher commit escaped its exact path allowlist'

[[ "$(sed -n '/^source_manifest_begin$/,/^source_manifest_end$/p' "${ROOT}/${SEMANTICS_CANDIDATE_REL}" | sed '1d;$d' | sha256sum | cut -d' ' -f1)" == "${SOURCE_MANIFEST_SHA256}" ]] ||
  fail 'source manifest digest drift'
[[ "$(sed -n '/^semantics_material_begin$/,/^semantics_material_end$/p' "${ROOT}/${SEMANTICS_CANDIDATE_REL}" | sed '1d;$d' | sha256sum | cut -d' ' -f1)" == "${SEMANTICS_SHA256}" ]] ||
  fail 'semantics material digest drift'
require_line "${ROOT}/${SEMANTICS_CANDIDATE_REL}" 'status=FREEZE_CANDIDATE'
require_line "${ROOT}/${SEMANTICS_CANDIDATE_REL}" 'freeze_replay_required=true'
require_line "${ROOT}/${SEMANTICS_CANDIDATE_REL}" 'expected_result_digests=0'
require_line "${ROOT}/${SEMANTICS_CANDIDATE_REL}" 'actual_block_execution_complete=false'
require_line "${ROOT}/${SEMANTICS_CANDIDATE_REL}" 'any_block_reused=false'
require_line "${ROOT}/${SEMANTICS_CANDIDATE_REL}" 'all_v13_probes_certified=false'
require_line "${ROOT}/${SEMANTICS_CANDIDATE_REL}" 'claim_ready=false'

[[ "$(wc -l < "${ROOT}/${FIRST_TRANSCRIPT_REL}")" -eq 76 ]] ||
  fail 'first transcript line count drift'
require_line "${ROOT}/${FIRST_TRANSCRIPT_REL}" \
  'SOUNIO_AUTHORITY schema=pireus-multiprobe-block-certification.v14 role=SEMANTIC_AUTHORITY stage=SOUNIO_EXECUTABLE matcher_free=1 expected_results=0'
require_line "${ROOT}/${FIRST_TRANSCRIPT_REL}" 'PIREUS_MBC_LINEAGE garden_match=1'
require_line "${ROOT}/${FIRST_TRANSCRIPT_REL}" ' contract_match=1'
require_line "${ROOT}/${FIRST_TRANSCRIPT_REL}" ' parent_transcript_match=1'
require_line "${ROOT}/${FIRST_TRANSCRIPT_REL}" 'PIREUS_MBC_PLAN work_domains=2048'
require_line "${ROOT}/${FIRST_TRANSCRIPT_REL}" ' address_recomputations=2048'
require_line "${ROOT}/${FIRST_TRANSCRIPT_REL}" ' address_recomputation_failures=0'
require_line "${ROOT}/${FIRST_TRANSCRIPT_REL}" ' completed_blocks=0 reuse_hits=0 expected_results=0'
require_line "${ROOT}/${FIRST_TRANSCRIPT_REL}" 'PIREUS_MBC_NEGATIVES passed=56'
require_line "${ROOT}/${FIRST_TRANSCRIPT_REL}" ' total=56'
require_line "${ROOT}/${FIRST_TRANSCRIPT_REL}" \
  'PIREUS_MBC_BOUNDARY actual_blocks=0 reused=0 certified=0 purity_proved=0 material_bound_profile=0 collision_free=0 formal_parity=0 effect_parity=0 material_parity=0 denotational_parity=0 performance=0 speedup=0 cache_hit_rate=0 algorithmic=0 material_novelty=0 scientific=0 global=0 historical=0 priority=0 claim_ready=0'
require_line "${ROOT}/${FIRST_TRANSCRIPT_REL}" 'PIREUS_MBC_RESULT valid=1'
require_line "${ROOT}/${FIRST_TRANSCRIPT_REL}" ' error=0'
require_line "${ROOT}/${FIRST_RECEIPT_REL}" 'gate_pass=false'
require_line "${ROOT}/${FIRST_RECEIPT_REL}" 'gate_failure_class=HARNESS_OUTPUT_LAYOUT'
require_line "${ROOT}/${FIRST_RECEIPT_REL}" 'semantics_frozen=false'
require_line "${ROOT}/${FIRST_RECEIPT_REL}" 'first_result_overwritten=false'

[[ ! -e "${ROOT}/${FROZEN_EVIDENCE_REL}" ]] ||
  fail "frozen replay evidence already exists: ${FROZEN_EVIDENCE_REL}"
[[ ! -e "${ROOT}/${FROZEN_TEST_EVIDENCE_REL}" ]] ||
  fail "frozen test evidence already exists: ${FROZEN_TEST_EVIDENCE_REL}"
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

FROZEN_COMMAND_SHA256="$(sha_text "${FROZEN_COMMAND}")"
FROZEN_TEST_COMMAND_SHA256="$(sha_text "${FROZEN_TEST_COMMAND}")"
PYTHON_COMMAND_SHA256="$(sha_text "${PYTHON_COMMAND}")"
RUST_COMMAND_SHA256="$(sha_text "${RUST_COMMAND}")"
PARITY_COMMAND_SHA256="$(sha_text "${PARITY_COMMAND}")"

check_guardian POLICY_MISSING \
  "$(authority_frame 2 3 1 1 0 "${FROZEN_COMMAND_SHA256}" "${FROZEN_EVIDENCE_SHA256}")" 101 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=101 reason=policy-missing next_stage=SOUNIO_EXECUTABLE'
check_guardian POLICY_TIMEOUT \
  "$(authority_frame 2 3 1 1 2 "${FROZEN_COMMAND_SHA256}" "${FROZEN_EVIDENCE_SHA256}")" 102 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=102 reason=policy-timeout next_stage=SOUNIO_EXECUTABLE'
check_guardian POLICY_ERROR \
  "$(authority_frame 2 3 1 1 3 "${FROZEN_COMMAND_SHA256}" "${FROZEN_EVIDENCE_SHA256}")" 103 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=103 reason=policy-error next_stage=SOUNIO_EXECUTABLE'
check_guardian PYTHON_ORACLE \
  "$(authority_frame 2 3 7 7 1 "${PYTHON_COMMAND_SHA256}" "${FROZEN_EVIDENCE_SHA256}")" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SOUNIO_EXECUTABLE'
check_guardian RUST_ORACLE \
  "$(authority_frame 2 3 8 7 1 "${RUST_COMMAND_SHA256}" "${FROZEN_EVIDENCE_SHA256}")" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SOUNIO_EXECUTABLE'
check_guardian PARITY_PREFREEZE \
  "$(authority_frame 2 4 2 2 1 "${PARITY_COMMAND_SHA256}" "${FROZEN_EVIDENCE_SHA256}")" 112 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=112 reason=wrong-stage next_stage=SOUNIO_EXECUTABLE'
check_guardian SOUNIO_FROZEN_REPLAY \
  "$(authority_frame 2 3 1 1 1 "${FROZEN_COMMAND_SHA256}" "${FROZEN_EVIDENCE_SHA256}")" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'

(
  cd "${ROOT}"
  bash -c "${FROZEN_COMMAND}"
)
printf 'PROCESS_RESULT label=SOUNIO_FROZEN_REPLAY process_launch_authorized=true process_launched=true exit_code=0\n'

check_guardian SOUNIO_FROZEN_MATCHER_TEST \
  "$(authority_frame 2 3 1 1 1 "${FROZEN_TEST_COMMAND_SHA256}" "${FROZEN_TEST_EVIDENCE_SHA256}")" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'

(
  cd "${ROOT}"
  bash -c "${FROZEN_TEST_COMMAND}"
)
printf 'PROCESS_RESULT label=SOUNIO_FROZEN_MATCHER_TEST process_launch_authorized=true process_launched=true exit_code=0\n'

[[ ! -e /tmp/pireus-mbc-python-oracle ]] ||
  fail 'forbidden Python oracle process created a marker'
require_hash "${ROOT}/${FROZEN_EVIDENCE_REL}" "${FROZEN_EVIDENCE_SHA256}"
require_hash "${ROOT}/${FROZEN_TEST_EVIDENCE_REL}" "${FROZEN_TEST_EVIDENCE_SHA256}"
[[ "$(wc -l < "${ROOT}/${FROZEN_EVIDENCE_REL}")" -eq 2 ]] ||
  fail 'frozen replay line count drift'
[[ "$(wc -l < "${ROOT}/${FROZEN_TEST_EVIDENCE_REL}")" -eq 1 ]] ||
  fail 'frozen matcher test line count drift'
require_line "${ROOT}/${FROZEN_EVIDENCE_REL}" \
  'SOUNIO_AUTHORITY schema=pireus-multiprobe-block-certification.v14 role=SEMANTIC_AUTHORITY stage=SEMANTICS_FROZEN matcher_free=0 observed_plan_matchers=6 expected_block_results=0'
require_line "${ROOT}/${FROZEN_EVIDENCE_REL}" \
  'PIREUS_MBC_FROZEN_REPLAY match=1 mismatch=0 completed_blocks=0 reuse_hits=0 certified=0 claim_ready=0'
require_line "${ROOT}/${FROZEN_TEST_EVIDENCE_REL}" \
  'PIREUS_MULTIPROBE_BLOCK_CERTIFICATION_FROZEN_MATCHER_OK'

printf 'PIREUS_MBC_FREEZE_REPLAY_GATE_PASS=true next_stage=SEMANTICS_FROZEN semantics_sha256=%s source_manifest_sha256=%s frozen_stdout_sha256=%s frozen_test_sha256=%s toolchain_sha256=%s hardware_sha256=%s actual_blocks=0 reused=0 certified=0 claim_ready=false python_dispatch=E110 rust_dispatch=E110 parity_prefreeze=E112 raw_elf_command_requested=false\n' \
  "${SEMANTICS_SHA256}" "${SOURCE_MANIFEST_SHA256}" \
  "${FROZEN_EVIDENCE_SHA256}" "${FROZEN_TEST_EVIDENCE_SHA256}" \
  "${TOOLCHAIN_SHA256}" "${HARDWARE_SHA256}"
