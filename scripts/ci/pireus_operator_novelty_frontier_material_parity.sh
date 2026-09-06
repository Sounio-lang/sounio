#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
BUILD_ROOT='/workspace/.home/openvscode-server/.cache/pireus/material-v11/gate'
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/pireus-novelty-material-v11.XXXXXX")"
trap 'rm -rf "${TMP_ROOT}"' EXIT

SOUNIO_REL='stdlib/hardware/pireus/operator_novelty_frontier.sio'
FREEZE_REL='tools/pireus/operator_novelty_frontier.freeze.v11'
OPEN_REL='tools/pireus/operator_novelty_frontier.parity-open.v11'
FORMAL_REL='tools/pireus/operator_novelty_frontier.formal-parity.v11'
FORMAL_GATE_REL='scripts/ci/pireus_operator_novelty_frontier_formal_parity.sh'
EFFECT_REL='tools/pireus/operator_novelty_frontier.effect-parity.v11'
EFFECT_GATE_REL='scripts/ci/pireus_operator_novelty_frontier_effect_parity.sh'
CPP_REL='tools/pireus/operator_novelty_frontier_material_parity.cpp'
XEON_REL='tools/pireus/evidence/operator_novelty_frontier_v11.material.xeon.txt'
APPLE_REL='tools/pireus/evidence/operator_novelty_frontier_v11.material.apple.txt'
DGX24_REL='tools/pireus/evidence/operator_novelty_frontier_v11.material.dgx24.txt'
DGX48_REL='tools/pireus/evidence/operator_novelty_frontier_v11.material.dgx48.txt'
U250_REL='tools/pireus/evidence/operator_novelty_frontier_v11.material.u250.txt'
RECEIPT_REL='tools/pireus/operator_novelty_frontier.material-parity.v11'

SOUNIO_SHA256='9289cd504385e2f1f4eed095d82a963cf2e5e67124bf8d267d1bc6ccda7ac36b'
SEMANTICS_SHA256='f1e339ec7bc290f412d42bba3fa1ba609fd89947408ea422ab96026cce5883dc'
FREEZE_SHA256='b57decc8ff929640345e47edc931bdfa6cd06c738d3ff9591d3a460593dae242'
OPEN_SHA256='f7cde0ed063d136bbef43cf9e820d734341f87717bb26e130a3643bc62fb31de'
FORMAL_SHA256='b56b1f331879c2a8bbb70dc0adfc5ac61e21e922834c391ce4d815397a589d21'
FORMAL_GATE_SHA256='09f4e776c44875af757314a64d44dba3c2245cc76f6e8a7eb198d4d4d8e023e3'
EFFECT_SHA256='b18f91987a5b169bebb1a02d3b200f4ecae513c28f83f16dabaf3a96f2524d71'
EFFECT_GATE_SHA256='fd1bc5834317f1dc077d5d3ce1c4071fb03059a79e6301e029bd28cc0b9f82f8'
CPP_SHA256='1f0bab7a936eecd60424175f8442494db6b1427dde30c46609a8b81d264345ca'
XEON_SHA256='bd21b9ce695766b4899f2e89e68536c3d4fec4f084e12f213294f47e6b9f2d5a'
APPLE_SHA256='6d48ae6e2995a1c7780b790a436c9e359afbb6d48b4ebe42c3bfed5218e40dd0'
DGX24_SHA256='bc55af6b8f0dbc80cdb0f20886b175cd48027139d08a4a721610efbbdfd7535d'
DGX48_SHA256='dd9629170d7fa1952304d007f77cdd9698865e09396b7a407766408256b39628'
U250_SHA256='b65f3e805ae793d5fa51ee7b9252f7eafcb3d6615926e3993aa509619b2561ab'
RECEIPT_SHA256='95c0a63f8833abd8636d4dd6d43097c5172e15b3992a3459a1f78c8dcbf198ad'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
GUARDIAN_POLICY_SHA256='64bb0118793fe46dcb392abc1a9212eb15bd55047461576a3ef1a6cefa3f17da'
FORMAL_COMMIT='02dd802f0bc4c0af816029b08869acc08b233879'
EFFECT_COMMIT='3358d0c00969a7391292ec5e5fbd6da0400822c6'

TOOLCHAIN_SHA256='febb0e137568fd21a7aa551a2949400b4c1b2bad12cf988044f2743dfae2478e'
HARDWARE_SHA256='f76a198d5b8ec270f694b8844ff39e0564bbc42aeb80b5e3eaa4d16b66560740'
COMMAND_SHA256='5ab27d3fde3b6604c8ebe2f0741d17dc272824da058fcf8f0ea8c09054d6c989'
BINARY_SHA256='40351e808d97243ea1f1cf2f7444644c8f5c4acaa3d5e174d949bbed99ab4ac1'
PREEXEC_FRAME_SHA256='bfa0b204b6e2ab7e54c3aee556e2187349389752e2a261d5eec3bad06c03a2fc'
SEAL_FRAME_SHA256='915b000ed966f09859d1893ee77990f1eed6d2d65e37b7c5145d51ec356e3db7'
WRITE_FRAME_SHA256='432ead89430717a3ba702cf3097937fe51a3782ee5e2d5718677248e84f740dd'
COMMIT_FRAME_SHA256='12845146dcbbe6863b168fdac49458ede3637a63125abbb6317b00968708adad'
SEMANTIC_WRITE_FRAME_SHA256='56b19b48aeefd79adc3190095d2e55d115fe9b2124140e475cdee1c80135c3d3'
POLICY_MISSING_FRAME_SHA256='e413d709be234801be49533cbff8e8f687851606df66b2e91dbbe4e2be5ffd84'
POLICY_TIMEOUT_FRAME_SHA256='835339e8a2f1cda824d148a2032917eb44a2cf9d38003ea76d6ac8e27694cdde'
POLICY_ERROR_FRAME_SHA256='0b1132ded29e9a44d95ffe78313a23db60707cc96c52b451d3019ec00265445b'
PYTHON_FRAME_SHA256='19713ccc4e9ba8bf7b6771a1f8c9ca81355d36acaba9a3dad1fb2bbf07fff7ec'
RUST_FRAME_SHA256='295c083f334724bd3e688780992946e66c0574c994936d711976f547e7f6fddc'
CLAIM_FRAME_SHA256='151d8d0d8305db67c47f3198fbe9fe229532a81683c461d301eaecc8c02e71f1'
PYTHON_TOOLCHAIN_SHA256='497ce0938df96d9bf3c159472d251a946c2d6bd832220937d1b885f7759b05ba'
PYTHON_COMMAND_SHA256='72566473f0019fb50e65175a0ac019af5ad2f495ccfc30b664b7735791e968fc'
RUST_TOOLCHAIN_SHA256='b58640570ba9ffcdb2c2d241e4ce8ece9c7d75c6b1e59e308dee3e5f0e10b56d'
RUST_COMMAND_SHA256='0421c7020d972449b7a1f0492945c18a10be46f6a16ade7f8efad157e2e01b00'
ZERO='0 0 0 0 0 0 0 0'
FRAME_WORDS=82
REPLAY_COMMAND='g++ -std=c++20 -O2 -Wall -Wextra -Werror tools/pireus/operator_novelty_frontier_material_parity.cpp -ldl -o /workspace/.home/openvscode-server/.cache/pireus/material-v11/gate/operator_novelty_frontier_material_parity && /workspace/.home/openvscode-server/.cache/pireus/material-v11/gate/operator_novelty_frontier_material_parity --target=xeon > /workspace/.home/openvscode-server/.cache/pireus/material-v11/gate/xeon.txt'
REPLAY_BINARY="${BUILD_ROOT}/operator_novelty_frontier_material_parity"
REPLAY_OUTPUT="${BUILD_ROOT}/xeon.txt"

fail() {
  printf 'pireus operator novelty frontier material parity: FAIL: %s\n' "$*" >&2
  exit 1
}

sha_file() { sha256sum "$1" | cut -d' ' -f1; }
sha_text() { printf '%s\n' "$1" | sha256sum | cut -d' ' -f1; }

sha_limbs() {
  local hex="$1" out='' i part
  [[ "${#hex}" -eq 64 && "${hex}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid sha256: ${hex}"
  for ((i = 0; i < 8; i++)); do
    part="${hex:$((i * 8)):8}"
    out="${out}${out:+ }$((16#${part}))"
  done
  printf '%s' "${out}"
}

require_hash() {
  local path="$1" expected="$2"
  [[ -f "${path}" ]] || fail "missing file: ${path}"
  [[ "$(sha_file "${path}")" == "${expected}" ]] || fail "hash drift: ${path}"
}

require_line() {
  grep -Fqx -- "$2" "$1" || fail "missing exact line in $1: $2"
}

authority_frame() {
  local stage="$1" action="$2" language="$3" role="$4" policy="$5"
  local semantic_write="$6" expected_write="$7" parity_valid="$8"
  local review_promoted="$9" toolchain_hash="${10}" command_hash="${11}"
  local result_hash="${12}" result_limbs="${ZERO}"
  [[ "${result_hash}" == zero ]] || result_limbs="$(sha_limbs "${result_hash}")"
  printf '9020 %s %s %s %s %s %s %s %s %s 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${stage}" "${action}" "${language}" "${role}" "${policy}" \
    "${semantic_write}" "${expected_write}" "${parity_valid}" "${review_promoted}" \
    "$(sha_limbs "${CPP_SHA256}")" "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" "$(sha_limbs "${toolchain_hash}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" "$(sha_limbs "${command_hash}")" \
    "${result_limbs}" "${ZERO}"
}

check_guardian() {
  local label="$1" frame="$2" expected_hash="$3" expected_rc="$4" expected="$5"
  local decision rc
  [[ "$(wc -w <<<"${frame}" | tr -d ' ')" -eq "${FRAME_WORDS}" ]] ||
    fail "${label}: frame word count drift"
  [[ "$(sha_text "${frame}")" == "${expected_hash}" ]] || fail "${label}: frame hash drift"
  set +e
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  rc=$?
  set -e
  [[ "${rc}" -eq "${expected_rc}" ]] || fail "${label}: rc=${rc}"
  [[ "${decision}" == "${expected}" ]] || fail "${label}: ${decision}"
  printf 'GUARDIAN_DECISION label=%s frame_sha256=%s rc=%s %s\n' \
    "${label}" "${expected_hash}" "${rc}" "${decision}"
  if [[ "${expected_rc}" -ne 0 ]]; then
    printf 'GUARDIAN_DISPATCH label=%s process_launched=false\n' "${label}"
  fi
}

validate_evidence() {
  local path="$1" target="$2" target_name="$3"
  require_line "${path}" 'schema=pireus-operator-novelty-frontier-material-parity-v11'
  require_line "${path}" 'producing_language=C++'
  require_line "${path}" 'producing_role=MATERIAL_PARITY'
  require_line "${path}" 'authority_language=Sounio'
  require_line "${path}" "sounio_source_sha256=${SOUNIO_SHA256}"
  require_line "${path}" "sounio_semantics_sha256=${SEMANTICS_SHA256}"
  require_line "${path}" "formal_parity_receipt_sha256=${FORMAL_SHA256}"
  require_line "${path}" "effect_parity_receipt_sha256=${EFFECT_SHA256}"
  require_line "${path}" "target=${target}"
  require_line "${path}" "target_name=${target_name}"
  require_line "${path}" 'grammar_candidates_declared=7200'
  require_line "${path}" 'grammar_candidates_evaluated_by_cpp=7200'
  require_line "${path}" 'grammar_enumeration_performed_by_cpp=true'
  require_line "${path}" 'codec_checks=14400'
  require_line "${path}" 'codec_failures=0'
  require_line "${path}" 'one_sparse_candidates=7200'
  require_line "${path}" 'atlas_representative_supports=0:176:512:474:96:272'
  require_line "${path}" 'atlas_collision_candidates=0'
  require_line "${path}" 'atlas_collision_edges=0'
  require_line "${path}" 'n2_relative_novelty=7200'
  require_line "${path}" 'separators=43200'
  require_line "${path}" 'separator_formula_checks=43200'
  require_line "${path}" 'separator_failures=0'
  require_line "${path}" 'c2_character_checks=4096'
  require_line "${path}" 'c2_failures=0'
  require_line "${path}" 'action_base_support=176'
  require_line "${path}" 'transported_mutation_inside_grammar=6272'
  require_line "${path}" 'transported_mutation_outside_grammar=928'
  require_line "${path}" 'transported_mutation_over_base_difference_support=228'
  require_line "${path}" 'transported_mutation_outside_base_difference_support=6972'
  require_line "${path}" 'quotient_outside=7200'
  require_line "${path}" 'quotient_fixed=0'
  require_line "${path}" 'quotient_pairs=0'
  require_line "${path}" 'quotient_singletons=7200'
  require_line "${path}" 'quotient_classes=7200'
  require_line "${path}" 'quotient_in_grammar_images=0'
  require_line "${path}" 'direct_quotient_checks=7200'
  require_line "${path}" 'quotient_failures=0'
  require_line "${path}" 'material_reconstruction_match=true'
  require_line "${path}" 'target_identity_observed=true'
  require_line "${path}" 'numeric_values_status=EXHAUSTIVE_CPP_RECONSTRUCTION_OF_FROZEN_SOUNIO_VALUES'
  require_line "${path}" 'analytic_proof_by_cpp=false'
  require_line "${path}" 'digest_parity_performed_by_cpp=false'
  require_line "${path}" 'native_gpu_operator_kernel_execution=false'
  require_line "${path}" 'fpga_operator_kernel_execution=false'
  require_line "${path}" 'semantic_write=false'
  require_line "${path}" 'expected_result_write=false'
  require_line "${path}" 'candidate_selected_by_cpp=false'
  require_line "${path}" 'candidate_selected=false'
  require_line "${path}" 'n3_novelty=false'
  require_line "${path}" 'n4_novelty=false'
  require_line "${path}" 'algorithmic_novelty=false'
  require_line "${path}" 'material_novelty=false'
  require_line "${path}" 'scientific_novelty=false'
  require_line "${path}" 'historical_novelty=false'
  require_line "${path}" 'priority_claim=false'
  require_line "${path}" 'claim_ready=false'
  require_line "${path}" 'result=PASS'
}

receipt_admitted() {
  local path="$1"
  grep -Fqx 'schema=pireus-operator-novelty-frontier.material-parity.v11' "${path}" &&
    grep -Fqx 'status=MATERIAL_FROZEN_CENSUS_RECONSTRUCTION_WITH_OPEN_NATIVE_AND_U250_CARD1_DEBT' "${path}" &&
    grep -Fqx 'authority_language=Sounio' "${path}" &&
    grep -Fqx 'producing_language=C++' "${path}" &&
    grep -Fqx 'canonical_target_class_coverage=4/4' "${path}" &&
    grep -Fqx 'canonical_target_class_coverage_complete=true' "${path}" &&
    grep -Fqx 'observed_physical_endpoints=5' "${path}" &&
    grep -Fqx 'unresolved_physical_endpoints=1' "${path}" &&
    grep -Fqx 'canonical_physical_endpoint_coverage=5/6' "${path}" &&
    grep -Fqx 'canonical_physical_endpoint_coverage_complete=false' "${path}" &&
    grep -Fqx 'spark_scheduler_route=KUBERNETES' "${path}" &&
    grep -Fqx 'slurm_route_used=false' "${path}" &&
    grep -Fqx 'grammar_candidates_evaluated_by_cpp=7200' "${path}" &&
    grep -Fqx 'separators=43200' "${path}" &&
    grep -Fqx 'c2_character_scope=TRIVIAL_UNSIGNED_CHARACTER_CHECKED_CELLWISE' "${path}" &&
    grep -Fqx 'analytic_proof_by_cpp=false' "${path}" &&
    grep -Fqx 'atlas_entries_claim_same_algebraic_variety=false' "${path}" &&
    grep -Fqx 'material_reconstruction_parity_complete=true' "${path}" &&
    grep -Fqx 'material_target_frozen_census_complete=true' "${path}" &&
    grep -Fqx 'native_target_lowering_parity_complete=false' "${path}" &&
    grep -Fqx 'material_target_coverage_complete=false' "${path}" &&
    grep -Fqx 'material_parity_complete=false' "${path}" &&
    grep -Fqx 'candidate_selected=false' "${path}" &&
    grep -Fqx 'historical_novelty=false' "${path}" &&
    grep -Fqx 'priority_claim=false' "${path}" &&
    grep -Fqx 'claim_ready=false' "${path}" &&
    grep -Fqx 'llm_confirmed_result=false' "${path}" &&
    grep -Fqx "commit_frame_sha256=${COMMIT_FRAME_SHA256}" "${path}" &&
    grep -Fqx 'commit_decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN' "${path}" &&
    grep -Fqx 'result=SCOPED_MATERIAL_RECONSTRUCTION_PARITY_PASS' "${path}" &&
    ! grep -Fqx 'observed_physical_endpoints=6' "${path}" &&
    ! grep -Fqx 'slurm_route_used=true' "${path}" &&
    ! grep -Fqx 'analytic_proof_by_cpp=true' "${path}" &&
    ! grep -Fqx 'material_parity_complete=true' "${path}" &&
    ! grep -Fqx 'candidate_selected=true' "${path}" &&
    ! grep -Fqx 'claim_ready=true' "${path}"
}

cd "${ROOT}"
require_hash "${ROOT}/${SOUNIO_REL}" "${SOUNIO_SHA256}"
require_hash "${ROOT}/${FREEZE_REL}" "${FREEZE_SHA256}"
require_hash "${ROOT}/${OPEN_REL}" "${OPEN_SHA256}"
require_hash "${ROOT}/${FORMAL_REL}" "${FORMAL_SHA256}"
require_hash "${ROOT}/${FORMAL_GATE_REL}" "${FORMAL_GATE_SHA256}"
require_hash "${ROOT}/${EFFECT_REL}" "${EFFECT_SHA256}"
require_hash "${ROOT}/${EFFECT_GATE_REL}" "${EFFECT_GATE_SHA256}"
require_hash "${ROOT}/${CPP_REL}" "${CPP_SHA256}"
require_hash "${ROOT}/${XEON_REL}" "${XEON_SHA256}"
require_hash "${ROOT}/${APPLE_REL}" "${APPLE_SHA256}"
require_hash "${ROOT}/${DGX24_REL}" "${DGX24_SHA256}"
require_hash "${ROOT}/${DGX48_REL}" "${DGX48_SHA256}"
require_hash "${ROOT}/${U250_REL}" "${U250_SHA256}"
require_hash "${ROOT}/${RECEIPT_REL}" "${RECEIPT_SHA256}"
require_hash "${GUARDIAN}" "${GUARDIAN_SHA256}"
require_hash "${ROOT}/stdlib/coordination/loom_language_authority.sio" "${GUARDIAN_POLICY_SHA256}"
[[ -x "${GUARDIAN}" ]] || fail 'native Sounio Guardian unavailable'
command -v g++ >/dev/null 2>&1 || fail 'g++ unavailable for local material replay'

git -C "${ROOT}" merge-base --is-ancestor "${FORMAL_COMMIT}" HEAD ||
  fail 'formal parity commit is not an ancestor'
git -C "${ROOT}" merge-base --is-ancestor "${EFFECT_COMMIT}" HEAD ||
  fail 'effect parity commit is not an ancestor'
[[ "$(git -C "${ROOT}" show "${FORMAL_COMMIT}:${FORMAL_REL}" | sha256sum | cut -d' ' -f1)" == "${FORMAL_SHA256}" ]] ||
  fail 'committed formal receipt drift'
[[ "$(git -C "${ROOT}" show "${EFFECT_COMMIT}:${EFFECT_REL}" | sha256sum | cut -d' ' -f1)" == "${EFFECT_SHA256}" ]] ||
  fail 'committed effect receipt drift'

require_line "${ROOT}/${FREEZE_REL}" "module_sha256=${SOUNIO_SHA256}"
require_line "${ROOT}/${FREEZE_REL}" "semantics_sha256=${SEMANTICS_SHA256}"
require_line "${ROOT}/${OPEN_REL}" 'stage=PARITY_OPEN'
require_line "${ROOT}/${FORMAL_REL}" 'formal_parity_complete=true'
require_line "${ROOT}/${EFFECT_REL}" 'effect_parity_complete=true'
require_line "${ROOT}/${EFFECT_REL}" 'material_parity_complete=false'

validate_evidence "${ROOT}/${XEON_REL}" xeon XEON
validate_evidence "${ROOT}/${APPLE_REL}" apple APPLE_SILICON
validate_evidence "${ROOT}/${DGX24_REL}" dgx24 DGX_GB10_24
validate_evidence "${ROOT}/${DGX48_REL}" dgx48 DGX_GB10_48
validate_evidence "${ROOT}/${U250_REL}" u250 AMD_ALVEO_U250_DUAL_CARD
require_line "${ROOT}/${DGX24_REL}" 'target_locator=cluster-node:spark-3c59'
require_line "${ROOT}/${DGX48_REL}" 'target_locator=cluster-node:spark-8e54'
require_line "${ROOT}/${U250_REL}" 'u250_declared_card_count=2'
require_line "${ROOT}/${U250_REL}" 'u250_observed_card_count=1'
require_line "${ROOT}/${U250_REL}" 'u250_unresolved_card_count=1'
receipt_admitted "${ROOT}/${RECEIPT_REL}" || fail 'material receipt admission failed'

for mutation in endpoint slurm analytic selected parity claim; do
  cp "${ROOT}/${RECEIPT_REL}" "${TMP_ROOT}/${mutation}.v11"
done
sed -i 's/^observed_physical_endpoints=5$/observed_physical_endpoints=6/' "${TMP_ROOT}/endpoint.v11"
sed -i 's/^slurm_route_used=false$/slurm_route_used=true/' "${TMP_ROOT}/slurm.v11"
sed -i 's/^analytic_proof_by_cpp=false$/analytic_proof_by_cpp=true/' "${TMP_ROOT}/analytic.v11"
sed -i 's/^candidate_selected=false$/candidate_selected=true/' "${TMP_ROOT}/selected.v11"
sed -i 's/^material_parity_complete=false$/material_parity_complete=true/' "${TMP_ROOT}/parity.v11"
sed -i 's/^claim_ready=false$/claim_ready=true/' "${TMP_ROOT}/claim.v11"
for mutation in endpoint slurm analytic selected parity claim; do
  receipt_admitted "${TMP_ROOT}/${mutation}.v11" && fail "receipt sabotage passed: ${mutation}"
done
printf 'SABOTAGE endpoint_6_of_6=REFUSED slurm_route=REFUSED analytic_proof=REFUSED candidate_selection=REFUSED material_parity=REFUSED claim=REFUSED\n'

check_guardian SEMANTIC_WRITE \
  "$(authority_frame 3 4 4 4 1 1 0 0 0 "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" zero)" \
  "${SEMANTIC_WRITE_FRAME_SHA256}" 113 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=SEMANTICS_FROZEN'
check_guardian POLICY_MISSING \
  "$(authority_frame 3 4 4 4 0 0 0 0 0 "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" zero)" \
  "${POLICY_MISSING_FRAME_SHA256}" 101 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=101 reason=policy-missing next_stage=SEMANTICS_FROZEN'
check_guardian POLICY_TIMEOUT \
  "$(authority_frame 3 4 4 4 2 0 0 0 0 "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" zero)" \
  "${POLICY_TIMEOUT_FRAME_SHA256}" 102 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=102 reason=policy-timeout next_stage=SEMANTICS_FROZEN'
check_guardian POLICY_ERROR \
  "$(authority_frame 3 4 4 4 3 0 0 0 0 "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" zero)" \
  "${POLICY_ERROR_FRAME_SHA256}" 103 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=103 reason=policy-error next_stage=SEMANTICS_FROZEN'
check_guardian PYTHON_ORACLE \
  "$(authority_frame 3 4 7 7 1 0 0 0 0 "${PYTHON_TOOLCHAIN_SHA256}" "${PYTHON_COMMAND_SHA256}" zero)" \
  "${PYTHON_FRAME_SHA256}" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
check_guardian RUST_ORACLE \
  "$(authority_frame 3 4 8 7 1 0 0 0 0 "${RUST_TOOLCHAIN_SHA256}" "${RUST_COMMAND_SHA256}" zero)" \
  "${RUST_FRAME_SHA256}" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
check_guardian CLAIM_PROMOTION \
  "$(authority_frame 4 7 4 4 1 0 0 1 0 "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" "${XEON_SHA256}")" \
  "${CLAIM_FRAME_SHA256}" 123 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=123 reason=action-forbidden-for-role next_stage=PARITY_OPEN'

check_guardian PREEXEC \
  "$(authority_frame 3 4 4 4 1 0 0 0 0 "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" zero)" \
  "${PREEXEC_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
mkdir -p "${BUILD_ROOT}"
bash -c "${REPLAY_COMMAND}"
require_hash "${REPLAY_BINARY}" "${BINARY_SHA256}"
require_hash "${REPLAY_OUTPUT}" "${XEON_SHA256}"
cmp -s "${REPLAY_OUTPUT}" "${ROOT}/${XEON_REL}" || fail 'local replay differs from frozen Xeon evidence'

check_guardian SEAL \
  "$(authority_frame 4 8 4 4 1 0 0 1 0 "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" "${XEON_SHA256}")" \
  "${SEAL_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
check_guardian RECEIPT_WRITE \
  "$(authority_frame 4 9 4 4 1 0 0 1 0 "${TOOLCHAIN_SHA256}" "${COMMAND_SHA256}" "${XEON_SHA256}")" \
  "${WRITE_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'

require_hash "${ROOT}/${RECEIPT_REL}" "${RECEIPT_SHA256}"
receipt_admitted "${ROOT}/${RECEIPT_REL}" || fail 'receipt drifted during gate'
printf '%s\n' \
  'pireus operator novelty frontier material parity: STAGE_REACHED_NOT_A_CLAIM stage=PARITY_OPEN language=C++ role=MATERIAL_PARITY scope=EXHAUSTIVE_FROZEN_CENSUS_RECONSTRUCTION classes=4/4 endpoints=5/6 grammar=7200 separators=43200 xeon=PASS apple=PASS dgx24=PASS_K8S dgx48=PASS_K8S u250_card0=PASS u250_card1=UNRESOLVED native_lowering=false material_parity_complete=false claim_ready=false python_process_launched=false rust_process_launched=false'
