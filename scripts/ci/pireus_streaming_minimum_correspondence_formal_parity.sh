#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN_CANDIDATE="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
GUARDIAN="$(readlink -f "${GUARDIAN_CANDIDATE}")"

SOUNIO_REL='stdlib/hardware/pireus/operator_orbit_canonicalization.sio'
FREEZE_REL='tools/pireus/operator_orbit_canonicalization.freeze.v13'
MAIN_REL='formal/lean4/SounioPireusStreamingMinimumCorrespondence.lean'
AUDIT_REL='formal/lean4/SounioPireusStreamingMinimumCorrespondenceAxiomAudit.lean'
CHECK_REL='formal/lean4/SounioPireusStreamingMinimumCorrespondenceCheck.lean'
LAKEFILE_REL='formal/lean4/lakefile.lean'
OFFLOAD_LOG_REL='.claude/llm_offload_log.md'
PARENT_GATE_REL='scripts/ci/pireus_concrete_quotient_action_formal_parity.sh'
PARENT_RECEIPT_REL='tools/pireus/concrete_quotient_action.formal-parity.v13'
PARENT_EVIDENCE_REL='tools/pireus/evidence/concrete_quotient_action_v13.formal-parity.txt'
GATE_REL='scripts/ci/pireus_streaming_minimum_correspondence_formal_parity.sh'
RECEIPT_REL='tools/pireus/streaming_minimum_correspondence.formal-parity.v13'
EVIDENCE_REL='tools/pireus/evidence/streaming_minimum_correspondence_v13.formal-parity.txt'

PARENT_GATE_COMMIT='6b253c47a51a6d15050c714ab2eb8b1f957fbc5b'
FORMAL_SOURCE_COMMIT='7781fc4dfe2ef50e27db9ed9d73a31eed35df418'
PROOF_CHECK_COMMIT='0035a87bd2dbb75eb11e5c6332760774365a2f20'
ARTIFACT_COMMIT='PENDING_ARTIFACT_COMMIT'
PRESEAL_GATE_SHA256='PENDING_PRESEAL_GATE_SHA256'
ARTIFACT_GATE_ANCHOR_COMMIT='PENDING_ARTIFACT_COMMIT'
ARTIFACT_GATE_ANCHOR_PRESEAL_SHA256='PENDING_PRESEAL_GATE_SHA256'

SOUNIO_SHA256='7ada1b17bf91fdb3f4c48877d2485f71a65bb4159d88cb7e4b288c77bfe3cdae'
FREEZE_SHA256='11893a34450729ff06ac40ade86c90decb7a6947daea3cc108cae17f73572f84'
MAIN_SHA256='4d244dbbeb8bb0ef85ceb27994dd84bbcad402d4f58b0e3b2ff5908692ff4397'
AUDIT_SHA256='c116a305cb28d73847c910b0db8725cab6d1215ed735f721b378a9ce36d42b8c'
CHECK_SHA256='d6520e6c3da3448ac78b3bccd959ebbc0a905c7c16197e4c39f8a108e8360ac3'
LAKEFILE_SHA256='7992ce727698567504989f963c46e89b0ba9d0cdf79b3ecb5859f2da831506b1'
FORMAL_SOURCE_REVIEW_LOG_SHA256='416902a5596de446026a24487e366dde4f3485e72b91d407d7379eaafc3df05a'
PROOF_CHECK_REVIEW_LOG_SHA256='7a30d67c84514f85fcfbec9f8f3bcfadb316c1f408daff5afb1329450185ae6e'
ARTIFACT_REVIEW_LOG_SHA256='4abea42f0763864ba15499b855c0b9c949d8e47fed0ff151d67c2f344d67ec23'
PARENT_GATE_SHA256='85592497911796c5111aabf1267bc4085ac19e91b345694ce12669e1e0330a50'
PARENT_RECEIPT_SHA256='60c96ae686e578124426f8dcf864f917c98572465ed65f22ca13c8693f2e2270'
PARENT_EVIDENCE_SHA256='d8255c3c26860a5754c1e5116d943e1606c1791351850d713c752a6f651e06b7'
RECEIPT_SHA256='92bbde468d6e091739366b7417edaefa78a05d5d08b3175ac05db5053686da22'
EVIDENCE_SHA256='f360f3fd3e966bc6b02c156b390a9538aad602e9d7bbe3209464028f7ad86f04'
SOURCE_BUNDLE_SHA256='bbe87fa7b7e4bd5e357e79483a26db1a1729271281294355a2f545b0f696150b'
SOUNIO_SEMANTICS_SHA256='0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c'
TOOLCHAIN_SHA256='a0786d46c580b27dad335f46e9c12a69cb6b59db438097f2ad4fe0d9eced1a6c'
HARDWARE_SHA256='e764b122c1b973fc54fe6c781bd92cc7c394f0dad4d895812f88f7d15af50d23'
LIVE_HARDWARE_MANIFEST_SHA256='ed98ba37afb72f73ed32b8d84fa17a221b5bb8483df454ffe870b65a913f1b7a'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
PYTHON_TOOLCHAIN_SHA256='422c73f42aff0794a916a210455ea8a5d2bbbd29430963b94c29966017abc517'
ZERO='0 0 0 0 0 0 0 0'

BUILD_COMMAND='cd formal/lean4 && lake build SounioPireusConcreteQuotientAction && lake env lean -j 1 SounioPireusConcreteQuotientTarget03Check.lean && lake env lean -j 1 SounioPireusStreamingMinimumCorrespondence.lean -o .lake/build/lib/lean/SounioPireusStreamingMinimumCorrespondence.olean -i .lake/build/lib/lean/SounioPireusStreamingMinimumCorrespondence.ilean'
CHECK_COMMAND='cd formal/lean4 && lake env lean -j 1 SounioPireusStreamingMinimumCorrespondenceCheck.lean'
AUDIT_COMMAND='cd formal/lean4 && lake env lean -j 1 SounioPireusStreamingMinimumCorrespondenceAxiomAudit.lean'
PYTHON_COMMAND='python3 forbidden_streaming_oracle.py'
BUILD_COMMAND_SHA256='1adc8b4d5ec459ed0fe6dc57bde4d40c0df7a7f431282ce3149e07c8184a0d4a'
CHECK_COMMAND_SHA256='68668ae9479b23c56cfc8ef3bdacd6a311c13c87af4fc3a6989c7e8f623c989c'
AUDIT_COMMAND_SHA256='648a4bfce666d373eeec3a28dad3ec14e3fe621afa09d4739fcaa01e93296459'
PYTHON_COMMAND_SHA256='6975a37c18a65e7186dc6aea5332cb2ce9543b61413d839c41bd79e7fb7e3c69'
BUILD_FRAME_SHA256='155fde96d190c6e56b0ab97a6fc8ecd6a0c8d2a2f3ecc2a5727d9ea5cfd88740'
CHECK_FRAME_SHA256='7254a63ddeefbe2ec901d035286f5e6e0419275616d4897f3bab1c0687e04814'
AUDIT_FRAME_SHA256='59cad38aef3c90e583b5e87abd95953d9a087d3180040e78bbbf60524b15231c'
PYTHON_FRAME_SHA256='2ecc0ddfedb6d8ba434d4cc70823c8f57cfd595d08b5e63f281864b4c0a587a7'
RUNTIME_EVIDENCE_MATERIAL_SHA256='0e5202126cae7b4d28206564f8603faec8a1afa129be0230e0bdedc2e4074c76'

EXPECTED_THEOREMS=(
  sounio_streaming_step_eq_min sounio_streaming_minimum_eq_list_min?
  quotient_streaming_minimum_eq_abstract_canonical every_frozen_scan_entry_mem
  every_analytic_basis_entry_mem every_frozen_scan_action_view_mem
  every_analytic_action_view_mem mapped_frozen_scan_action_views_membership
  scan_witness_eq_mapped_analytic_witness
  action_of_frozen_scan_view_eq_mapped_analytic_action
  frozen_scan_candidate_eq_mapped_quotient_action
  frozen_scan_candidate_membership_eq_analytic_orbit
  frozen_scan_model_streaming_minimum_eq_declared_canonical
  streaming_fold_closed_without_executed_parity_promotion
)
EXPECTED_DEFINITIONS=(
  sounioStreamingStep sounioStreamingMinimum quotientStreamingCanonicalOption
  frozenScanActionViews scanViewToAnalytic actionOfFrozenScanView
  frozenScanCandidate frozenScanCandidateList frozenScanModelStreamingCanonicalOption
  StreamingMinimumCorrespondenceBoundary streamingMinimumCorrespondenceBoundary
)
EXPECTED_ABBREVIATIONS=(FrozenScanActionView)

fail() {
  printf 'pireus streaming minimum correspondence formal parity: FAIL: %s\n' "$*" >&2
  exit 1
}

sha_file() { sha256sum "$1" | cut -d' ' -f1; }
sha_text() { printf '%s\n' "$1" | sha256sum | cut -d' ' -f1; }
count_occurrences() { grep -c -- "$1" <<<"$2" || true; }

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

parity_frame() {
  local source_sha="$1" command_sha="$2"
  printf '9020 4 4 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${source_sha}")" "$(sha_limbs "${SOUNIO_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SOUNIO_SEMANTICS_SHA256}")" "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" "$(sha_limbs "${command_sha}")" "${ZERO}" "${ZERO}"
}

python_oracle_frame() {
  printf '9020 4 4 7 7 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${SOURCE_BUNDLE_SHA256}")" "$(sha_limbs "${SOUNIO_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SOUNIO_SEMANTICS_SHA256}")" "$(sha_limbs "${PYTHON_TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" "$(sha_limbs "${PYTHON_COMMAND_SHA256}")" "${ZERO}" "${ZERO}"
}

authorize() {
  local label="$1" frame="$2" expected_sha="$3" decision
  [[ "$(wc -w <<<"${frame}" | tr -d ' ')" -eq 82 ]] || fail "Guardian frame words: ${label}"
  [[ "$(sha_text "${frame}")" == "${expected_sha}" ]] || fail "Guardian frame drift: ${label}"
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  [[ "${decision}" == 'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN' ]] ||
    fail "Guardian decision drift: ${label}: ${decision}"
  printf 'GUARDIAN_DECISION label=%s decision=%s\n' "${label}" "${decision}"
}

cd "${ROOT}"
require_hash "${ROOT}/${SOUNIO_REL}" "${SOUNIO_SHA256}"
require_hash "${ROOT}/${FREEZE_REL}" "${FREEZE_SHA256}"
require_hash "${ROOT}/${MAIN_REL}" "${MAIN_SHA256}"
require_hash "${ROOT}/${AUDIT_REL}" "${AUDIT_SHA256}"
require_hash "${ROOT}/${CHECK_REL}" "${CHECK_SHA256}"
require_hash "${ROOT}/${LAKEFILE_REL}" "${LAKEFILE_SHA256}"
require_hash "${ROOT}/${PARENT_GATE_REL}" "${PARENT_GATE_SHA256}"
require_hash "${ROOT}/${PARENT_RECEIPT_REL}" "${PARENT_RECEIPT_SHA256}"
require_hash "${ROOT}/${PARENT_EVIDENCE_REL}" "${PARENT_EVIDENCE_SHA256}"
require_hash "${ROOT}/${RECEIPT_REL}" "${RECEIPT_SHA256}"
require_hash "${ROOT}/${EVIDENCE_REL}" "${EVIDENCE_SHA256}"
require_hash "${GUARDIAN}" "${GUARDIAN_SHA256}"

semantics_sha256="$(
  sed -n '/^semantics_material_begin$/,/^semantics_material_end$/p' "${ROOT}/${FREEZE_REL}" |
    sed '1d;$d' | sha256sum | cut -d' ' -f1
)"
[[ "${semantics_sha256}" == "${SOUNIO_SEMANTICS_SHA256}" ]] || fail 'Sounio semantics digest drift'

source_manifest="$(printf '%s\n' \
  "${SOUNIO_REL}=${SOUNIO_SHA256}" "${MAIN_REL}=${MAIN_SHA256}" \
  "${AUDIT_REL}=${AUDIT_SHA256}" "${CHECK_REL}=${CHECK_SHA256}")"
[[ "$(sha_text "${source_manifest}")" == "${SOURCE_BUNDLE_SHA256}" ]] || fail 'source bundle digest drift'

git -C "${ROOT}" merge-base --is-ancestor "${PARENT_GATE_COMMIT}" "${FORMAL_SOURCE_COMMIT}" || fail 'Target-03 parent does not precede formal source'
git -C "${ROOT}" merge-base --is-ancestor "${FORMAL_SOURCE_COMMIT}" "${PROOF_CHECK_COMMIT}" || fail 'formal source does not precede proof check'
git -C "${ROOT}" merge-base --is-ancestor "${PROOF_CHECK_COMMIT}" "${ARTIFACT_COMMIT}" || fail 'proof check does not precede artifact seal'
git -C "${ROOT}" merge-base --is-ancestor "${ARTIFACT_COMMIT}" HEAD || fail 'artifact seal not in current history'
require_committed_hash "${PARENT_GATE_COMMIT}" "${SOUNIO_REL}" "${SOUNIO_SHA256}"
require_committed_hash "${PARENT_GATE_COMMIT}" "${FREEZE_REL}" "${FREEZE_SHA256}"
require_committed_hash "${PARENT_GATE_COMMIT}" "${PARENT_GATE_REL}" "${PARENT_GATE_SHA256}"
require_committed_hash "${PARENT_GATE_COMMIT}" "${PARENT_RECEIPT_REL}" "${PARENT_RECEIPT_SHA256}"
require_committed_hash "${PARENT_GATE_COMMIT}" "${PARENT_EVIDENCE_REL}" "${PARENT_EVIDENCE_SHA256}"
require_committed_hash "${FORMAL_SOURCE_COMMIT}" "${MAIN_REL}" "${MAIN_SHA256}"
require_committed_hash "${FORMAL_SOURCE_COMMIT}" "${AUDIT_REL}" "${AUDIT_SHA256}"
require_committed_hash "${FORMAL_SOURCE_COMMIT}" "${OFFLOAD_LOG_REL}" "${FORMAL_SOURCE_REVIEW_LOG_SHA256}"
require_committed_hash "${PROOF_CHECK_COMMIT}" "${CHECK_REL}" "${CHECK_SHA256}"
require_committed_hash "${PROOF_CHECK_COMMIT}" "${OFFLOAD_LOG_REL}" "${PROOF_CHECK_REVIEW_LOG_SHA256}"
require_committed_hash "${ARTIFACT_COMMIT}" "${RECEIPT_REL}" "${RECEIPT_SHA256}"
require_committed_hash "${ARTIFACT_COMMIT}" "${EVIDENCE_REL}" "${EVIDENCE_SHA256}"
require_committed_hash "${ARTIFACT_COMMIT}" "${OFFLOAD_LOG_REL}" "${ARTIFACT_REVIEW_LOG_SHA256}"
require_committed_hash "${ARTIFACT_COMMIT}" "${GATE_REL}" "${PRESEAL_GATE_SHA256}"

normalized_live_gate_sha256="$(
  sed \
    -e "s/^ARTIFACT_COMMIT=.*/ARTIFACT_COMMIT='${ARTIFACT_GATE_ANCHOR_COMMIT}'/" \
    -e "s/^PRESEAL_GATE_SHA256=.*/PRESEAL_GATE_SHA256='${ARTIFACT_GATE_ANCHOR_PRESEAL_SHA256}'/" \
    "${ROOT}/${GATE_REL}" | sha256sum | cut -d' ' -f1
)"
[[ "${normalized_live_gate_sha256}" == "${PRESEAL_GATE_SHA256}" ]] || fail 'executing gate bytes are not the sealed two-field transformation'

[[ "$(lean --version | head -1)" == 'Lean (version 4.33.1, x86_64-unknown-linux-gnu, commit 819816b2e0a3bf405af45ae5c7af2491d8f5bee6, Release)' ]] || fail 'Lean version drift'
[[ "$(lake --version)" == 'Lake version 5.0.0-src+819816b (Lean version 4.33.1)' ]] || fail 'Lake version drift'
[[ "$(uname -m)" == 'x86_64' ]] || fail 'architecture drift'
live_node="$(hostname)"
live_arch="$(uname -m)"
live_cpu="$(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1 | tr ' ' '_')"
live_logical_cpus="$(lscpu | sed -n 's/^CPU(s):[[:space:]]*//p' | head -1)"
live_sockets="$(lscpu | sed -n 's/^Socket(s):[[:space:]]*//p' | head -1)"
live_cores_per_socket="$(lscpu | sed -n 's/^Core(s) per socket:[[:space:]]*//p' | head -1)"
live_threads_per_core="$(lscpu | sed -n 's/^Thread(s) per core:[[:space:]]*//p' | head -1)"
live_hardware_manifest="$(printf '%s\n' \
  'execution_route=LOCAL_XEON_WORKSPACE_CONTROL' \
  "execution_node=${live_node}" "execution_architecture=${live_arch}" "execution_cpu=${live_cpu}" \
  "execution_logical_cpu_count=${live_logical_cpus}" "execution_socket_count=${live_sockets}" \
  "execution_cores_per_socket=${live_cores_per_socket}" "execution_threads_per_core=${live_threads_per_core}")"
[[ "$(sha_text "${live_hardware_manifest}")" == "${LIVE_HARDWARE_MANIFEST_SHA256}" ]] || fail 'live hardware manifest drift'

[[ "$(sha_text "${BUILD_COMMAND}")" == "${BUILD_COMMAND_SHA256}" ]] || fail 'build command drift'
[[ "$(sha_text "${CHECK_COMMAND}")" == "${CHECK_COMMAND_SHA256}" ]] || fail 'check command drift'
[[ "$(sha_text "${AUDIT_COMMAND}")" == "${AUDIT_COMMAND_SHA256}" ]] || fail 'audit command drift'
[[ "$(sha_text "${PYTHON_COMMAND}")" == "${PYTHON_COMMAND_SHA256}" ]] || fail 'Python command drift'

[[ "$(grep -Ec '^theorem ' "${ROOT}/${MAIN_REL}")" -eq 14 ]] || fail 'theorem count drift'
[[ "$(grep -Ec '^(def|structure) ' "${ROOT}/${MAIN_REL}")" -eq 11 ]] || fail 'definition count drift'
[[ "$(grep -Ec '^abbrev ' "${ROOT}/${MAIN_REL}")" -eq 1 ]] || fail 'abbreviation count drift'
[[ "$(grep -c '^#print axioms ' "${ROOT}/${AUDIT_REL}")" -eq 26 ]] || fail 'axiom audit count drift'
[[ "$(grep -c '^example ' "${ROOT}/${CHECK_REL}")" -eq 9 ]] || fail 'proof-check obligation count drift'

for theorem_name in "${EXPECTED_THEOREMS[@]}"; do
  awk -v name="${theorem_name}" '$1 == "theorem" && $2 == name { found = 1 } END { exit !found }' \
    "${ROOT}/${MAIN_REL}" || fail "missing theorem: ${theorem_name}"
  grep -Fqx "#print axioms ${theorem_name}" "${ROOT}/${AUDIT_REL}" || fail "missing theorem audit: ${theorem_name}"
done
for definition_name in "${EXPECTED_DEFINITIONS[@]}"; do
  grep -Eq "^(def|structure) ${definition_name}([[:space:]]|:|$)" "${ROOT}/${MAIN_REL}" || fail "missing definition: ${definition_name}"
  grep -Fqx "#print axioms ${definition_name}" "${ROOT}/${AUDIT_REL}" || fail "missing definition audit: ${definition_name}"
done
for abbreviation_name in "${EXPECTED_ABBREVIATIONS[@]}"; do
  grep -Eq "^abbrev ${abbreviation_name}([[:space:]]|:|$)" "${ROOT}/${MAIN_REL}" || fail "missing abbreviation: ${abbreviation_name}"
  grep -Fqx "#print axioms ${abbreviation_name}" "${ROOT}/${AUDIT_REL}" || fail "missing abbreviation audit: ${abbreviation_name}"
done

for source_file in "${ROOT}/${MAIN_REL}" "${ROOT}/${CHECK_REL}"; do
  [[ "$(grep -Ec '\bnative_decide\b|\bsorry\b|sorryAx' "${source_file}" || true)" -eq 0 ]] ||
    fail "source trust-marker drift: ${source_file}"
done
require_line "${ROOT}/${SOUNIO_REL}" 'pub const PIREUS_POC_ACTIONS: i64 = 40320'
require_line "${ROOT}/${SOUNIO_REL}" '    while code < PIREUS_POC_MATRIX_CODES {'
require_line "${ROOT}/${SOUNIO_REL}" '        if poc_matrix_invertible(code) {'
require_line "${ROOT}/${SOUNIO_REL}" '    var have_best = false'
require_line "${ROOT}/${SOUNIO_REL}" '            var smaller = !have_best'
require_line "${ROOT}/${SOUNIO_REL}" '            if have_best {'
require_line "${ROOT}/${SOUNIO_REL}" '                    if value < best[cell] { smaller = true; break }'
require_line "${ROOT}/${SOUNIO_REL}" '                    if value > best[cell] { break }'
require_line "${ROOT}/${SOUNIO_REL}" '            if smaller {'
require_line "${ROOT}/${SOUNIO_REL}" '                have_best = true'
require_line "${ROOT}/${MAIN_REL}" 'theorem frozen_scan_model_streaming_minimum_eq_declared_canonical'
require_line "${ROOT}/${MAIN_REL}" '  , frozenSounioSourceHashBindingProved := false'
require_line "${ROOT}/${MAIN_REL}" '  , executedSounioStreamingMinimumEqualityProved := false'
require_line "${ROOT}/${RECEIPT_REL}" 'full_formal_source_hash_pins_theorem_statements=true'
require_line "${ROOT}/${RECEIPT_REL}" 'status=PARTIAL_PASS'
require_line "${ROOT}/${RECEIPT_REL}" 'preexec_sounio_source_hash_match=true'
require_line "${ROOT}/${RECEIPT_REL}" 'source_hash_binding_proved=false'
require_line "${ROOT}/${RECEIPT_REL}" 'executed_sounio_streaming_minimum_equality_proved=false'
require_line "${ROOT}/${RECEIPT_REL}" 'executed_sounio_equality_scope=NONE_UNTIL_SEMANTIC_LINK'
require_line "${ROOT}/${RECEIPT_REL}" 'packed_word_layout_parity_claim=false'
require_line "${ROOT}/${RECEIPT_REL}" 'lean_model_correspondence_proved=true'
require_line "${ROOT}/${RECEIPT_REL}" 'v13_canonicalizer_formal_parity_complete=false'
require_line "${ROOT}/${RECEIPT_REL}" 'formal_parity_complete=false'
require_line "${ROOT}/${RECEIPT_REL}" 'computable_extraction_claim=false'
require_line "${ROOT}/${RECEIPT_REL}" 'claim_ready=false'
require_line "${ROOT}/${RECEIPT_REL}" 'spark_route_policy=KUBERNETES_ONLY'
require_line "${ROOT}/${RECEIPT_REL}" 'u250_declared_card_count=2'
require_line "${ROOT}/${RECEIPT_REL}" 'u250_installed_card_count=1'
require_line "${ROOT}/${RECEIPT_REL}" 'u250_pending_installation_card_count=1'
require_line "${ROOT}/${RECEIPT_REL}" 'result_enum=V13_LEAN_MODEL_MINIMUM_THEOREMS_CHECKED_EXECUTED_LINK_OPEN_NO_CLAIM'

python_frame="$(python_oracle_frame)"
[[ "$(wc -w <<<"${python_frame}" | tr -d ' ')" -eq 82 ]] || fail 'Python frame word count drift'
[[ "$(sha_text "${python_frame}")" == "${PYTHON_FRAME_SHA256}" ]] || fail 'Python frame drift'
set +e
python_decision="$(printf '%s\n' "${python_frame}" | "${GUARDIAN}")"
python_rc=$?
set -e
[[ "${python_rc}" -eq 110 ]] || fail "Python oracle exit drift: ${python_rc}"
[[ "${python_decision}" == 'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=PARITY_OPEN' ]] || fail 'Python oracle decision drift'
[[ ! -e "${ROOT}/forbidden_streaming_oracle.py" ]] || fail 'forbidden Python oracle file exists'

negative_dir="$(mktemp -d /tmp/pireus-streaming-minimum-negative.XXXXXX)"
trap 'rm -rf "${negative_dir}"' EXIT
set +e
(
  GUARDIAN=/bin/false
  authorize LOCAL_XEON_BUILD "$(parity_frame "${MAIN_SHA256}" "${BUILD_COMMAND_SHA256}")" "${BUILD_FRAME_SHA256}"
  printf 'LEAN_STARTED\n' >"${negative_dir}/lean-started.txt"
) >"${negative_dir}/guardian-false.txt" 2>&1
negative_rc=$?
set -e
[[ "${negative_rc}" -eq 1 ]] || fail "Guardian override negative exit drift: ${negative_rc}"
[[ ! -e "${negative_dir}/lean-started.txt" ]] || fail 'Guardian override reached Lean'

authorize LOCAL_XEON_BUILD "$(parity_frame "${MAIN_SHA256}" "${BUILD_COMMAND_SHA256}")" "${BUILD_FRAME_SHA256}"
set +e
build_output="$(bash -c "${BUILD_COMMAND}" 2>&1)"
build_rc=$?
set -e
[[ "${build_rc}" -eq 0 ]] || fail "Lean build exit drift: ${build_rc}"
if grep -Eq 'SounioPireus(ConcreteQuotientTarget03Check|StreamingMinimumCorrespondence)\.lean:' <<<"${build_output}"; then fail 'formal source warning drift'; fi

authorize LOCAL_XEON_PROOF_CHECK "$(parity_frame "${CHECK_SHA256}" "${CHECK_COMMAND_SHA256}")" "${CHECK_FRAME_SHA256}"
check_output="$(bash -c "${CHECK_COMMAND}" 2>&1)"
[[ "$(count_occurrences '^warning:' "${check_output}")" -eq 0 ]] || fail 'proof check warning drift'

authorize LOCAL_XEON_AXIOM_AUDIT "$(parity_frame "${AUDIT_SHA256}" "${AUDIT_COMMAND_SHA256}")" "${AUDIT_FRAME_SHA256}"
audit_output="$(bash -c "${AUDIT_COMMAND}" 2>&1)"
[[ "$(count_occurrences "^'SounioPireus" "${audit_output}")" -eq 26 ]] || fail 'axiom report count drift'
[[ "$(count_occurrences 'depends on axioms:' "${audit_output}")" -eq 23 ]] || fail 'axiom-bearing count drift'
[[ "$(count_occurrences 'does not depend on any axioms' "${audit_output}")" -eq 3 ]] || fail 'axiom-free count drift'
[[ "$(count_occurrences 'propext' "${audit_output}")" -eq 23 ]] || fail 'propext count drift'
[[ "$(count_occurrences 'Classical.choice' "${audit_output}")" -eq 13 ]] || fail 'Classical.choice count drift'
[[ "$(count_occurrences 'Quot.sound' "${audit_output}")" -eq 21 ]] || fail 'Quot.sound count drift'
[[ "$(count_occurrences 'native_decide' "${audit_output}")" -eq 0 ]] || fail 'native_decide count drift'
[[ "$(count_occurrences 'sorryAx' "${audit_output}")" -eq 0 ]] || fail 'sorryAx count drift'
axiom_blocks="$(tr '\n' ' ' <<<"${audit_output}" | grep -oE 'depends on axioms: \[[^]]+\]')"
[[ "$(grep -c '^depends on axioms:' <<<"${axiom_blocks}")" -eq 23 ]] || fail 'axiom parser coverage drift'
axiom_set="$(
  sed 's/^depends on axioms: \[//;s/\]$//' <<<"${axiom_blocks}" |
    tr ',' '\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | sort -u
)"
[[ "${axiom_set}" == $'Classical.choice\nQuot.sound\npropext' ]] || fail "unexpected axiom set: ${axiom_set}"

runtime_evidence_material="$(printf '%s\n' \
  "parent_target03_gate_commit=${PARENT_GATE_COMMIT}" "formal_source_commit=${FORMAL_SOURCE_COMMIT}" \
  "proof_check_commit=${PROOF_CHECK_COMMIT}" "sounio_source_sha256=${SOUNIO_SHA256}" \
  "sounio_semantics_sha256=${SOUNIO_SEMANTICS_SHA256}" 'build_exit_code=0' \
  'proof_check_exit_code=0' 'axiom_audit_exit_code=0' 'proof_check_obligations=9' \
  'axiom_partition=26,23,3' 'axiom_allowlist=Classical.choice,Quot.sound,propext' \
  'python_guardian_rc=110' 'guardian_false_exit=1' \
  'full_formal_source_hash_pins_statements=true' 'preexec_sounio_source_hash_match=true' \
  'source_hash_binding_proved=false' 'executed_sounio_streaming_minimum_equality=false' \
  'lean_model_correspondence_proved=true' 'packed_word_layout_parity=false' \
  'spark_route_policy=KUBERNETES_ONLY' 'u250_inventory=2,1,1,0' \
  'result=V13_LEAN_MODEL_MINIMUM_THEOREMS_CHECKED_EXECUTED_LINK_OPEN_NO_CLAIM')"
[[ "$(sha_text "${runtime_evidence_material}")" == "${RUNTIME_EVIDENCE_MATERIAL_SHA256}" ]] || fail 'runtime evidence material drift'

printf 'PIREUS_STREAMING_MINIMUM_CORRESPONDENCE_RESULT=V13_LEAN_MODEL_MINIMUM_THEOREMS_CHECKED_EXECUTED_LINK_OPEN_NO_CLAIM stage=PARITY_OPEN language=Lean4 role=FORMAL_PARITY semantic_authority=Sounio full_formal_source_hash_pins_statements=true preexec_sounio_source_hash_match=true source_hash_binding_proved=false lean_model_streaming_step_eq_min=true lean_model_streaming_fold_option_eq_list_min_option=true lean_model_scan_candidate_set_eq_analytic_orbit=true lean_model_minimum_eq_declared_canonical=true lean_model_correspondence_proved=true executed_sounio_streaming_equality=false executed_scope=NONE_UNTIL_SEMANTIC_LINK packed_word_layout_parity=false v13_canonicalizer_formal_parity_complete=false formal_parity_complete=false effect_parity_complete=false material_parity_complete=false computable_extraction_claim=false performance_claim=false subquadratic_claim=false guardian_role=PREEXECUTION_POLICY axiom_reports=26 axiom_free=3 axiom_allowlist=Classical.choice,Quot.sound,propext unexpected_axioms=0 native_decide=0 sorryax=0 python_dispatch=REFUSED_PREEXEC_E110 guardian_current_gate_false_exit=1 spark_route_policy=KUBERNETES_ONLY spark_dispatches_by_this_gate=0 dgx_dispatches_by_this_gate=0 slurm_dispatches_by_this_gate=0 live_cluster_usage=NOT_MEASURED u250_declared=2 u250_installed=1 u250_pending_installation=1 u250_enumeration_failures=0 u250_dispatches_by_this_gate=0 llm_role=REVIEW_ONLY llm_confirmed_result=false novelty_confirmed=false claim_ready=false runtime_evidence_sha256=%s\n' "${RUNTIME_EVIDENCE_MATERIAL_SHA256}"
