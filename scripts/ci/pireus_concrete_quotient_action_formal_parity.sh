#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN_CANDIDATE="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
GUARDIAN="$(readlink -f "${GUARDIAN_CANDIDATE}")"

SIGN_REL='formal/lean4/SounioPireusSignTableBitVecLex.lean'
CLOSURE_REL='formal/lean4/SounioPireusAnalyticActionClosure.lean'
QUOTIENT_REL='formal/lean4/SounioPireusConcreteQuotientAction.lean'
SIGN_AUDIT_REL='formal/lean4/SounioPireusSignTableBitVecLexAxiomAudit.lean'
CLOSURE_AUDIT_REL='formal/lean4/SounioPireusAnalyticActionClosureAxiomAudit.lean'
QUOTIENT_AUDIT_REL='formal/lean4/SounioPireusConcreteQuotientActionAxiomAudit.lean'
TARGET03_CHECK_REL='formal/lean4/SounioPireusConcreteQuotientTarget03Check.lean'
LAKEFILE_REL='formal/lean4/lakefile.lean'
OFFLOAD_LOG_REL='.claude/llm_offload_log.md'
BASE_GATE_REL='scripts/ci/pireus_operator_orbit_canonicalization.sh'
BASE_FREEZE_REL='tools/pireus/operator_orbit_canonicalization.freeze.v13'
PARITY_OPEN_REL='tools/pireus/operator_orbit_canonicalization.parity-open.v13'
GAUGE_DEFINITION_REL='formal/lean4/SounioPireusGaugeCoboundaryAction.lean'
PARENT_RECEIPT_REL='tools/pireus/gl4_analytic_scan_bijection.formal-parity.v13'
PARENT_GATE_REL='scripts/ci/pireus_gl4_analytic_scan_bijection_formal_parity.sh'
PARENT_EVIDENCE_REL='tools/pireus/evidence/gl4_analytic_scan_bijection_v13.formal-parity.txt'
RECEIPT_REL='tools/pireus/concrete_quotient_action.formal-parity.v13'
EVIDENCE_REL='tools/pireus/evidence/concrete_quotient_action_v13.formal-parity.txt'
GATE_REL='scripts/ci/pireus_concrete_quotient_action_formal_parity.sh'

PARENT_GATE_COMMIT='e04304321bf55e2538c31fffc07ea23c2d69be77'
SOURCE_COMMIT='5b014f2dd2527237da6c8a9bd82aa9224ec29640'
ARTIFACT_COMMIT='dca735b391a159a7fdfd1b4cd27a77fce056861b'
PRESEAL_GATE_SHA256='42ae74ff7407fc122574e7e9abbc8c302d83305dcdef5979ec38262440993a41'
ARTIFACT_GATE_ANCHOR_COMMIT='cbbef9f945f57df4e90b62fbff27096c6ea3b71e'
ARTIFACT_GATE_ANCHOR_PRESEAL_SHA256='68bcbae017580da4d51d6b282217f7cee379543ea6c183e39efa27f2fcf0aa1a'
SOURCE_BUNDLE_SHA256='55fcedae89418f94271773815f09805d2999b1f75268408421b7ac72e1646387'
AUDIT_BUNDLE_SHA256='8a2c2485b5111b4b18ad8093743c117ad66564ccc79d025a2641229c6196731b'
SIGN_SHA256='160b68484acedc4f501f181b28eed3e3a17bf483b1207cca1c5ca433661696c9'
CLOSURE_SHA256='0db799b30a9d6da25f441ec83e421e038b5b32fb0a0175671406be0d0aa98e06'
QUOTIENT_SHA256='4c702485789ff233d563d95039ae39c9fee3f01eeaac7147a7d88eb5234f3d6d'
SIGN_AUDIT_SHA256='4f539d3c0adc342acd456a80451bc167e717206c5c21d9d34d17530ce784320c'
CLOSURE_AUDIT_SHA256='02ca9ba1c0110c58c25fbb4edb0579bf37e51148ab970b24f335855f0b5f8e44'
QUOTIENT_AUDIT_SHA256='99a6ae72d01c5b200086771058f1835eb78e5bd3c5b154c61cd53670951a8840'
TARGET03_CHECK_SHA256='c64ff31970a700f21e96e34719a05e341321c8ccf636641b46b9004d4b681d66'
LAKEFILE_SHA256='7992ce727698567504989f963c46e89b0ba9d0cdf79b3ecb5859f2da831506b1'
SOURCE_OFFLOAD_LOG_SHA256='d93426d27e26cac031b1943629bb4b7d8a9889aefbeec712e057c1db2afe24ff'
GATE_OFFLOAD_LOG_SHA256='8a14c942966821adf67cd4fec8e1cbf7f048e46f3f62e147e7f3db28a850da3a'
PARENT_RECEIPT_SHA256='36c23734c1a05ddea1716de5d0013cb795aeaf20ebb811848a7d6699211e6b3e'
PARENT_GATE_SHA256='aa413f08d12c447a992bcacf54e68a376cac1babd71f6b3b2fd662c2ccb32dd2'
PARENT_EVIDENCE_SHA256='5363ce3100b5429c7a8d7c9dd9dba14a836f2af10d1f1e4f5823e521ea250218'
RECEIPT_SHA256='60c96ae686e578124426f8dcf864f917c98572465ed65f22ca13c8693f2e2270'
EVIDENCE_SHA256='d8255c3c26860a5754c1e5116d943e1606c1791351850d713c752a6f651e06b7'
BASE_GATE_SHA256='6a18d7061bd408a3050d468d65c53231d0010865543346352e7ae91a0ff11f0e'
BASE_FREEZE_SHA256='11893a34450729ff06ac40ade86c90decb7a6947daea3cc108cae17f73572f84'
PARITY_OPEN_SHA256='4d24259d1807cffa999a90aea4e4797fcbce50659c2e59c34deccb3ca33bdfbf'
GAUGE_DEFINITION_SHA256='bf25ef66f9b5fab5f1c08e7aa16fb3875bc8ee990a7615d20ea87d4357db5901'
SOUNIO_SEMANTICS_SHA256='0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c'
TOOLCHAIN_SHA256='a0786d46c580b27dad335f46e9c12a69cb6b59db438097f2ad4fe0d9eced1a6c'
HARDWARE_SHA256='e764b122c1b973fc54fe6c781bd92cc7c394f0dad4d895812f88f7d15af50d23'
LIVE_HARDWARE_MANIFEST_SHA256='ed98ba37afb72f73ed32b8d84fa17a221b5bb8483df454ffe870b65a913f1b7a'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
PYTHON_TOOLCHAIN_SHA256='422c73f42aff0794a916a210455ea8a5d2bbbd29430963b94c29966017abc517'
ZERO='0 0 0 0 0 0 0 0'

BUILD_COMMAND='cd formal/lean4 && lake build SounioPireusSignTableBitVecLex SounioPireusAnalyticActionClosure SounioPireusConcreteQuotientAction'
SIGN_COMMAND='cd formal/lean4 && lake env lean -j 1 SounioPireusSignTableBitVecLex.lean'
CLOSURE_COMMAND='cd formal/lean4 && lake env lean -j 1 SounioPireusAnalyticActionClosure.lean'
QUOTIENT_COMMAND='cd formal/lean4 && lake env lean -j 1 SounioPireusConcreteQuotientAction.lean'
TARGET03_CHECK_COMMAND='cd formal/lean4 && lake env lean -j 1 SounioPireusConcreteQuotientTarget03Check.lean'
AUDIT_COMMAND='cd formal/lean4 && lake env lean -j 1 SounioPireusSignTableBitVecLexAxiomAudit.lean && lake env lean -j 1 SounioPireusAnalyticActionClosureAxiomAudit.lean && lake env lean -j 1 SounioPireusConcreteQuotientActionAxiomAudit.lean'
PYTHON_COMMAND='python3 forbidden_oracle.py'
BUILD_COMMAND_SHA256='da0c1c450c1437bee9ce8ddf9ddec3091e2fa39603ca02987d5beb96e37de3fe'
SIGN_COMMAND_SHA256='d9a07b8c4a82bc5b893a785556211990591ecad6ed793599d5e8544c0d840e8a'
CLOSURE_COMMAND_SHA256='8c2aafd73910b7c5466a67b1ed0bc8e77b87366127b7ba9b57b0b90d3b3d13db'
QUOTIENT_COMMAND_SHA256='941f6459d0ee0866339ad1598cf6648407461a91db072cbbedb147dedb804a1a'
TARGET03_CHECK_COMMAND_SHA256='c3375a51e0e07e3269abad95f0e064bb14b932891a4e47232872689620c895d0'
AUDIT_COMMAND_SHA256='d7780c7e7c4fffac07d4b3500ed95158e90718170c97594dacd540c5347af497'
PYTHON_COMMAND_SHA256='7aeabfbcf642a0ee63423bc26a51d01b475474d2a29a0d173b900a697942ce31'
BUILD_FRAME_SHA256='de3b347e4fdbd9fc783a7889a726f357c526648afadfc50768dccd7fb21839cd'
SIGN_FRAME_SHA256='55b4c9559ee23e78114e134447ffeda456a939b925caa0eea400fdfda1cb6fc6'
CLOSURE_FRAME_SHA256='e692e5d013e8b3cb8f9924d6e2e88007b2a5a8d3bd209e52f6110f534036f8bc'
QUOTIENT_FRAME_SHA256='63e110687055831cef7a1ad8551827e61e1cb6ef5180384feba7ba26dce0bf87'
TARGET03_CHECK_FRAME_SHA256='d02e9f8640f7798c01caae418087406d1a12c3167933f78b2b3ec2857225cbce'
AUDIT_FRAME_SHA256='b80a942dba98d83bf24767b6a2edae7533303764928e42af98b9e637428536e2'
RUNTIME_EVIDENCE_MATERIAL_SHA256='e005fe57cde21c155c578c4e067ac8b834ce4e540d81e8559aa1aafaa5534334'

SOURCE_FILES=("${ROOT}/${SIGN_REL}" "${ROOT}/${CLOSURE_REL}" "${ROOT}/${QUOTIENT_REL}")
AUDIT_FILES=("${ROOT}/${SIGN_AUDIT_REL}" "${ROOT}/${CLOSURE_AUDIT_REL}" "${ROOT}/${QUOTIENT_AUDIT_REL}")

EXPECTED_THEOREMS=(
  cell_index_of_index cell_of_index_index table_cell_list_length pack_table_get_msb
  unpack_pack_table pack_unpack_table pack_table_injective unpack_table_injective
  xor_lane_equiv_injective linear_map_span_zero linear_map_span_one linear_map_span_two
  linear_map_span_three linear_map_span_four basis_of_linear_first_mem
  basis_of_linear_second_mem basis_of_linear_third_mem basis_of_linear_fourth_mem
  basis_of_linear_mem_analytic_ordered_bases analytic_basis_entry_of_linear_mem
  view_of_action_mem xor_linear_to_fun_eq_of_basis recoded_action_lane1
  recoded_action_lane2 recoded_action_lane4 recoded_action_lane8 recoded_action_to_fun
  raw_act_view_of_action identity_view_mem compose_view_mem inverse_view_mem
  raw_act_identity_view raw_act_compose_view raw_act_inverse_view
  normalize_raw_action_absorbs_normalize table_of_normalized_bits_in_section
  table_of_normalized_bits_of_table normalized_bits_eq_of_tables_eq table_of_quotient_act
  normalized_bits_of_table_eq_iff_same_gauge_orbit quotient_act_on_normalized_table
  quotient_action_identity quotient_action_compose quotient_action_inverse
  concrete_quotient_action_count_is_40320 concrete_quotient_canonical_eq_iff_same_orbit
  normalized_same_orbit_iff_same_declared_linear_swap_gauge_orbit
  declared_canonical_eq_iff_same_declared_linear_swap_gauge_orbit
  concrete_quotient_closes_target03_without_claim_promotion
)

EXPECTED_DEFINITIONS=(
  cellIndex cellOfIndex tableCellList packTable unpackTable basisOfLinear
  analyticBasisEntryOfLinear viewOfAction identityView composeView inverseView
  IsNormalizedBits normalizedBitsOfTable tableOfNormalizedBits
  quotientAct concreteQuotientActionSystem SameDeclaredLinearSwapGaugeOrbit
  declaredCanonicalOption ConcreteQuotientBoundary concreteQuotientBoundary
)

EXPECTED_TYPE_ABBREVIATIONS=(AnalyticActionView NormalizedBits)
EXPECTED_INSTANCES=(normalizedBitsMin normalizedBitsLawfulOrderMin)

fail() {
  printf 'pireus concrete quotient action formal parity: FAIL: %s\n' "$*" >&2
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
require_hash "${ROOT}/${BASE_GATE_REL}" "${BASE_GATE_SHA256}"
require_hash "${ROOT}/${BASE_FREEZE_REL}" "${BASE_FREEZE_SHA256}"
require_hash "${ROOT}/${PARITY_OPEN_REL}" "${PARITY_OPEN_SHA256}"
require_hash "${ROOT}/${GAUGE_DEFINITION_REL}" "${GAUGE_DEFINITION_SHA256}"
require_hash "${ROOT}/${SIGN_REL}" "${SIGN_SHA256}"
require_hash "${ROOT}/${CLOSURE_REL}" "${CLOSURE_SHA256}"
require_hash "${ROOT}/${QUOTIENT_REL}" "${QUOTIENT_SHA256}"
require_hash "${ROOT}/${SIGN_AUDIT_REL}" "${SIGN_AUDIT_SHA256}"
require_hash "${ROOT}/${CLOSURE_AUDIT_REL}" "${CLOSURE_AUDIT_SHA256}"
require_hash "${ROOT}/${QUOTIENT_AUDIT_REL}" "${QUOTIENT_AUDIT_SHA256}"
require_hash "${ROOT}/${TARGET03_CHECK_REL}" "${TARGET03_CHECK_SHA256}"
require_hash "${ROOT}/${LAKEFILE_REL}" "${LAKEFILE_SHA256}"
require_hash "${ROOT}/${OFFLOAD_LOG_REL}" "${GATE_OFFLOAD_LOG_SHA256}"
require_hash "${ROOT}/${PARENT_RECEIPT_REL}" "${PARENT_RECEIPT_SHA256}"
require_hash "${ROOT}/${PARENT_GATE_REL}" "${PARENT_GATE_SHA256}"
require_hash "${ROOT}/${PARENT_EVIDENCE_REL}" "${PARENT_EVIDENCE_SHA256}"
require_hash "${ROOT}/${RECEIPT_REL}" "${RECEIPT_SHA256}"
require_hash "${ROOT}/${EVIDENCE_REL}" "${EVIDENCE_SHA256}"
require_hash "${GUARDIAN}" "${GUARDIAN_SHA256}"

base_semantics_sha256="$(
  sed -n '/^semantics_material_begin$/,/^semantics_material_end$/p' "${ROOT}/${BASE_FREEZE_REL}" |
    sed '1d;$d' | sha256sum | cut -d' ' -f1
)"
[[ "${base_semantics_sha256}" == "${SOUNIO_SEMANTICS_SHA256}" ]] || fail 'live Sounio semantics digest drift'
base_output="$("${ROOT}/${BASE_GATE_REL}")"
grep -Fq 'PIREUS_OPERATOR_ORBIT_CANONICALIZATION_GATE_PASS=true' <<<"${base_output}" || fail 'Sounio authority gate failed'
grep -Fq ' stage=SEMANTICS_FROZEN ' <<<"${base_output}" || fail 'Sounio stage drift'
grep -Fq ' language=Sounio role=SEMANTIC_AUTHORITY ' <<<"${base_output}" || fail 'Sounio authority role drift'
grep -Fq ' spark_route=KUBERNETES_ONLY ' <<<"${base_output}" || fail 'Spark route drift'
grep -Fq ' u250_declared=2 u250_installed=1 u250_pending_installation=1 ' <<<"${base_output}" || fail 'U250 inventory drift'

source_manifest="$(printf '%s\n' \
  "${SIGN_REL}=${SIGN_SHA256}" "${CLOSURE_REL}=${CLOSURE_SHA256}" "${QUOTIENT_REL}=${QUOTIENT_SHA256}")"
audit_manifest="$(printf '%s\n' \
  "${SIGN_AUDIT_REL}=${SIGN_AUDIT_SHA256}" "${CLOSURE_AUDIT_REL}=${CLOSURE_AUDIT_SHA256}" \
  "${QUOTIENT_AUDIT_REL}=${QUOTIENT_AUDIT_SHA256}")"
[[ "$(sha_text "${source_manifest}")" == "${SOURCE_BUNDLE_SHA256}" ]] || fail 'source bundle digest drift'
[[ "$(sha_text "${audit_manifest}")" == "${AUDIT_BUNDLE_SHA256}" ]] || fail 'audit bundle digest drift'

git -C "${ROOT}" merge-base --is-ancestor "${PARENT_GATE_COMMIT}" "${SOURCE_COMMIT}" || fail 'parent gate does not precede source'
git -C "${ROOT}" merge-base --is-ancestor "${SOURCE_COMMIT}" "${ARTIFACT_COMMIT}" || fail 'source does not precede artifact seal'
git -C "${ROOT}" merge-base --is-ancestor "${ARTIFACT_COMMIT}" HEAD || fail 'artifact seal not in current history'
require_committed_hash "${PARENT_GATE_COMMIT}" "${PARENT_RECEIPT_REL}" "${PARENT_RECEIPT_SHA256}"
require_committed_hash "${PARENT_GATE_COMMIT}" "${PARENT_GATE_REL}" "${PARENT_GATE_SHA256}"
require_committed_hash "${PARENT_GATE_COMMIT}" "${PARENT_EVIDENCE_REL}" "${PARENT_EVIDENCE_SHA256}"
require_committed_hash "${PARENT_GATE_COMMIT}" "${PARITY_OPEN_REL}" "${PARITY_OPEN_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${SIGN_REL}" "${SIGN_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${CLOSURE_REL}" "${CLOSURE_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${QUOTIENT_REL}" "${QUOTIENT_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${SIGN_AUDIT_REL}" "${SIGN_AUDIT_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${CLOSURE_AUDIT_REL}" "${CLOSURE_AUDIT_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${QUOTIENT_AUDIT_REL}" "${QUOTIENT_AUDIT_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${LAKEFILE_REL}" "${LAKEFILE_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${OFFLOAD_LOG_REL}" "${SOURCE_OFFLOAD_LOG_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${GAUGE_DEFINITION_REL}" "${GAUGE_DEFINITION_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${TARGET03_CHECK_REL}" "${TARGET03_CHECK_SHA256}"
require_committed_hash "${ARTIFACT_COMMIT}" "${RECEIPT_REL}" "${RECEIPT_SHA256}"
require_committed_hash "${ARTIFACT_COMMIT}" "${EVIDENCE_REL}" "${EVIDENCE_SHA256}"
require_committed_hash "${ARTIFACT_COMMIT}" "${GATE_REL}" "${PRESEAL_GATE_SHA256}"
require_committed_hash "${ARTIFACT_COMMIT}" "${TARGET03_CHECK_REL}" "${TARGET03_CHECK_SHA256}"

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
[[ "$(sha_text "${SIGN_COMMAND}")" == "${SIGN_COMMAND_SHA256}" ]] || fail 'sign command drift'
[[ "$(sha_text "${CLOSURE_COMMAND}")" == "${CLOSURE_COMMAND_SHA256}" ]] || fail 'closure command drift'
[[ "$(sha_text "${QUOTIENT_COMMAND}")" == "${QUOTIENT_COMMAND_SHA256}" ]] || fail 'quotient command drift'
[[ "$(sha_text "${TARGET03_CHECK_COMMAND}")" == "${TARGET03_CHECK_COMMAND_SHA256}" ]] || fail 'Target-03 check command drift'
[[ "$(sha_text "${AUDIT_COMMAND}")" == "${AUDIT_COMMAND_SHA256}" ]] || fail 'audit command drift'
[[ "$(sha_text "${PYTHON_COMMAND}")" == "${PYTHON_COMMAND_SHA256}" ]] || fail 'Python command drift'

[[ "$(grep -Ec '^(@\[[^]]+\][[:space:]]+)?theorem ' "${ROOT}/${SIGN_REL}")" -eq 8 ]] || fail 'sign theorem count drift'
[[ "$(grep -Ec '^(def|abbrev|structure) ' "${ROOT}/${SIGN_REL}")" -eq 5 ]] || fail 'sign definition count drift'
[[ "$(grep -Ec '^(@\[[^]]+\][[:space:]]+)?theorem ' "${ROOT}/${CLOSURE_REL}")" -eq 26 ]] || fail 'closure theorem count drift'
[[ "$(grep -Ec '^(def|abbrev|structure) ' "${ROOT}/${CLOSURE_REL}")" -eq 7 ]] || fail 'closure definition count drift'
[[ "$(grep -Ec '^(@\[[^]]+\][[:space:]]+)?theorem ' "${ROOT}/${QUOTIENT_REL}")" -eq 15 ]] || fail 'quotient theorem count drift'
[[ "$(grep -Ec '^(def|abbrev|structure) ' "${ROOT}/${QUOTIENT_REL}")" -eq 10 ]] || fail 'quotient definition count drift'
[[ "$(grep -c '^#print axioms ' "${ROOT}/${SIGN_AUDIT_REL}")" -eq 13 ]] || fail 'sign audit count drift'
[[ "$(grep -c '^#print axioms ' "${ROOT}/${CLOSURE_AUDIT_REL}")" -eq 32 ]] || fail 'closure audit count drift'
[[ "$(grep -c '^#print axioms ' "${ROOT}/${QUOTIENT_AUDIT_REL}")" -eq 26 ]] || fail 'quotient audit count drift'

for theorem_name in "${EXPECTED_THEOREMS[@]}"; do
  grep -Eq "^(@\[[^]]+\][[:space:]]+)?theorem ${theorem_name}([[:space:]]|:|$)" "${SOURCE_FILES[@]}" || fail "missing theorem: ${theorem_name}"
  grep -Fqx "#print axioms ${theorem_name}" "${AUDIT_FILES[@]}" || fail "missing theorem audit: ${theorem_name}"
done
for definition_name in "${EXPECTED_DEFINITIONS[@]}"; do
  grep -Eq "^(def|structure) ${definition_name}([[:space:]]|:|$)" "${SOURCE_FILES[@]}" || fail "missing definition: ${definition_name}"
  grep -Fqx "#print axioms ${definition_name}" "${AUDIT_FILES[@]}" || fail "missing definition audit: ${definition_name}"
done
for abbreviation_name in "${EXPECTED_TYPE_ABBREVIATIONS[@]}"; do
  grep -Eq "^abbrev ${abbreviation_name}([[:space:]]|:|$)" "${SOURCE_FILES[@]}" || fail "missing type abbreviation: ${abbreviation_name}"
done
for instance_name in "${EXPECTED_INSTANCES[@]}"; do
  grep -Eq "^instance ${instance_name}([[:space:]]|:|$)" "${SOURCE_FILES[@]}" || fail "missing instance: ${instance_name}"
  grep -Fqx "#print axioms ${instance_name}" "${AUDIT_FILES[@]}" || fail "missing instance audit: ${instance_name}"
done
for source_file in "${SOURCE_FILES[@]}"; do
  [[ "$(grep -c 'native_decide' "${source_file}" || true)" -eq 0 ]] || fail "native_decide drift: ${source_file}"
  [[ "$(grep -Ec '\bsorry\b|sorryAx' "${source_file}" || true)" -eq 0 ]] || fail "sorry drift: ${source_file}"
done

require_line "${ROOT}/${QUOTIENT_REL}" '  { exactLexBitVecRepresentationProved := true'
require_line "${ROOT}/${QUOTIENT_REL}" '  , analytic40320ActionClosureProved := true'
require_line "${ROOT}/${QUOTIENT_REL}" '  , gaugeNormalizationAbsorptionProved := true'
require_line "${ROOT}/${QUOTIENT_REL}" '  , concreteQuotientActionLawsProved := true'
require_line "${ROOT}/${QUOTIENT_REL}" '  , concreteQuotientCanonicalIffOrbitProved := true'
require_line "${ROOT}/${QUOTIENT_REL}" '  , executedSounioStreamingMinimumEqualityProved := false'
require_line "${ROOT}/${QUOTIENT_REL}" '  , concreteCanonicalEqualityIffDeclaredLinearSwapGaugeOrbitProved := true'
require_line "${ROOT}/${QUOTIENT_REL}" '  , formalTarget03Closed := true'
require_line "${ROOT}/${QUOTIENT_REL}" '  , formalParityClosed := false'
require_line "${ROOT}/${QUOTIENT_REL}" '  , claimReady := false }'
require_line "${ROOT}/${PARITY_OPEN_REL}" 'formal_target_03=PROVE_CANONICAL_REPRESENTATIVE_EQUALITY_IFF_SAME_DECLARED_GL4_SWAP_GAUGE_ORBIT'
require_line "${ROOT}/${GAUGE_DEFINITION_REL}" 'def gaugeBits : Nat := 11'
require_line "${ROOT}/${GAUGE_DEFINITION_REL}" 'def gaugeWords : Nat := 2 ^ gaugeBits'
require_line "${ROOT}/${GAUGE_DEFINITION_REL}" 'abbrev GaugeWord := Fin (2 ^ gaugeBits)'
require_line "${ROOT}/${QUOTIENT_REL}" 'theorem declared_canonical_eq_iff_same_declared_linear_swap_gauge_orbit'
require_line "${ROOT}/${QUOTIENT_REL}" '    (left right : SignTable) :'
require_line "${ROOT}/${QUOTIENT_REL}" '    declaredCanonicalOption left = declaredCanonicalOption right ↔'
require_line "${ROOT}/${QUOTIENT_REL}" '      SameDeclaredLinearSwapGaugeOrbit left right := by'
require_line "${ROOT}/${QUOTIENT_REL}" 'theorem concrete_quotient_closes_target03_without_claim_promotion :'
require_line "${ROOT}/${QUOTIENT_REL}" '      !concreteQuotientBoundary.claimReady) = true := by'
require_line "${ROOT}/${TARGET03_CHECK_REL}" '      !concreteQuotientBoundary.claimReady) = true :='
require_line "${ROOT}/${RECEIPT_REL}" 'formal_target_03=PROVE_CANONICAL_REPRESENTATIVE_EQUALITY_IFF_SAME_DECLARED_GL4_SWAP_GAUGE_ORBIT'
require_line "${ROOT}/${RECEIPT_REL}" 'formal_target_03_closed=true'
require_line "${ROOT}/${RECEIPT_REL}" 'executed_sounio_streaming_minimum_equality_proved=false'
require_line "${ROOT}/${RECEIPT_REL}" 'formal_parity_complete=false'
require_line "${ROOT}/${RECEIPT_REL}" 'spark_route_policy=KUBERNETES_ONLY'
require_line "${ROOT}/${RECEIPT_REL}" 'u250_declared_card_count=2'
require_line "${ROOT}/${RECEIPT_REL}" 'u250_installed_card_count=1'
require_line "${ROOT}/${RECEIPT_REL}" 'u250_pending_installation_card_count=1'
require_line "${ROOT}/${RECEIPT_REL}" 'u250_enumeration_failure_count=0'
require_line "${ROOT}/${RECEIPT_REL}" 'llm_role=REVIEW_ONLY'
require_line "${ROOT}/${RECEIPT_REL}" 'llm_confirmed_result=false'
require_line "${ROOT}/${RECEIPT_REL}" 'axiom_audit_public_theorem_coverage=49_OF_49'
require_line "${ROOT}/${RECEIPT_REL}" 'axiom_audit_public_definition_reports=20'
require_line "${ROOT}/${RECEIPT_REL}" 'axiom_audit_public_instance_reports=2'
require_line "${ROOT}/${RECEIPT_REL}" 'axiom_audit_public_instance_names=normalizedBitsMin,normalizedBitsLawfulOrderMin'
require_line "${ROOT}/${RECEIPT_REL}" 'axiom_audit_unreported_type_abbreviations=2'
require_line "${ROOT}/${RECEIPT_REL}" 'source_hashed_type_abbreviations=true'
require_line "${ROOT}/${RECEIPT_REL}" 'computable_extraction_claim=false'
require_line "${ROOT}/${RECEIPT_REL}" 'choice_free_core_claim=false'
require_line "${ROOT}/${RECEIPT_REL}" 'guardian_role=PREEXECUTION_POLICY_NOT_FORMAL_PROOF'
require_line "${ROOT}/${RECEIPT_REL}" 'result_enum=TARGET03_CLOSED_PARITY_OPEN_CLAIM_NOT_READY'
require_line "${ROOT}/${RECEIPT_REL}" 'claim_ready=false'
grep -Fq 'V13 TARGET-03 CONCRETE QUOTIENT ACTION SOURCE REVIEW COMPLETE' "${ROOT}/${OFFLOAD_LOG_REL}" || fail 'source review log row missing'
grep -Fq 'provider error 1313' "${ROOT}/${OFFLOAD_LOG_REL}" || fail 'Z.AI failure disclosure missing'

python_frame="$(python_oracle_frame)"
[[ "$(wc -w <<<"${python_frame}" | tr -d ' ')" -eq 82 ]] || fail 'Python frame word count drift'
set +e
python_decision="$(printf '%s\n' "${python_frame}" | "${GUARDIAN}")"
python_rc=$?
set -e
[[ "${python_rc}" -eq 110 ]] || fail "Python oracle exit drift: ${python_rc}"
[[ "${python_decision}" == 'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=PARITY_OPEN' ]] || fail "Python oracle decision drift"
[[ ! -e "${ROOT}/forbidden_oracle.py" ]] || fail 'forbidden Python oracle file exists at gate root'

negative_dir="$(mktemp -d /tmp/pireus-concrete-quotient-negative.XXXXXX)"
trap 'rm -rf "${negative_dir}"' EXIT
set +e
(
  GUARDIAN=/bin/false
  authorize LOCAL_XEON_BUILD "$(parity_frame "${SOURCE_BUNDLE_SHA256}" "${BUILD_COMMAND_SHA256}")" "${BUILD_FRAME_SHA256}"
  printf 'LEAN_STARTED\n' >"${negative_dir}/lean-started.txt"
) >"${negative_dir}/guardian-false.txt" 2>&1
negative_rc=$?
set -e
[[ "${negative_rc}" -eq 1 ]] || fail "Guardian override negative exit drift: ${negative_rc}"
[[ ! -e "${negative_dir}/lean-started.txt" ]] || fail 'Guardian override reached Lean'

authorize LOCAL_XEON_BUILD "$(parity_frame "${SOURCE_BUNDLE_SHA256}" "${BUILD_COMMAND_SHA256}")" "${BUILD_FRAME_SHA256}"
set +e
build_output="$(cd "${ROOT}/formal/lean4" && lake build SounioPireusSignTableBitVecLex SounioPireusAnalyticActionClosure SounioPireusConcreteQuotientAction 2>&1)"
build_rc=$?
set -e
[[ "${build_rc}" -eq 0 ]] || fail "Lean build exit drift: ${build_rc}"
if grep -Eq 'SounioPireus(SignTableBitVecLex|AnalyticActionClosure|ConcreteQuotientAction)\.lean:' <<<"${build_output}"; then fail 'local source warning drift'; fi

authorize LOCAL_XEON_SIGN_TYPECHECK "$(parity_frame "${SIGN_SHA256}" "${SIGN_COMMAND_SHA256}")" "${SIGN_FRAME_SHA256}"
sign_output="$(cd "${ROOT}/formal/lean4" && lake env lean -j 1 SounioPireusSignTableBitVecLex.lean 2>&1)"
[[ "$(count_occurrences '^warning:' "${sign_output}")" -eq 0 ]] || fail 'sign typecheck warning drift'
authorize LOCAL_XEON_CLOSURE_TYPECHECK "$(parity_frame "${CLOSURE_SHA256}" "${CLOSURE_COMMAND_SHA256}")" "${CLOSURE_FRAME_SHA256}"
closure_output="$(cd "${ROOT}/formal/lean4" && lake env lean -j 1 SounioPireusAnalyticActionClosure.lean 2>&1)"
[[ "$(count_occurrences '^warning:' "${closure_output}")" -eq 0 ]] || fail 'closure typecheck warning drift'
authorize LOCAL_XEON_QUOTIENT_TYPECHECK "$(parity_frame "${QUOTIENT_SHA256}" "${QUOTIENT_COMMAND_SHA256}")" "${QUOTIENT_FRAME_SHA256}"
quotient_output="$(cd "${ROOT}/formal/lean4" && lake env lean -j 1 SounioPireusConcreteQuotientAction.lean 2>&1)"
[[ "$(count_occurrences '^warning:' "${quotient_output}")" -eq 0 ]] || fail 'quotient typecheck warning drift'

authorize LOCAL_XEON_TARGET03_CHECK "$(parity_frame "${TARGET03_CHECK_SHA256}" "${TARGET03_CHECK_COMMAND_SHA256}")" "${TARGET03_CHECK_FRAME_SHA256}"
target03_check_output="$(cd "${ROOT}/formal/lean4" && lake env lean -j 1 SounioPireusConcreteQuotientTarget03Check.lean 2>&1)"
[[ "$(count_occurrences '^warning:' "${target03_check_output}")" -eq 0 ]] || fail 'Target-03 proof check warning drift'

authorize LOCAL_XEON_AXIOM_AUDIT "$(parity_frame "${AUDIT_BUNDLE_SHA256}" "${AUDIT_COMMAND_SHA256}")" "${AUDIT_FRAME_SHA256}"
audit_output="$({
  cd "${ROOT}/formal/lean4"
  lake env lean -j 1 SounioPireusSignTableBitVecLexAxiomAudit.lean
  lake env lean -j 1 SounioPireusAnalyticActionClosureAxiomAudit.lean
  lake env lean -j 1 SounioPireusConcreteQuotientActionAxiomAudit.lean
} 2>&1)"
[[ "$(count_occurrences "^'SounioPireus" "${audit_output}")" -eq 71 ]] || fail 'axiom report count drift'
[[ "$(count_occurrences 'depends on axioms:' "${audit_output}")" -eq 68 ]] || fail 'axiom-bearing count drift'
[[ "$(count_occurrences 'does not depend on any axioms' "${audit_output}")" -eq 3 ]] || fail 'axiom-free count drift'
[[ "$(count_occurrences 'propext' "${audit_output}")" -eq 68 ]] || fail 'propext count drift'
[[ "$(count_occurrences 'Classical.choice' "${audit_output}")" -eq 46 ]] || fail 'Classical.choice count drift'
[[ "$(count_occurrences 'Quot.sound' "${audit_output}")" -eq 64 ]] || fail 'Quot.sound count drift'
[[ "$(count_occurrences 'native_decide' "${audit_output}")" -eq 0 ]] || fail 'native_decide count drift'
[[ "$(count_occurrences 'sorryAx' "${audit_output}")" -eq 0 ]] || fail 'sorryAx count drift'
axiom_blocks="$(
  tr '\n' ' ' <<<"${audit_output}" |
    grep -oE 'depends on axioms: \[[^]]+\]'
)"
parsed_axiom_report_count="$(grep -c '^depends on axioms:' <<<"${axiom_blocks}")"
[[ "${parsed_axiom_report_count}" -eq 68 ]] || fail 'axiom allowlist parser coverage drift'
axiom_set="$(
  sed 's/^depends on axioms: \[//;s/\]$//' <<<"${axiom_blocks}" |
    tr ',' '\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | sort -u
)"
[[ "${axiom_set}" == $'Classical.choice\nQuot.sound\npropext' ]] || fail "unexpected axiom set: ${axiom_set}"

runtime_evidence_material="$(printf '%s\n' \
  "source_commit=${SOURCE_COMMIT}" "sounio_semantics=${SOUNIO_SEMANTICS_SHA256}" \
  'target03_definition=PROVE_CANONICAL_REPRESENTATIVE_EQUALITY_IFF_SAME_DECLARED_GL4_SWAP_GAUGE_ORBIT' \
  "target03_check_sha256=${TARGET03_CHECK_SHA256}" 'build_exit_code=0' \
  'sign_typecheck_exit_code=0' 'closure_typecheck_exit_code=0' 'quotient_typecheck_exit_code=0' \
  'target03_check_exit_code=0' 'axiom_reports=71' 'axiom_bearing=68' 'axiom_free=3' \
  'axiom_allowlist=Classical.choice,Quot.sound,propext' 'python_guardian_rc=110' \
  'python_dispatch_after_denial=ABSENT_BY_CONTROL_FLOW' 'guardian_false_exit=1' \
  "live_hardware_manifest_sha256=${LIVE_HARDWARE_MANIFEST_SHA256}" 'spark_route_policy=KUBERNETES_ONLY' \
  'u250_inventory=2,1,1,0' 'result=TARGET03_CLOSED_PARITY_OPEN_CLAIM_NOT_READY')"
[[ "$(sha_text "${runtime_evidence_material}")" == "${RUNTIME_EVIDENCE_MATERIAL_SHA256}" ]] || fail 'runtime evidence material drift'

printf 'PIREUS_CONCRETE_QUOTIENT_ACTION_RESULT=TARGET03_CLOSED_PARITY_OPEN_CLAIM_NOT_READY stage=PARITY_OPEN language=Lean4 role=FORMAL_PARITY semantic_authority=Sounio source_commit=%s exact_bitvec_lex=true action_count_40320=true action_closure=true gauge_word_type=Fin_2_POW_11 gauge_word_count=2048 normalization_absorption=true quotient_action_laws=true target03_scope=DECLARED_GL4_SWAP_GAUGE_ORBIT_IFF executed_sounio_streaming_equality=false formal_parity_complete=false computable_extraction_claim=false choice_free_core_claim=false guardian_role=PREEXECUTION_POLICY axiom_reports=71 axiom_free=3 axiom_allowlist=Classical.choice,Quot.sound,propext unexpected_axioms=0 native_decide=0 sorryax=0 python_dispatch=REFUSED_PREEXEC_E110 guardian_current_gate_false_exit=1 spark_route_policy=KUBERNETES_ONLY spark_dispatches_by_this_gate=0 dgx_dispatches_by_this_gate=0 slurm_dispatches_by_this_gate=0 live_cluster_usage=NOT_MEASURED u250_inventory_source=HASH_BOUND_SOUNIO_BASE_GATE u250_declared=2 u250_installed=1 u250_pending_installation=1 u250_enumeration_failures=0 u250_dispatches_by_this_gate=0 llm_role=REVIEW_ONLY llm_confirmed_result=false zai=ERROR_1313 novelty_confirmed=false claim_ready=false runtime_evidence_sha256=%s\n' "${SOURCE_COMMIT}" "${RUNTIME_EVIDENCE_MATERIAL_SHA256}"
