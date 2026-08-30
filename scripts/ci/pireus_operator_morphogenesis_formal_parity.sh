#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"

SOUNIO_REL='stdlib/hardware/pireus/operator_morphogenesis.sio'
FREEZE_REL='tools/pireus/operator_morphogenesis.freeze.v12'
PARITY_OPEN_REL='tools/pireus/operator_morphogenesis.parity-open.v12'
LEAN_REL='formal/lean4/SounioPireusOperatorMorphogenesis.lean'
AUDIT_REL='formal/lean4/SounioPireusOperatorMorphogenesisAxiomAudit.lean'
LAKE_REL='formal/lean4/lakefile.lean'
EVIDENCE_REL='tools/pireus/evidence/operator_morphogenesis_v12.lean.txt'
RECEIPT_REL='tools/pireus/operator_morphogenesis.formal-parity.v12'
OLEAN_REL='formal/lean4/.lake/build/lib/lean/SounioPireusOperatorMorphogenesis.olean'
ILEAN_REL='formal/lean4/.lake/build/lib/lean/SounioPireusOperatorMorphogenesis.ilean'

SOUNIO_SHA256='0a637f7f3ac84ac501be337f22dff37e16a05dbc4a51d2090441b9cba4c8d05c'
FREEZE_SHA256='14277a28f21a044bd55bd670b5b7447789c2f4e2780251c861ee4880ef739de7'
PARITY_OPEN_SHA256='b3cc6a9e67471c61eab5d42d103b21a3874ade3e6dd9a340dd8856dea4bd2909'
SEMANTIC_SOURCE_MANIFEST_SHA256='fdb08abdd1a689dd8a3c50ae9ad16948f6de42137e8c2660fe38b1450f9e3cdf'
SEMANTICS_SHA256='999c6e7a0051f702cf40bb2adab7dc91c4f026230830096377f525005067c2f4'
FORMAL_SOURCE_MANIFEST_SHA256='cf245714a49ba848a4497a9278f9f10334986beea2c88d0c4ff0c0ca3c9d4cd5'
LEAN_SHA256='8959e411abb8c3840ebfdadb62c49237aae16de6a8335635d88778103c87e273'
AUDIT_SHA256='ae780e265914895cae70aa67463410b4b2a884286874a3d39a73504a5f3f8f86'
LAKE_SHA256='080a316f8a3658f1408c32d13a30eee714aa38a2c8f6f8f55654826e66df6303'
EVIDENCE_SHA256='fd97194cd6c13d032a80df23c06b5e4e318f9fc49928147d44f3d1a2e48812c1'
RECEIPT_SHA256='0eb932b96838383a800f3889a331d16a10886621f29cda9c19e4e1ef74e0077c'
OLEAN_SHA256='200e990db4c447c0a0003e1f6afb6dcb97e54364463393f40c1e68c4673a3176'
ILEAN_SHA256='670ec5af77302b3168b47824024ade0414a46b35e5260cf1f2a100f89abd9f62'
TOOLCHAIN_SHA256='03526d62a2416b90e4cad0c369f73bbed9f38f700e0182f220f511235548bf63'
LEAN_BINARY_SHA256='19d38963260cfb376f1aab0f0fbcf4e80ec25c8bd0ba3b1797d95141d56ec55a'
LAKE_BINARY_SHA256='19d38963260cfb376f1aab0f0fbcf4e80ec25c8bd0ba3b1797d95141d56ec55a'
HARDWARE_SHA256='ba649e3df628e482654987fc29b86489bfdfc9f69f6bfdc141f3a8d238ee33cd'
BUILD_COMMAND_SHA256='e087daa649542a5d083072265b3f440b395c769c0153f06004734d4bba0d3770'
AUDIT_COMMAND_SHA256='e08645d8529e00c7aeba43781f979949ed233b0cd21a7b5603b9b53cb8601d2c'
WRITE_COMMAND_SHA256='7bd8207937a521de462600a144ce46dfd5159a58a4e7bc61451e522333afa3b5'
BUILD_FRAME_SHA256='3adcef2de1bde2e185979ff93a2738a706d186dba4ab4b3d1f53da2aca62a6c6'
AUDIT_FRAME_SHA256='89d909fbb7289079e9377da9269fd6ac90e08effa4493f7b0e077dd790435d98'
SEAL_FRAME_SHA256='d2f2bbe2f5fa87dd286c31203a32ce043de73a95b23fa4e7aba227f3b04e0070'
WRITE_FRAME_SHA256='663fe4d3b3fbb7ae11bcb8657aebe7bbea0793817903c3d16a1cabbc5aa78d79'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
GUARDIAN_POLICY_SHA256='64bb0118793fe46dcb392abc1a9212eb15bd55047461576a3ef1a6cefa3f17da'
PYTHON_TOOLCHAIN_SHA256='497ce0938df96d9bf3c159472d251a946c2d6bd832220937d1b885f7759b05ba'
PYTHON_COMMAND_SHA256='72566473f0019fb50e65175a0ac019af5ad2f495ccfc30b664b7735791e968fc'
RUST_TOOLCHAIN_SHA256='b58640570ba9ffcdb2c2d241e4ce8ece9c7d75c6b1e59e308dee3e5f0e10b56d'
RUST_COMMAND_SHA256='0421c7020d972449b7a1f0492945c18a10be46f6a16ade7f8efad157e2e01b00'
ZERO='0 0 0 0 0 0 0 0'
FRAME_WORDS=82

fail() {
  printf 'pireus operator morphogenesis formal parity: FAIL: %s\n' "$*" >&2
  exit 1
}

sha_file() { sha256sum "$1" | cut -d' ' -f1; }
sha_text() { printf '%s\n' "$1" | sha256sum | cut -d' ' -f1; }

sha_limbs() {
  local hex="$1" out='' i part
  [[ "${#hex}" -eq 64 && "${hex}" =~ ^[0-9a-f]{64}$ ]] ||
    fail "invalid sha256 digest: ${hex}"
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
  local path="$1" expected="$2"
  rg -Fqx -- "${expected}" "${path}" ||
    fail "missing exact line in ${path}: ${expected}"
}

authority_frame() {
  local stage="$1" action="$2" language="$3" role="$4" policy="$5"
  local semantic_write="$6" expected_write="$7" parity_valid="$8"
  local review_promoted="$9" parent_hash="${10}" toolchain_hash="${11}"
  local command_hash="${12}" result_hash="${13}" result_limbs="${ZERO}"
  if [[ "${result_hash}" != zero ]]; then
    result_limbs="$(sha_limbs "${result_hash}")"
  fi
  printf '9020 %s %s %s %s %s %s %s %s %s 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${stage}" "${action}" "${language}" "${role}" "${policy}" \
    "${semantic_write}" "${expected_write}" "${parity_valid}" \
    "${review_promoted}" "$(sha_limbs "${SEMANTIC_SOURCE_MANIFEST_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" "$(sha_limbs "${parent_hash}")" \
    "$(sha_limbs "${toolchain_hash}")" "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${command_hash}")" "${result_limbs}" "${ZERO}"
}

authorize() {
  local label="$1" frame="$2" expected_hash="$3" expected_rc="$4" expected="$5"
  local decision rc words
  words="$(wc -w <<<"${frame}" | tr -d ' ')"
  [[ "${words}" -eq "${FRAME_WORDS}" ]] || fail "${label}: frame words=${words}"
  [[ "$(sha_text "${frame}")" == "${expected_hash}" ]] ||
    fail "${label}: frame hash drift"
  set +e
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  rc=$?
  set -e
  [[ "${rc}" -eq "${expected_rc}" ]] || fail "${label}: rc=${rc}"
  [[ "${decision}" == "${expected}" ]] || fail "${label}: ${decision}"
  printf 'GUARDIAN_DECISION label=%s frame_sha256=%s rc=%s %s\n' \
    "${label}" "${expected_hash}" "${rc}" "${decision}"
}

deny_without_dispatch() {
  local label="$1" frame="$2" expected_rc="$3" expected="$4"
  authorize "${label}" "${frame}" "$(sha_text "${frame}")" "${expected_rc}" "${expected}"
  printf 'GUARDIAN_DISPATCH label=%s process_launched=false\n' "${label}"
}

receipt_admitted() {
  local path="$1" key
  for key in status stage formal_obligations formal_obligations_discharged \
    formal_parity_complete effect_parity_complete material_parity_complete \
    algebraic_novelty algorithmic_novelty material_novelty historical_novelty \
    priority_claim claim_ready; do
    [[ "$(rg -c "^${key}=" "${path}")" -eq 1 ]] || return 1
  done
  rg -Fqx 'status=FORMAL_PARITY_COMPLETE' "${path}" &&
    rg -Fqx 'stage=PARITY_OPEN' "${path}" &&
    rg -Fqx 'formal_obligations=6' "${path}" &&
    rg -Fqx 'formal_obligations_discharged=6' "${path}" &&
    rg -Fqx 'formal_parity_complete=true' "${path}" &&
    rg -Fqx 'effect_parity_complete=false' "${path}" &&
    rg -Fqx 'material_parity_complete=false' "${path}" &&
    rg -Fqx 'algebraic_novelty=false' "${path}" &&
    rg -Fqx 'algorithmic_novelty=false' "${path}" &&
    rg -Fqx 'material_novelty=false' "${path}" &&
    rg -Fqx 'historical_novelty=false' "${path}" &&
    rg -Fqx 'priority_claim=false' "${path}" &&
    rg -Fqx 'claim_ready=false' "${path}"
}

cd "${ROOT}"
require_hash "${ROOT}/${SOUNIO_REL}" "${SOUNIO_SHA256}"
require_hash "${ROOT}/${FREEZE_REL}" "${FREEZE_SHA256}"
require_hash "${ROOT}/${PARITY_OPEN_REL}" "${PARITY_OPEN_SHA256}"
require_hash "${ROOT}/${LEAN_REL}" "${LEAN_SHA256}"
require_hash "${ROOT}/${AUDIT_REL}" "${AUDIT_SHA256}"
require_hash "${ROOT}/${LAKE_REL}" "${LAKE_SHA256}"
require_hash "${ROOT}/${EVIDENCE_REL}" "${EVIDENCE_SHA256}"
require_hash "${ROOT}/${RECEIPT_REL}" "${RECEIPT_SHA256}"
require_hash "${GUARDIAN}" "${GUARDIAN_SHA256}"
require_hash "${ROOT}/stdlib/coordination/loom_language_authority.sio" "${GUARDIAN_POLICY_SHA256}"
[[ -x "${GUARDIAN}" ]] || fail 'native Loom Guardian unavailable'

formal_manifest="$({
  cd "${ROOT}"
  sha256sum "${LEAN_REL}" "${AUDIT_REL}" "${LAKE_REL}"
})"
[[ "$(sha_text "${formal_manifest}")" == "${FORMAL_SOURCE_MANIFEST_SHA256}" ]] ||
  fail 'formal source manifest drift'

lean_version="$(lean --version | sed -n '1p')"
lake_version="$(cd "${ROOT}/formal/lean4" && lake --version)"
toolchain_record="lean=${lean_version} lake=${lake_version}"
[[ "$(sha_text "${toolchain_record}")" == "${TOOLCHAIN_SHA256}" ]] ||
  fail 'Lean toolchain record drift'
require_hash "$(command -v lean)" "${LEAN_BINARY_SHA256}"
require_hash "$(command -v lake)" "${LAKE_BINARY_SHA256}"
[[ "$(stat -Lc '%d:%i' "$(command -v lean)")" == \
   "$(stat -Lc '%d:%i' "$(command -v lake)")" ]] ||
  fail 'Lean and Lake are not the pinned Elan multicall hardlink'

cpu_model="$(lscpu | sed -n 's/^Model name:[[:space:]]*//p')"
hardware_record="host=$(hostname) arch=$(uname -m) kernel=$(uname -s) $(uname -r) online_cpus=$(getconf _NPROCESSORS_ONLN) model=${cpu_model}"
[[ "$(sha_text "${hardware_record}")" == "${HARDWARE_SHA256}" ]] ||
  fail 'formal hardware record drift'

require_line "${ROOT}/${FREEZE_REL}" "semantics_sha256=${SEMANTICS_SHA256}"
require_line "${ROOT}/${PARITY_OPEN_REL}" 'status=OPEN_NOT_EXECUTED'
require_line "${ROOT}/${PARITY_OPEN_REL}" 'stage=PARITY_OPEN'
require_line "${ROOT}/${PARITY_OPEN_REL}" 'lean_status=OPEN_NOT_EXECUTED'
require_line "${ROOT}/${PARITY_OPEN_REL}" 'spark_scheduler_route=KUBERNETES'
require_line "${ROOT}/${PARITY_OPEN_REL}" 'spark_slurm_route_allowed=false'
require_line "${ROOT}/${RECEIPT_REL}" 'spark_scheduler_route=KUBERNETES'
require_line "${ROOT}/${RECEIPT_REL}" 'spark_k8s_node_01=spark-3c59'
require_line "${ROOT}/${RECEIPT_REL}" 'spark_k8s_node_02=spark-8e54'
require_line "${ROOT}/${RECEIPT_REL}" 'spark_slurm_route_allowed=false'
require_line "${ROOT}/${RECEIPT_REL}" 'target_processes_launched=0'
require_line "${ROOT}/${RECEIPT_REL}" 'lean_lake_elan_multicall_same_inode=true'
require_line "${ROOT}/${RECEIPT_REL}" 'formal_obligation_01_theorems=interior_codec_is_a_225_cell_bijection'
require_line "${ROOT}/${RECEIPT_REL}" 'formal_obligation_02_theorems=mixed_nonempty_anf_mobius_matrix_is_self_inverse,mixed_nonempty_phase_extension_vanishes_on_both_axes'
require_line "${ROOT}/${RECEIPT_REL}" 'formal_obligation_03_theorems=list_index_complement_separates_every_prior_row'
require_line "${ROOT}/${RECEIPT_REL}" 'formal_obligation_04_theorems=actTable_involutive,paired_archive_is_closed_under_c2,transported_separator_sound,c2_involution_and_orbit_accounting_complete'
require_line "${ROOT}/${RECEIPT_REL}" 'formal_obligation_05_theorems=orbit_insertion_accounting_is_96_plus_2_times_16'
require_line "${ROOT}/${RECEIPT_REL}" 'formal_obligation_06_theorems=executable_certificate_scope_does_not_promote_classification'
require_line "${ROOT}/${RECEIPT_REL}" 'formal_binding_theorems=formal_parity_summary_matches_frozen_sounio,formal_parity_is_bound_to_frozen_sounio_hashes'
receipt_admitted "${ROOT}/${RECEIPT_REL}" || fail 'formal receipt admission failed'

if rg -n '(^|[^A-Za-z])(sorry|admit|axiom)([^A-Za-z]|$)' \
    "${ROOT}/${LEAN_REL}" "${ROOT}/${AUDIT_REL}" >/dev/null; then
  fail 'forbidden proof escape hatch found'
fi
require_line "${ROOT}/${LEAN_REL}" 'theorem mixed_nonempty_phase_extension_vanishes_on_both_axes'
require_line "${ROOT}/${LEAN_REL}" 'theorem actTable_involutive'
require_line "${ROOT}/${LEAN_REL}" 'theorem list_index_complement_separates_every_prior_row'
require_line "${ROOT}/${LEAN_REL}" 'theorem paired_archive_is_closed_under_c2'
require_line "${ROOT}/${LEAN_REL}" 'theorem transported_separator_sound'
require_line "${ROOT}/${LEAN_REL}" 'theorem formal_parity_summary_matches_frozen_sounio :'
require_line "${ROOT}/${LEAN_REL}" 'theorem formal_parity_is_bound_to_frozen_sounio_hashes :'
require_line "${ROOT}/${LEAN_REL}" 'theorem interior_codec_is_a_225_cell_bijection :'
require_line "${ROOT}/${LEAN_REL}" 'theorem mixed_nonempty_anf_mobius_matrix_is_self_inverse :'
require_line "${ROOT}/${LEAN_REL}" 'theorem c2_involution_and_orbit_accounting_complete :'
require_line "${ROOT}/${LEAN_REL}" 'theorem orbit_insertion_accounting_is_96_plus_2_times_16 :'
require_line "${ROOT}/${LEAN_REL}" 'theorem executable_certificate_scope_does_not_promote_classification :'
rg -F 'SounioPireusOperatorMorphogenesis.lean' \
  "${ROOT}/.claude/llm_offload_log.md" >/dev/null || fail 'missing mandatory math review log'

wrong_parent='a99c6e7a0051f702cf40bb2adab7dc91c4f026230830096377f525005067c2f4'
deny_without_dispatch PARENT_LAUNDERING \
  "$(authority_frame 3 4 2 2 1 0 0 0 0 "${wrong_parent}" "${TOOLCHAIN_SHA256}" "${BUILD_COMMAND_SHA256}" zero)" 117 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=117 reason=parent-semantics-hash-mismatch next_stage=SEMANTICS_FROZEN'
deny_without_dispatch LEAN_SEMANTIC_WRITE \
  "$(authority_frame 3 4 2 2 1 1 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${BUILD_COMMAND_SHA256}" zero)" 113 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=SEMANTICS_FROZEN'
deny_without_dispatch LEAN_EXPECTED_RESULT_WRITE \
  "$(authority_frame 3 4 2 2 1 0 1 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${BUILD_COMMAND_SHA256}" zero)" 114 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=114 reason=expected-result-authority-required next_stage=SEMANTICS_FROZEN'
deny_without_dispatch REVIEW_PROMOTION \
  "$(authority_frame 3 5 6 6 1 0 0 0 1 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${BUILD_COMMAND_SHA256}" zero)" 119 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=119 reason=review-promoted-to-authority next_stage=SEMANTICS_FROZEN'
deny_without_dispatch POLICY_MISSING \
  "$(authority_frame 3 4 2 2 0 0 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${BUILD_COMMAND_SHA256}" zero)" 101 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=101 reason=policy-missing next_stage=SEMANTICS_FROZEN'
deny_without_dispatch POLICY_TIMEOUT \
  "$(authority_frame 3 4 2 2 2 0 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${BUILD_COMMAND_SHA256}" zero)" 102 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=102 reason=policy-timeout next_stage=SEMANTICS_FROZEN'
deny_without_dispatch POLICY_ERROR \
  "$(authority_frame 3 4 2 2 3 0 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${BUILD_COMMAND_SHA256}" zero)" 103 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=103 reason=policy-error next_stage=SEMANTICS_FROZEN'
deny_without_dispatch PYTHON_ORACLE \
  "$(authority_frame 3 4 7 7 1 0 0 0 0 "${SEMANTICS_SHA256}" "${PYTHON_TOOLCHAIN_SHA256}" "${PYTHON_COMMAND_SHA256}" zero)" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
deny_without_dispatch RUST_ORACLE \
  "$(authority_frame 3 4 8 7 1 0 0 0 0 "${SEMANTICS_SHA256}" "${RUST_TOOLCHAIN_SHA256}" "${RUST_COMMAND_SHA256}" zero)" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
deny_without_dispatch WRONG_STAGE \
  "$(authority_frame 2 4 2 2 1 0 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${BUILD_COMMAND_SHA256}" zero)" 112 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=112 reason=wrong-stage next_stage=SOUNIO_EXECUTABLE'

authorize BUILD_PREEXEC \
  "$(authority_frame 3 4 2 2 1 0 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${BUILD_COMMAND_SHA256}" zero)" \
  "${BUILD_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
(cd "${ROOT}/formal/lean4" && lake build SounioPireusOperatorMorphogenesis)
require_hash "${ROOT}/${OLEAN_REL}" "${OLEAN_SHA256}"
require_hash "${ROOT}/${ILEAN_REL}" "${ILEAN_SHA256}"

authorize AXIOM_AUDIT_PREEXEC \
  "$(authority_frame 3 4 2 2 1 0 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${AUDIT_COMMAND_SHA256}" zero)" \
  "${AUDIT_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
axiom_output="$({
  cd "${ROOT}/formal/lean4"
  lake env lean -j 1 SounioPireusOperatorMorphogenesisAxiomAudit.lean 2>&1
})"
[[ "$(rg -c ' depends on axioms:' <<<"${axiom_output}")" -eq 10 ]] ||
  fail 'axiom-bearing theorem count drift'
[[ "$(rg -c 'does not depend on any axioms' <<<"${axiom_output}")" -eq 2 ]] ||
  fail 'axiom-free theorem count drift'
[[ "$(rg -c 'native_decide.ax_1_1' <<<"${axiom_output}")" -eq 6 ]] ||
  fail 'native_decide axiom profile drift'
[[ "$(rg -c 'Quot.sound' <<<"${axiom_output}")" -eq 3 ]] ||
  fail 'Quot.sound profile drift'
[[ "$(rg -c 'propext' <<<"${axiom_output}")" -eq 10 ]] ||
  fail 'propext profile drift'
if rg -F 'Classical.choice' <<<"${axiom_output}" >/dev/null; then
  fail 'unexpected Classical.choice dependency'
fi
printf 'LEAN_AXIOM_AUDIT theorems=12 propext=10 quot_sound=3 native_decide=6 axiom_free=2 unexpected=0\n'

authorize SEAL \
  "$(authority_frame 4 8 2 2 1 0 0 1 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${BUILD_COMMAND_SHA256}" "${OLEAN_SHA256}")" \
  "${SEAL_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
authorize RECEIPT_WRITE \
  "$(authority_frame 4 9 2 2 1 0 0 1 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${WRITE_COMMAND_SHA256}" "${OLEAN_SHA256}")" \
  "${WRITE_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'

require_hash "${ROOT}/${EVIDENCE_REL}" "${EVIDENCE_SHA256}"
require_hash "${ROOT}/${RECEIPT_REL}" "${RECEIPT_SHA256}"
receipt_admitted "${ROOT}/${RECEIPT_REL}" || fail 'formal receipt drifted during gate'

printf '%s\n' \
  'pireus operator morphogenesis formal parity: STAGE_REACHED_NOT_A_CLAIM stage=PARITY_OPEN language=Lean4 role=FORMAL_PARITY obligations=6/6 theorem_exports=12 interior=PINNED_LEAN_225_CELL_BIJECTION anf_matrix=PINNED_LEAN_SELF_INVERSE diagonal_separator=PINNED_LEAN_GENERIC c2_archive_closure=PINNED_LEAN_GENERIC accounting=PINNED_LEAN_96:16:128:1776:3552 concrete_archive_reconstructed=false formal=PINNED_LEAN_COMPLETE effect=OPEN_NOT_EXECUTED material=OPEN_NOT_EXECUTED spark_route=KUBERNETES spark_nodes=spark-3c59:spark-8e54 slurm_route_used=false algebraic_novelty=false material_novelty=false historical_novelty=false priority_claim=false claim_ready=false python_process_launched=false rust_process_launched=false target_processes_launched=0'
