#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"

SOUNIO_REL='stdlib/hardware/pireus/quotient_novelty_forge.sio'
FREEZE_REL='tools/pireus/quotient_novelty_forge.freeze.v5'
PARITY_OPEN_REL='tools/pireus/quotient_novelty_forge.parity-open.v5'
LEAN_REL='formal/lean4/SounioPireusQuotientNoveltyForge.lean'
LAKE_REL='formal/lean4/lakefile.lean'
RECEIPT_REL='tools/pireus/quotient_novelty_forge.formal-parity.v5'
EVIDENCE_REL='tools/pireus/evidence/quotient_novelty_forge_v5.lean.txt'
OLEAN_REL='formal/lean4/.lake/build/lib/lean/SounioPireusQuotientNoveltyForge.olean'
PARENT_GATE_REL='scripts/ci/pireus_quotient_novelty_forge.sh'

SOUNIO_SHA256='791d85d4b336d854c6ed3b2e662e8f09b05f8a6f6d1dc4c03807c87150751667'
SEMANTICS_SHA256='9dde3079369c2c90f66805c18abf93a302f5a1d5facf909e39292283ed65bb21'
FREEZE_SHA256='640a271bbe1966a3993e72be8fe019b1152530372cfb3ab91ede92011c0fc8c7'
PARITY_OPEN_SHA256='108ac3dd8df394e01a5a3293aab8d9fe312d522245ed2ee02e8bc5db37fa2943'
LEAN_SHA256='2af97e4c949c5187cfdc532d28a543a59d0423cdea63aa7c9f3f9b258c1152b0'
LAKE_SHA256='1dcd62e5f436c5d92d0a7f5b6429169a7900b3d8dd9d64bd719288ebc222dcb8'
OLEAN_SHA256='282bbb0f0c57e4e58798a2fe6c833492649b26e1e62119504b50d9952af04b1d'
EVIDENCE_SHA256='f8ccad01b0d6cd4c70ca65492aca15544fc156332d315ff1409ff3377892fbff'
RECEIPT_SHA256='cff661497206523f273613e07fd8455ba7f036c62853b3b978c1bc29aa527593'
PARENT_GATE_SHA256='1bc9f27cea5a9f4e36a213efae17753b172db101e218a5982c6c3c674d70e29f'
TOOLCHAIN_SHA256='03526d62a2416b90e4cad0c369f73bbed9f38f700e0182f220f511235548bf63'
HARDWARE_SHA256='ba649e3df628e482654987fc29b86489bfdfc9f69f6bfdc141f3a8d238ee33cd'
COMMAND_SHA256='17c3965be2c9c8eedd6bc5400d1c17189928fc85776ee75e2d01da44fbb6bb11'
PREEXEC_FRAME_SHA256='4d78ad897aaa67949c58cd3a568337d67611e677a4bceab5f1b2d36c206dc944'
SEAL_FRAME_SHA256='4006e493e313dd67d9fe8e838bcca5bd77efec8a86567252482e9497276e3e36'

fail() {
  printf 'pireus quotient novelty formal parity: FAIL: %s\n' "$*" >&2
  exit 1
}

sha_file() { sha256sum "$1" | cut -d' ' -f1; }
sha_text() { printf '%s\n' "$1" | sha256sum | cut -d' ' -f1; }

sha_limbs() {
  local hex="$1" out='' i part
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
  grep -Fqx -- "${expected}" "${path}" || fail "missing exact line in ${path}: ${expected}"
}

parity_frame() {
  local action="$1" semantic_write="$2" parent_hash="$3" result_hash="$4"
  local receipt_valid=0
  [[ "${action}" == 8 ]] && receipt_valid=1
  printf '9020 %s %s 2 2 1 %s 0 %s 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$([[ "${action}" == 4 ]] && printf 3 || printf 4)" \
    "${action}" "${semantic_write}" "${receipt_valid}" \
    "$(sha_limbs "${SOUNIO_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${parent_hash}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${COMMAND_SHA256}")" \
    "$([[ "${result_hash}" == zero ]] && printf '0 0 0 0 0 0 0 0' || sha_limbs "${result_hash}")" \
    '0 0 0 0 0 0 0 0'
}

authorize() {
  local label="$1" frame="$2" expected_sha="$3" expected_rc="$4" expected="$5"
  local decision rc
  [[ "$(sha_text "${frame}")" == "${expected_sha}" ]] || fail "Guardian frame drift: ${label}"
  set +e
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  rc=$?
  set -e
  [[ "${rc}" -eq "${expected_rc}" ]] || fail "Guardian exit drift for ${label}: ${rc}"
  [[ "${decision}" == "${expected}" ]] || fail "Guardian decision drift for ${label}: ${decision}"
  printf 'GUARDIAN_DECISION label=%s decision=%s\n' "${label}" "${decision}"
}

require_hash "${ROOT}/${SOUNIO_REL}" "${SOUNIO_SHA256}"
require_hash "${ROOT}/${FREEZE_REL}" "${FREEZE_SHA256}"
require_hash "${ROOT}/${PARITY_OPEN_REL}" "${PARITY_OPEN_SHA256}"
require_hash "${ROOT}/${LEAN_REL}" "${LEAN_SHA256}"
require_hash "${ROOT}/${LAKE_REL}" "${LAKE_SHA256}"
require_hash "${ROOT}/${EVIDENCE_REL}" "${EVIDENCE_SHA256}"
require_hash "${ROOT}/${RECEIPT_REL}" "${RECEIPT_SHA256}"
require_hash "${ROOT}/${PARENT_GATE_REL}" "${PARENT_GATE_SHA256}"
[[ -x "${GUARDIAN}" ]] || fail 'native Loom Guardian unavailable'

if grep -Eq '\b(sorry|axiom)\b' "${ROOT}/${LEAN_REL}"; then
  fail 'Lean parity contains sorry or axiom'
fi
for theorem in \
  formal_parity_summary_matches_frozen_sounio \
  gauge_kernel_dimension_and_normalizer_uniqueness \
  parent_stabilizer_in_GL4_x_C2_group_law_inverse_and_action_equivariance \
  Q0_Q1_Q2_equivalence_and_refinement \
  canonical_partition_and_witness_soundness; do
  grep -Fq "theorem ${theorem}" "${ROOT}/${LEAN_REL}" || fail "missing theorem: ${theorem}"
done

require_line "${ROOT}/${FREEZE_REL}" "frozen_source_sha256=${SOUNIO_SHA256}"
require_line "${ROOT}/${FREEZE_REL}" "semantics_sha256=${SEMANTICS_SHA256}"
require_line "${ROOT}/${PARITY_OPEN_REL}" 'stage=PARITY_OPEN'
require_line "${ROOT}/${PARITY_OPEN_REL}" 'lean_status=OPEN_NOT_EXECUTED'

parent_gate_output="$("${ROOT}/${PARENT_GATE_REL}")"
printf '%s\n' "${parent_gate_output}" | grep -Fqx -- \
  'pireus quotient novelty forge: STAGE_REACHED_NOT_A_CLAIM stage=PARITY_OPEN language=Sounio admitted_actions=12 q0_classes=48 q1_classes=48 q2_classes=14 targets=4 unresolved=1920 selected_child=-1 formal=OPEN_NOT_EXECUTED effect=OPEN_NOT_EXECUTED material=OPEN_NOT_EXECUTED claim_ready=false python_process_launched=false rust_process_launched=false' ||
  fail 'Sounio authority gate terminal marker drift'

lean_version="$(lean --version | sed -n '1p')"
lake_version="$(cd "${ROOT}/formal/lean4" && lake --version)"
toolchain_record="lean=${lean_version} lake=${lake_version}"
[[ "$(sha_text "${toolchain_record}")" == "${TOOLCHAIN_SHA256}" ]] || fail 'Lean toolchain drift'
hardware_record="host=$(hostname) arch=$(uname -m) kernel=$(uname -s) $(uname -r) online_cpus=$(getconf _NPROCESSORS_ONLN) model=$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -n 1)"
[[ "$(sha_text "${hardware_record}")" == "${HARDWARE_SHA256}" ]] || fail 'hardware drift'
command_record='cd formal/lean4 && lake build SounioPireusQuotientNoveltyForge'
[[ "$(sha_text "${command_record}")" == "${COMMAND_SHA256}" ]] || fail 'command drift'

authorize PREEXEC "$(parity_frame 4 0 "${SEMANTICS_SHA256}" zero)" \
  "${PREEXEC_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'

wrong_parent="0dde3079369c2c90f66805c18abf93a302f5a1d5facf909e39292283ed65bb21"
wrong_parent_frame="$(parity_frame 4 0 "${wrong_parent}" zero)"
set +e
wrong_parent_decision="$(printf '%s\n' "${wrong_parent_frame}" | "${GUARDIAN}")"
wrong_parent_rc=$?
set -e
[[ "${wrong_parent_rc}" -eq 117 ]] || fail 'parent laundering did not fail closed'
[[ "${wrong_parent_decision}" == \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=117 reason=parent-semantics-hash-mismatch next_stage=SEMANTICS_FROZEN' ]] ||
  fail "parent laundering decision drift: ${wrong_parent_decision}"
printf 'GUARDIAN_DISPATCH label=PARENT_LAUNDERING process_launched=false\n'

semantic_write_frame="$(parity_frame 4 1 "${SEMANTICS_SHA256}" zero)"
set +e
semantic_write_decision="$(printf '%s\n' "${semantic_write_frame}" | "${GUARDIAN}")"
semantic_write_rc=$?
set -e
[[ "${semantic_write_rc}" -eq 113 ]] || fail 'Lean semantic write did not fail closed'
[[ "${semantic_write_decision}" == \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=SEMANTICS_FROZEN' ]] ||
  fail 'Lean semantic write decision drift'
printf 'GUARDIAN_DISPATCH label=LEAN_SEMANTIC_WRITE process_launched=false\n'

(cd "${ROOT}/formal/lean4" && lake build SounioPireusQuotientNoveltyForge)
require_hash "${ROOT}/${OLEAN_REL}" "${OLEAN_SHA256}"

authorize SEAL "$(parity_frame 8 0 "${SEMANTICS_SHA256}" "${OLEAN_SHA256}")" \
  "${SEAL_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'

require_line "${ROOT}/${RECEIPT_REL}" 'status=FORMAL_PARITY_COMPLETE'
require_line "${ROOT}/${RECEIPT_REL}" 'producing_language=Lean_4'
require_line "${ROOT}/${RECEIPT_REL}" 'producing_role=FORMAL_PARITY'
require_line "${ROOT}/${RECEIPT_REL}" "sounio_source_sha256=${SOUNIO_SHA256}"
require_line "${ROOT}/${RECEIPT_REL}" "sounio_semantics_sha256=${SEMANTICS_SHA256}"
require_line "${ROOT}/${RECEIPT_REL}" "result_sha256=${OLEAN_SHA256}"
require_line "${ROOT}/${RECEIPT_REL}" 'formal_parity_complete=true'
require_line "${ROOT}/${RECEIPT_REL}" 'effect_parity_complete=false'
require_line "${ROOT}/${RECEIPT_REL}" 'material_parity_complete=false'
require_line "${ROOT}/${RECEIPT_REL}" 'semantic_write=false'
require_line "${ROOT}/${RECEIPT_REL}" 'expected_result_write=false'
require_line "${ROOT}/${RECEIPT_REL}" 'claim_ready=false'

require_line "${ROOT}/${EVIDENCE_REL}" 'discharged_obligations=4'
require_line "${ROOT}/${EVIDENCE_REL}" 'admitted_actions=12'
require_line "${ROOT}/${EVIDENCE_REL}" 'q0_classes=48'
require_line "${ROOT}/${EVIDENCE_REL}" 'q1_classes=48'
require_line "${ROOT}/${EVIDENCE_REL}" 'q2_classes=14'
require_line "${ROOT}/${EVIDENCE_REL}" 'witnesses_sound=true'
require_line "${ROOT}/${EVIDENCE_REL}" 'selected_child=-1'
require_line "${ROOT}/${EVIDENCE_REL}" 'claim_ready=false'

printf '%s\n' \
  'pireus quotient novelty formal parity: STAGE_REACHED_NOT_A_CLAIM stage=PARITY_OPEN language=Lean4 role=FORMAL_PARITY obligations=4/4 admitted_actions=12 q0_classes=48 q1_classes=48 q2_classes=14 formal=COMPLETE effect=OPEN_NOT_EXECUTED material=OPEN_NOT_EXECUTED selected_child=-1 claim_ready=false python_process_launched=false rust_process_launched=false'
