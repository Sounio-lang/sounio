#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN_CANDIDATE="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
GUARDIAN="$(readlink -f "${GUARDIAN_CANDIDATE}")"

SOURCE_REL='formal/lean4/SounioPireusOperatorOrbitArchiveReconstruction.lean'
AUDIT_REL='formal/lean4/SounioPireusOperatorOrbitArchiveReconstructionAxiomAudit.lean'
LAKEFILE_REL='formal/lean4/lakefile.lean'
PARENT_RECEIPT_REL='tools/pireus/operator_orbit_canonicalization.formal-parity.v13'
PARENT_GATE_REL='scripts/ci/pireus_operator_orbit_canonicalization_formal_parity.sh'
PARENT_EVIDENCE_REL='tools/pireus/evidence/operator_orbit_canonicalization_v13.formal-parity.txt'
RECEIPT_REL='tools/pireus/operator_orbit_archive_reconstruction.formal-parity.v13'

PARENT_COMMIT='ca4bf9023fa379daf37a063f8dd9c2071e9c4d5e'
SOURCE_COMMIT='fbef7df8266d26fdc6f0d6a07b6e00ccc24eb8ef'
SOURCE_SHA256='d52010927d1e6209aa8033329156bc2dfe6f0b2740da56c7277b08408ac40399'
AUDIT_SHA256='102ced949057ca15ca9c2de8ad30715a2749ff5e4ae136ef97bc008923d05479'
LAKEFILE_SHA256='c4a8bff4cb01c216a6f7f235b66f7cd8b9fe24287c0d86d931aaf8055afe1a7d'
PARENT_RECEIPT_SHA256='8ba5fd22f677b12af33f3269fbc2c78851720ba851c76424d96d092b9da3e871'
PARENT_GATE_SHA256='1439e5a63a2a68ecf64931f40fb38f151a3673ee1c2b0d51c7a4553f6a8134be'
PARENT_EVIDENCE_SHA256='3aa6de97edd7966aed1505e0132aa3295e273bd6d7f3548fa4812ef55fb3ce95'
RECEIPT_SHA256='98f736dcfc947e5669d8c2a1329afc3f4d12fa16ae07d0631152fe5e115e962b'
SOUNIO_SEMANTICS_SHA256='0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c'
TOOLCHAIN_SHA256='03526d62a2416b90e4cad0c369f73bbed9f38f700e0182f220f511235548bf63'
HARDWARE_SHA256='ba649e3df628e482654987fc29b86489bfdfc9f69f6bfdc141f3a8d238ee33cd'
BUILD_COMMAND_SHA256='794b0a4153550ec9d317b18307e8bf6d411145cf0356ed3e87e0a66b705873aa'
AUDIT_COMMAND_SHA256='0e8e8b4b0973e7a6d732b36d75376a0cbaa193afdb2b988732ed938028726097'
BUILD_FRAME_SHA256='29ffe0e6d51f4b20e43ab68209f2b617656c1d181f8281a9c0360c0d27952c69'
AUDIT_FRAME_SHA256='7b466da0db3a30ba70ff29cd0f341801b3c61b25afba904b09687f0752c142ad'
BUILD_OUTPUT_SHA256='ef46bed6b21ea71fb7e89c5ccd3a826b3738f2ece5246cd69f34eb33cb9184ba'
OLEAN_SHA256='3c30475f7f2071d3d220dc6eb607191a9b124a4714d95bea76265086a26d6616'
AUDIT_OUTPUT_SHA256='dc293c7e903763defcc96a12d6167aff4190384ac1a71752c5580a8c066a71db'
ZERO='0 0 0 0 0 0 0 0'
EXPECTED_THEOREMS=(
  concrete_archive_reconstruction_matches_declared_frozen_summary
  reconstructed_archive_has_exactly_128_concrete_tables
  forty_eight_cubic_children_reconstruct_ninety_six_action_images
  sixteen_fresh_paired_epochs_reconstruct_128_images
  archive_reconstruction_carries_declared_frozen_hash_literals
  archive_reconstruction_does_not_close_class_parity
)

fail() {
  printf 'pireus orbit archive reconstruction formal parity: FAIL: %s\n' "$*" >&2
  exit 1
}

sha_file() { sha256sum "$1" | cut -d' ' -f1; }
sha_text() { printf '%s\n' "$1" | sha256sum | cut -d' ' -f1; }

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
    "$(sha_limbs "${source_sha}")" \
    "$(sha_limbs "${SOUNIO_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SOUNIO_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${command_sha}")" "${ZERO}" "${ZERO}"
}

check_guardian() {
  local label="$1" frame="$2" expected_sha="$3" decision
  [[ "$(wc -w <<<"${frame}" | tr -d ' ')" -eq 82 ]] || fail "Guardian frame words: ${label}"
  [[ "$(sha_text "${frame}")" == "${expected_sha}" ]] || fail "Guardian frame drift: ${label}"
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  [[ "${decision}" == 'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN' ]] ||
    fail "Guardian decision drift: ${label}: ${decision}"
  printf 'GUARDIAN_DECISION label=%s decision=%s\n' "${label}" "${decision}"
}

"${ROOT}/scripts/ci/pireus_operator_orbit_canonicalization.sh" >/dev/null

require_hash "${ROOT}/${SOURCE_REL}" "${SOURCE_SHA256}"
require_hash "${ROOT}/${AUDIT_REL}" "${AUDIT_SHA256}"
require_hash "${ROOT}/${LAKEFILE_REL}" "${LAKEFILE_SHA256}"
require_hash "${ROOT}/${PARENT_RECEIPT_REL}" "${PARENT_RECEIPT_SHA256}"
require_hash "${ROOT}/${PARENT_GATE_REL}" "${PARENT_GATE_SHA256}"
require_hash "${ROOT}/${PARENT_EVIDENCE_REL}" "${PARENT_EVIDENCE_SHA256}"
require_hash "${ROOT}/${RECEIPT_REL}" "${RECEIPT_SHA256}"
git -C "${ROOT}" merge-base --is-ancestor "${PARENT_COMMIT}" "${SOURCE_COMMIT}" ||
  fail 'parent structural parity does not precede archive reconstruction'
git -C "${ROOT}" merge-base --is-ancestor "${SOURCE_COMMIT}" HEAD ||
  fail 'archive reconstruction source commit is not in current history'
require_committed_hash "${PARENT_COMMIT}" "${PARENT_RECEIPT_REL}" "${PARENT_RECEIPT_SHA256}"
require_committed_hash "${PARENT_COMMIT}" "${PARENT_GATE_REL}" "${PARENT_GATE_SHA256}"
require_committed_hash "${PARENT_COMMIT}" "${PARENT_EVIDENCE_REL}" "${PARENT_EVIDENCE_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${SOURCE_REL}" "${SOURCE_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${AUDIT_REL}" "${AUDIT_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${LAKEFILE_REL}" "${LAKEFILE_SHA256}"

require_line "${ROOT}/${RECEIPT_REL}" 'status=PARTIAL_PASS'
require_line "${ROOT}/${RECEIPT_REL}" 'producing_language=Lean4'
require_line "${ROOT}/${RECEIPT_REL}" 'language_role=FORMAL_PARITY'
require_line "${ROOT}/${RECEIPT_REL}" 'initial_action_images_reconstructed=96'
require_line "${ROOT}/${RECEIPT_REL}" 'generated_epochs_reconstructed=16'
require_line "${ROOT}/${RECEIPT_REL}" 'final_concrete_tables_reconstructed=128'
require_line "${ROOT}/${RECEIPT_REL}" 'concrete_128_image_census_complete=true'
require_line "${ROOT}/${RECEIPT_REL}" 'dependency_fidelity_scope=IMPORTED_FORMAL_PARITY_DEFINITIONS_AND_DECLARED_HASH_LITERALS_NOT_EXTERNAL_FILE_HASH_PROOF'
require_line "${ROOT}/${RECEIPT_REL}" 'concrete_30_class_reconstruction_proved=false'
require_line "${ROOT}/${RECEIPT_REL}" 'canonical_representative_equality_iff_same_declared_orbit_proved=false'
require_line "${ROOT}/${RECEIPT_REL}" 'formal_parity_complete=false'
require_line "${ROOT}/${RECEIPT_REL}" 'commit_command_sha256=820d2bd9c59439f0603f5b227b35247160d44a8f13388d097804986c12526069'
require_line "${ROOT}/${RECEIPT_REL}" 'commit_frame_sha256=b80b998bfbeb42974e04bd6f33bb83e2d310f3207b30dc8ed88f06b0bceee52d'
require_line "${ROOT}/${RECEIPT_REL}" 'commit_decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
require_line "${ROOT}/${RECEIPT_REL}" 'claim_ready=false'
require_line "${ROOT}/${RECEIPT_REL}" 'python_processes_launched=0'
require_line "${ROOT}/${RECEIPT_REL}" 'rust_processes_launched=0'
require_line "${ROOT}/${RECEIPT_REL}" 'spark_route_policy=KUBERNETES_ONLY'
require_line "${ROOT}/${RECEIPT_REL}" 'u250_declared_card_count=2'
require_line "${ROOT}/${RECEIPT_REL}" 'u250_installed_card_count=1'
require_line "${ROOT}/${RECEIPT_REL}" 'u250_pending_installation_card_count=1'

[[ "$(grep -c '^theorem ' "${ROOT}/${SOURCE_REL}")" -eq 6 ]] || fail 'theorem count drift'
[[ "$(grep -c '^#print axioms ' "${ROOT}/${AUDIT_REL}")" -eq 6 ]] || fail 'axiom audit count drift'
for theorem_name in "${EXPECTED_THEOREMS[@]}"; do
  grep -Eq "^theorem ${theorem_name}([[:space:]]|:|$)" "${ROOT}/${SOURCE_REL}" ||
    fail "missing Lean theorem declaration: ${theorem_name}"
  require_line "${ROOT}/${AUDIT_REL}" "#print axioms ${theorem_name}"
done
if grep -Eq '\bsorry\b|sorryAx' "${ROOT}/${SOURCE_REL}" "${ROOT}/${AUDIT_REL}"; then
  fail 'sorry or sorryAx found in archive reconstruction proof surface'
fi

(( 48 * 2 == 96 )) || fail 'initial action-image arithmetic drift'
(( 15 * 15 == 225 )) || fail 'interior-cell arithmetic drift'
(( 96 + 2 * 16 == 128 )) || fail 'archive growth arithmetic drift'

TMPDIR_PIREUS="$(mktemp -d)"
trap 'rm -rf "${TMPDIR_PIREUS}"' EXIT
check_guardian BUILD "$(parity_frame "${SOURCE_SHA256}" "${BUILD_COMMAND_SHA256}")" "${BUILD_FRAME_SHA256}"
(
  cd "${ROOT}/formal/lean4"
  lake build SounioPireusOperatorOrbitArchiveReconstruction >"${TMPDIR_PIREUS}/build.txt" 2>&1
)
require_hash "${TMPDIR_PIREUS}/build.txt" "${BUILD_OUTPUT_SHA256}"
require_hash "${ROOT}/formal/lean4/.lake/build/lib/lean/SounioPireusOperatorOrbitArchiveReconstruction.olean" "${OLEAN_SHA256}"

check_guardian AXIOM_AUDIT "$(parity_frame "${AUDIT_SHA256}" "${AUDIT_COMMAND_SHA256}")" "${AUDIT_FRAME_SHA256}"
(
  cd "${ROOT}/formal/lean4"
  lake env lean -j 1 SounioPireusOperatorOrbitArchiveReconstructionAxiomAudit.lean >"${TMPDIR_PIREUS}/axioms.txt"
)
require_hash "${TMPDIR_PIREUS}/axioms.txt" "${AUDIT_OUTPUT_SHA256}"
[[ "$(grep -c ' depends on axioms:' "${TMPDIR_PIREUS}/axioms.txt")" -eq 6 ]] ||
  fail 'axiom-bearing report count drift'
[[ "$(grep -c 'propext' "${TMPDIR_PIREUS}/axioms.txt")" -eq 6 ]] ||
  fail 'propext report count drift'
[[ "$(grep -o '_native\.native_decide\.ax_1_1' "${TMPDIR_PIREUS}/axioms.txt" | wc -l)" -eq 6 ]] ||
  fail 'native_decide report count drift'
if grep -q 'sorryAx' "${TMPDIR_PIREUS}/axioms.txt"; then
  fail 'sorryAx appeared in axiom audit'
fi

printf '%s\n' \
  'PIREUS_OPERATOR_ORBIT_ARCHIVE_RECONSTRUCTION_FORMAL_PARITY_PASS=true status=PARTIAL_PASS language=Lean4 role=FORMAL_PARITY cubic_children=48 initial_action_images=96 epochs=16 final_concrete_tables=128 interior_cells=225 theorem_reports=6 propext_mentions=6 native_decide_mentions=6 sorryax_mentions=0 concrete_128_image_census_complete=true concrete_30_class_reconstruction=false canonical_iff_orbit=false formal_parity_complete=false semantic_authority=Sounio expected_results_supplied_by_lean=false spark_route=KUBERNETES_ONLY spark_nodes_used=false u250_declared=2 u250_installed=1 u250_pending_installation=1 claim_ready=false'
