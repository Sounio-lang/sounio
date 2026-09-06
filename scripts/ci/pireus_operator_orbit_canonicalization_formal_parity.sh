#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN_CANDIDATE="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
GUARDIAN="$(readlink -f "${GUARDIAN_CANDIDATE}")"

SOURCE_REL='formal/lean4/SounioPireusOperatorOrbitCanonicalization.lean'
AUDIT_REL='formal/lean4/SounioPireusOperatorOrbitCanonicalizationAxiomAudit.lean'
LAKEFILE_REL='formal/lean4/lakefile.lean'
PARITY_OPEN_REL='tools/pireus/operator_orbit_canonicalization.parity-open.v13'
RECEIPT_REL='tools/pireus/operator_orbit_canonicalization.formal-parity.v13'

PARITY_OPEN_COMMIT='3ded53a962f32849a904536668cb5c0b75d83323'
SOURCE_COMMIT='a4b0abd83c70a2c864770d6d781335ef15754372'
SOURCE_SHA256='2f6d04c9e0f82552d13f1cbc73776bc91832c5eb018976870d6c877c7e33d70f'
AUDIT_SHA256='f661eb908e425c6ee8e2e30daec4537069d7a57447d5297cad52d3d4b22f9d63'
LAKEFILE_SHA256='44e2914313660ff625cd01970552779c9d054e1b8548fe43e2848867917b3e64'
PARITY_OPEN_SHA256='4d24259d1807cffa999a90aea4e4797fcbce50659c2e59c34deccb3ca33bdfbf'
RECEIPT_SHA256='8ba5fd22f677b12af33f3269fbc2c78851720ba851c76424d96d092b9da3e871'
SOUNIO_SEMANTICS_SHA256='0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c'
TOOLCHAIN_SHA256='03526d62a2416b90e4cad0c369f73bbed9f38f700e0182f220f511235548bf63'
HARDWARE_SHA256='ba649e3df628e482654987fc29b86489bfdfc9f69f6bfdc141f3a8d238ee33cd'
BUILD_COMMAND_SHA256='b3aba8ca9f89bf915ce8cf3c4941f568d9693a1574083609849d397e2d08c04c'
AUDIT_COMMAND_SHA256='0aa4abd00d78a6beb1afdd64a4b79e90d53555e73f3b1bf192cbd02c2f99b9e9'
BUILD_FRAME_SHA256='a207b5d2d0a39e324165a2d38ecd808ed8db0890eabe082ed13903701e9baa47'
AUDIT_FRAME_SHA256='2483aaf190c856ee69c71f1738beea7ac8d2f1dfb5c8c0c832d542b83a6f0626'
AUDIT_OUTPUT_SHA256='4fc60614513d472a3d907284900d13dd725dcf7ea20e623d762031372b8148ac'
ZERO='0 0 0 0 0 0 0 0'
EXPECTED_THEOREMS=(
  gl4_f2_enumeration_has_exactly_20160_matrices
  every_invertible_4x4_code_is_in_the_scan
  declared_linear_swap_view_census_is_40320
  basis_fixed_gauge_codec_roundtrips_all_2048_words
  every_11_bit_gauge_word_is_in_the_scan
  interior_codec_is_a_225_cell_bijection
  destination_major_microprogram_indexes_all_256_cells
  exact_cell_separator_implies_distinct_tables
  admission_accounting_is_33_minus_1_equals_32
  class_accounting_is_30_plus_32_equals_62
  separator_accounting_is_32_times_30_plus_choose_32_2
  formal_parity_summary_matches_frozen_sounio_snapshot
  formal_parity_is_bound_to_frozen_v13_hashes
  formal_parity_remains_open_on_concrete_class_reconstruction
  executable_scope_does_not_promote_broader_classification
)

fail() {
  printf 'pireus orbit formal parity: FAIL: %s\n' "$*" >&2
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
  local label="$1" frame="$2" expected_sha="$3"
  local decision
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
require_hash "${ROOT}/${PARITY_OPEN_REL}" "${PARITY_OPEN_SHA256}"
require_hash "${ROOT}/${RECEIPT_REL}" "${RECEIPT_SHA256}"
git -C "${ROOT}" merge-base --is-ancestor "${PARITY_OPEN_COMMIT}" "${SOURCE_COMMIT}" || fail 'Parity-open does not precede Lean source'
git -C "${ROOT}" merge-base --is-ancestor "${SOURCE_COMMIT}" HEAD || fail 'Lean source commit is not in current history'
require_committed_hash "${SOURCE_COMMIT}" "${SOURCE_REL}" "${SOURCE_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${AUDIT_REL}" "${AUDIT_SHA256}"
require_committed_hash "${SOURCE_COMMIT}" "${LAKEFILE_REL}" "${LAKEFILE_SHA256}"

require_line "${ROOT}/${RECEIPT_REL}" 'status=PARTIAL_PASS'
require_line "${ROOT}/${RECEIPT_REL}" 'producing_language=Lean4'
require_line "${ROOT}/${RECEIPT_REL}" 'language_role=FORMAL_PARITY'
require_line "${ROOT}/${RECEIPT_REL}" 'formal_parity_complete=false'
require_line "${ROOT}/${RECEIPT_REL}" 'proved_gl4_f2_matrix_code_scan_complete=true'
require_line "${ROOT}/${RECEIPT_REL}" 'proved_gl4_f2_order_20160=true'
require_line "${ROOT}/${RECEIPT_REL}" 'proved_linear_swap_operator_distinctness=false'
require_line "${ROOT}/${RECEIPT_REL}" 'proved_separator_formula=32_TIMES_30_PLUS_(32_TIMES_31)_DIV_2'
require_line "${ROOT}/${RECEIPT_REL}" 'concrete_30_class_reconstruction_proved=false'
require_line "${ROOT}/${RECEIPT_REL}" 'canonical_representative_equality_iff_same_declared_orbit_proved=false'
require_line "${ROOT}/${RECEIPT_REL}" 'axiom_audit_sorryax_mentions=0'
require_line "${ROOT}/${RECEIPT_REL}" 'kernel_axiom_free_claim=false'
require_line "${ROOT}/${RECEIPT_REL}" 'commit_command_sha256=ae6239fbea80ada695818023c8c8e3e9a775a1cff60f629691fbb3211f8fade3'
require_line "${ROOT}/${RECEIPT_REL}" 'commit_frame_sha256=113a3ae5babfaab96f709bedd61bf502282b2852a734079bbdac0c99bbc14a6d'
require_line "${ROOT}/${RECEIPT_REL}" 'commit_decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
require_line "${ROOT}/${RECEIPT_REL}" 'claim_ready=false'

[[ "$(grep -c '^theorem ' "${ROOT}/${SOURCE_REL}")" -eq 15 ]] || fail 'theorem surface count drift'
[[ "$(grep -c '^#print axioms ' "${ROOT}/${AUDIT_REL}")" -eq 15 ]] || fail 'axiom audit surface count drift'
for theorem_name in "${EXPECTED_THEOREMS[@]}"; do
  grep -Eq "^theorem ${theorem_name}([[:space:]]|:|$)" "${ROOT}/${SOURCE_REL}" ||
    fail "missing Lean theorem declaration: ${theorem_name}"
  require_line "${ROOT}/${AUDIT_REL}" "#print axioms ${theorem_name}"
done
require_line "${ROOT}/${SOURCE_REL}" 'theorem gl4_f2_enumeration_has_exactly_20160_matrices :'
require_line "${ROOT}/${SOURCE_REL}" '    invertibleMatrixCount = expectedGL4Order := by'
require_line "${ROOT}/${SOURCE_REL}" 'theorem separator_accounting_is_32_times_30_plus_choose_32_2 :'
require_line "${ROOT}/${SOURCE_REL}" '        admittedClasses * baselineClasses +'
require_line "${ROOT}/${SOURCE_REL}" '          admittedClasses * (admittedClasses - 1) / 2 ∧'
require_line "${ROOT}/${SOURCE_REL}" '      separatorCertificates = 1456 := by'
(( 2 ** 16 == 65536 )) || fail 'F2 matrix-code cardinality drift'
(( 2 * 20160 == 40320 )) || fail 'linear/swap pair arithmetic drift'
(( 2 ** 11 == 2048 )) || fail 'gauge-word cardinality drift'
(( 15 * 15 == 225 )) || fail 'interior-cell cardinality drift'
(( 16 * 16 == 256 )) || fail 'microprogram-cell cardinality drift'
(( 32 * 30 + (32 * 31) / 2 == 1456 )) || fail 'separator arithmetic drift'
if grep -Eq '\bsorry\b|sorryAx' "${ROOT}/${SOURCE_REL}" "${ROOT}/${AUDIT_REL}"; then
  fail 'sorry or sorryAx found in Lean source'
fi

TMPDIR_PIREUS="$(mktemp -d)"
trap 'rm -rf "${TMPDIR_PIREUS}"' EXIT
check_guardian BUILD "$(parity_frame "${SOURCE_SHA256}" "${BUILD_COMMAND_SHA256}")" "${BUILD_FRAME_SHA256}"
(
  cd "${ROOT}/formal/lean4"
  lake build SounioPireusOperatorOrbitCanonicalization >"${TMPDIR_PIREUS}/build.txt" 2>&1
)
check_guardian AXIOM_AUDIT "$(parity_frame "${AUDIT_SHA256}" "${AUDIT_COMMAND_SHA256}")" "${AUDIT_FRAME_SHA256}"
(
  cd "${ROOT}/formal/lean4"
  lake env lean -j 1 SounioPireusOperatorOrbitCanonicalizationAxiomAudit.lean >"${TMPDIR_PIREUS}/axioms.txt"
)
[[ "$(sha_file "${TMPDIR_PIREUS}/axioms.txt")" == "${AUDIT_OUTPUT_SHA256}" ]] || fail 'axiom audit output drift'
[[ "$(grep -c "^'SounioPireusOperatorOrbitCanonicalization\\..*' depends on axioms:" "${TMPDIR_PIREUS}/axioms.txt")" -eq 12 ]] || fail 'axiom-bearing report count drift'
[[ "$(grep -c 'does not depend on any axioms' "${TMPDIR_PIREUS}/axioms.txt")" -eq 3 ]] || fail 'axiom-free report count drift'
[[ "$(grep -o '_native\.native_decide\.ax_1_1' "${TMPDIR_PIREUS}/axioms.txt" | wc -l)" -eq 9 ]] || fail 'native_decide report count drift'
if grep -q 'sorryAx' "${TMPDIR_PIREUS}/axioms.txt"; then
  fail 'sorryAx appeared in axiom audit'
fi

printf '%s\n' \
  'PIREUS_OPERATOR_ORBIT_CANONICALIZATION_FORMAL_PARITY_PASS=true status=PARTIAL_PASS language=Lean4 role=FORMAL_PARITY gl4_f2_matrix_codes=65536 gl4_f2_order=20160 linear_swap_pairs=40320 gauge_words=2048 interior_cells=225 microprogram_cells=256 separator_formula=32*30+(32*31)/2 separator_certificates=1456 theorem_reports=15 no_axiom_reports=3 native_decide_mentions=9 sorryax_mentions=0 concrete_30_class_reconstruction=false canonical_iff_orbit=false formal_parity_complete=false semantic_authority=Sounio expected_results_supplied_by_lean=false claim_ready=false'
