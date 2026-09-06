#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"

SOUNIO_REL='stdlib/hardware/pireus/operator_novelty_feedback.sio'
FREEZE_REL='tools/pireus/operator_novelty_feedback.freeze.v7'
PARITY_OPEN_REL='tools/pireus/operator_novelty_feedback.parity-open.v7'
SOURCE_MANIFEST_REL='tools/pireus/operator_novelty_feedback.lean-sources.v7'
LEAN_REL='formal/lean4/SounioPireusOperatorNoveltyFeedback.lean'
AXIOM_AUDIT_REL='formal/lean4/SounioPireusOperatorNoveltyFeedbackAxiomAudit.lean'
LAKE_REL='formal/lean4/lakefile.lean'
EVIDENCE_REL='tools/pireus/evidence/operator_novelty_feedback_v7.lean.txt'
RECEIPT_REL='tools/pireus/operator_novelty_feedback.formal-parity.v7'
OLEAN_REL='formal/lean4/.lake/build/lib/lean/SounioPireusOperatorNoveltyFeedback.olean'
ILEAN_REL='formal/lean4/.lake/build/lib/lean/SounioPireusOperatorNoveltyFeedback.ilean'
PARENT_GATE_REL='scripts/ci/pireus_operator_novelty_feedback.sh'

PARITY_OPEN_COMMIT='eb51babec189fbdc94ec6c20cdf2b8144ca9a03f'
FORMAL_PARITY_COMMIT='87797660939e8dbafd570d378c57cae1b2e6e782'
SOUNIO_SHA256='b73cc3fb6a905193a68a65eb6afd5d27da80395a0c38ae3772f9df56e8c8deaf'
SEMANTICS_SHA256='a1be292392727cf515baf6d95a376d6060d56f9b807fc58d8998fbe23bdc7726'
FREEZE_SHA256='7293594eb7a881d1f89d9593b1cc19e3e611f99a491a4cd1146afe0a68cd623a'
PARITY_OPEN_SHA256='1ae8fe022071d12193624477f531595a789ffd05f97489e4ccd05d93cf78f7ef'
SOURCE_MANIFEST_SHA256='35f30a3a464efcef97c5659ef271420319096f539efcb79a3d87a66838d43bc0'
LEAN_SHA256='8ecf3690d6bacd727275ffc6f77fe191319e7711a6b03cb7e8b57d5fc3855e92'
AXIOM_AUDIT_SHA256='3d319d886f93ff7f2b285dc4e93afcf9dc27d632667df2f9e5f8b91b48026cd4'
LAKE_SHA256='dfa2a47e05abf07714d4df7ed412b3c93f76220f4c939bb92672dbb751c9128b'
EVIDENCE_SHA256='a453ef1c359669b03105b47ca134d623c965403fd17a603010cd0ffa338929e9'
RECEIPT_SHA256='305b75a4bb40f2568ff743780f5866565ac5da8e9d46c2fee9952f6990248b47'
OLEAN_SHA256='79de0831cb50cc553055331a49d295fc0975870dbebf5de2569f403311019a43'
ILEAN_SHA256='81bf30b18ddf8c46a9cb5684549a8cada508c756177582a2c61773ae64636e69'
PARENT_GATE_SHA256='f6fe50d51b8335ec0358c5c3d074237f95c9803159e69ced148b82f956a6360d'
TOOLCHAIN_SHA256='03526d62a2416b90e4cad0c369f73bbed9f38f700e0182f220f511235548bf63'
LEAN_BINARY_SHA256='e8baaa71855a616dc351028f3ad2200051b0671f423a1696a100e809302d5550'
LAKE_BINARY_SHA256='60330ab6f07dce20f3fa9ebb08e8b984ea9549eac172afeb15d9d2227060e2b3'
LEAN_COMMAND_SHA256='19d38963260cfb376f1aab0f0fbcf4e80ec25c8bd0ba3b1797d95141d56ec55a'
LAKE_COMMAND_SHA256='19d38963260cfb376f1aab0f0fbcf4e80ec25c8bd0ba3b1797d95141d56ec55a'
HARDWARE_SHA256='ba649e3df628e482654987fc29b86489bfdfc9f69f6bfdc141f3a8d238ee33cd'
COMMAND_SHA256='02eeb3b8191071128aeb1e5a6909249b0731d63d5acdf06932d60077610840a1'
AXIOM_AUDIT_COMMAND_SHA256='b7322e745758e31cf13ec35a3935088c271c280d50305a28ba4a460eb3812c49'
PREEXEC_FRAME_SHA256='51c9679daaa7a1e38e145311843dbb941d47ab5b44e5d49cd568ba7f524ed0f0'
AXIOM_AUDIT_PREEXEC_FRAME_SHA256='b4b959c6f0d3c800a2a9252bf4121fa2f4e06b64aa89c06266da3f6bd1701591'
SEAL_FRAME_SHA256='a5c553a5f90e4e02d293c25a890869afe110082119ab89a96155a129934a8226'
BAD_SEAL_FRAME_SHA256='b83ed6abcf6970449850d3063c0470163e3f0e88a8ac2e8778b8772e229505e9'
WRONG_PARENT_FRAME_SHA256='acc329d9d649fe06c4b2eae3cd95d09be69198fb84ebc8beb67952672add7dfe'
LEAN_WRITE_FRAME_SHA256='67d442f73e75059b369897c9ab17517ae076e5c5c5b480657acf08da0215b30d'
PYTHON_FRAME_SHA256='57c6f88327fd01441fcc12f7893cf4ade97576a23697cc725db312e6863f7f71'
CLAIM_PROMOTION_FRAME_SHA256='fac688057b486c3f97971c57476eea6655f866f67aaa0c1809a5bf265f9f0283'
PYTHON_TOOLCHAIN_SHA256='497ce0938df96d9bf3c159472d251a946c2d6bd832220937d1b885f7759b05ba'
PYTHON_COMMAND_SHA256='72566473f0019fb50e65175a0ac019af5ad2f495ccfc30b664b7735791e968fc'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
GUARDIAN_POLICY_SHA256='64bb0118793fe46dcb392abc1a9212eb15bd55047461576a3ef1a6cefa3f17da'
WRONG_PARENT='0d69d1d890506ebf90ff14fe5e3f2b653d5651b968d2fca7fe8ef9298a0c26c1'
ZERO='0 0 0 0 0 0 0 0'
GUARDIAN_FRAME_WORDS=82

fail() {
  printf 'pireus operator novelty formal parity: FAIL: %s\n' "$*" >&2
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
  [[ "$(sha_file "${path}")" == "${expected}" ]] ||
    fail "hash drift: ${path}"
}

require_line() {
  local path="$1" expected="$2"
  grep -Fqx -- "${expected}" "${path}" ||
    fail "missing exact line in ${path}: ${expected}"
}

receipt_admitted() {
  local path="$1" key
  for key in \
    status stage formal_obligations formal_obligations_discharged \
    formal_parity_complete positive_bridge_branch_reconstructed \
    broad_novelty historical_novelty priority_claim claim_ready; do
    [[ "$(grep -c "^${key}=" "${path}")" -eq 1 ]] || return 1
  done
  grep -Fqx -- 'status=FORMAL_PARITY_COMPLETE' "${path}" &&
    grep -Fqx -- 'stage=PARITY_OPEN' "${path}" &&
    grep -Fqx -- 'formal_obligations=39' "${path}" &&
    grep -Fqx -- 'formal_obligations_discharged=39' "${path}" &&
    grep -Fqx -- 'formal_parity_complete=true' "${path}" &&
    grep -Fqx -- 'positive_bridge_branch_reconstructed=false' "${path}" &&
    grep -Fqx -- 'broad_novelty=false' "${path}" &&
    grep -Fqx -- 'historical_novelty=false' "${path}" &&
    grep -Fqx -- 'priority_claim=false' "${path}" &&
    grep -Fqx -- 'claim_ready=false' "${path}"
}

manifest_value() {
  local key="$1" path="$2" count
  count="$(grep -Fc "${key}=" "${path}")"
  [[ "${count}" -eq 1 ]] || fail "manifest key count drift: ${key}=${count}"
  sed -n "s/^${key}=//p" "${path}"
}

manifest_admitted() {
  local manifest="$1" committed="$2" n i key path expected actual
  require_line "${manifest}" 'status=PARITY_SOURCE_CLOSED'
  require_line "${manifest}" 'stage=PARITY_OPEN'
  require_line "${manifest}" 'source_count=35'
  require_line "${manifest}" 'parent_source_count=2'
  require_line "${manifest}" 'certificate_source_count=32'
  for ((n = 1; n <= 35; n++)); do
    printf -v i '%02d' "${n}"
    key="source_${i}"
    path="$(manifest_value "${key}_path" "${manifest}")"
    expected="$(manifest_value "${key}_sha256" "${manifest}")"
    [[ -f "${ROOT}/${path}" ]] || return 1
    actual="$(sha_file "${ROOT}/${path}")"
    [[ "${actual}" == "${expected}" ]] || return 1
    if [[ "${committed}" == true ]]; then
      actual="$(git -C "${ROOT}" show "${FORMAL_PARITY_COMMIT}:${path}" |
        sha256sum | cut -d' ' -f1)"
      [[ "${actual}" == "${expected}" ]] || return 1
    fi
  done
}

parity_frame() {
  local command_hash="$1" action="$2" semantic_write="$3"
  local parent_hash="$4" result_hash="$5"
  local stage=3 receipt_valid=0 result_limbs="${ZERO}"
  if [[ "${action}" == 8 ]]; then
    stage=4
    receipt_valid=1
    result_limbs="$(sha_limbs "${result_hash}")"
  fi
  printf '9020 %s %s 2 2 1 %s 0 %s 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${stage}" "${action}" "${semantic_write}" "${receipt_valid}" \
    "$(sha_limbs "${SOURCE_MANIFEST_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${parent_hash}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${command_hash}")" "${result_limbs}" "${ZERO}"
}

bad_seal_frame() {
  printf '9020 4 8 2 2 1 0 1 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${SOURCE_MANIFEST_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${COMMAND_SHA256}")" \
    "$(sha_limbs "${OLEAN_SHA256}")" "${ZERO}"
}

python_frame() {
  printf '9020 3 4 7 7 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${SOURCE_MANIFEST_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PYTHON_TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${PYTHON_COMMAND_SHA256}")" "${ZERO}" "${ZERO}"
}

claim_promotion_frame() {
  printf '9020 4 7 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${SOURCE_MANIFEST_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${COMMAND_SHA256}")" \
    "$(sha_limbs "${OLEAN_SHA256}")" "${ZERO}"
}

authorize() {
  local label="$1" frame="$2" expected_sha="$3" expected_rc="$4" expected="$5"
  local decision rc words
  words="$(wc -w <<<"${frame}" | tr -d ' ')"
  [[ "${words}" -eq "${GUARDIAN_FRAME_WORDS}" ]] ||
    fail "${label}: Guardian frame word count drift: ${words}"
  [[ "$(sha_text "${frame}")" == "${expected_sha}" ]] ||
    fail "${label}: Guardian frame hash drift"
  set +e
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  rc=$?
  set -e
  [[ "${rc}" -eq "${expected_rc}" ]] ||
    fail "${label}: Guardian exit drift: ${rc}"
  [[ "${decision}" == "${expected}" ]] ||
    fail "${label}: Guardian decision drift: ${decision}"
  printf 'GUARDIAN_DECISION label=%s frame_sha256=%s rc=%s %s\n' \
    "${label}" "${expected_sha}" "${rc}" "${decision}"
}

cd "${ROOT}"
require_hash "${ROOT}/${SOUNIO_REL}" "${SOUNIO_SHA256}"
require_hash "${ROOT}/${FREEZE_REL}" "${FREEZE_SHA256}"
require_hash "${ROOT}/${PARITY_OPEN_REL}" "${PARITY_OPEN_SHA256}"
require_hash "${ROOT}/${SOURCE_MANIFEST_REL}" "${SOURCE_MANIFEST_SHA256}"
require_hash "${ROOT}/${LEAN_REL}" "${LEAN_SHA256}"
require_hash "${ROOT}/${AXIOM_AUDIT_REL}" "${AXIOM_AUDIT_SHA256}"
require_hash "${ROOT}/${LAKE_REL}" "${LAKE_SHA256}"
require_hash "${ROOT}/${EVIDENCE_REL}" "${EVIDENCE_SHA256}"
require_hash "${ROOT}/${RECEIPT_REL}" "${RECEIPT_SHA256}"
require_hash "${ROOT}/${PARENT_GATE_REL}" "${PARENT_GATE_SHA256}"
require_hash "${GUARDIAN}" "${GUARDIAN_SHA256}"
require_hash "${ROOT}/stdlib/coordination/loom_language_authority.sio" \
  "${GUARDIAN_POLICY_SHA256}"
[[ -x "${GUARDIAN}" ]] || fail 'native Loom Guardian unavailable'

manifest_admitted "${ROOT}/${SOURCE_MANIFEST_REL}" true ||
  fail 'Lean source manifest admission failed'
[[ "$(manifest_value source_01_path "${ROOT}/${SOURCE_MANIFEST_REL}")" == \
  'formal/lean4/SounioCDCocycle.lean' ]] || fail 'challenge parent partition drift'
[[ "$(manifest_value source_02_path "${ROOT}/${SOURCE_MANIFEST_REL}")" == \
  'formal/lean4/SounioPireusQuotientNoveltyForge.lean' ]] ||
  fail 'quotient parent partition drift'
[[ "$(manifest_value source_03_path "${ROOT}/${SOURCE_MANIFEST_REL}")" == \
  'formal/lean4/SounioPireusOperatorNoveltyFeedbackCore.lean' ]] ||
  fail 'certificate partition start drift'
[[ "$(manifest_value source_34_path "${ROOT}/${SOURCE_MANIFEST_REL}")" == \
  'formal/lean4/SounioPireusOperatorNoveltyFeedbackAxiomAudit.lean' ]] ||
  fail 'certificate partition end drift'
[[ "$(manifest_value source_35_path "${ROOT}/${SOURCE_MANIFEST_REL}")" == \
  'formal/lean4/lakefile.lean' ]] || fail 'build definition partition drift'

git -C "${ROOT}" merge-base --is-ancestor \
  "${PARITY_OPEN_COMMIT}" "${FORMAL_PARITY_COMMIT}" ||
  fail 'formal parity predates PARITY_OPEN'
git -C "${ROOT}" merge-base --is-ancestor "${FORMAL_PARITY_COMMIT}" HEAD ||
  fail 'formal parity commit missing from current history'
for pair in \
  "${SOURCE_MANIFEST_REL}:${SOURCE_MANIFEST_SHA256}" \
  "${LEAN_REL}:${LEAN_SHA256}" \
  "${AXIOM_AUDIT_REL}:${AXIOM_AUDIT_SHA256}" \
  "${LAKE_REL}:${LAKE_SHA256}" \
  "${EVIDENCE_REL}:${EVIDENCE_SHA256}" \
  "${RECEIPT_REL}:${RECEIPT_SHA256}"; do
  path="${pair%%:*}"
  expected="${pair#*:}"
  actual="$(git -C "${ROOT}" show "${FORMAL_PARITY_COMMIT}:${path}" |
    sha256sum | cut -d' ' -f1)"
  [[ "${actual}" == "${expected}" ]] ||
    fail "committed formal artifact drift: ${path}"
done

proof_sources=()
for ((n = 3; n <= 34; n++)); do
  printf -v i '%02d' "${n}"
  proof_sources+=("${ROOT}/$(manifest_value "source_${i}_path" "${ROOT}/${SOURCE_MANIFEST_REL}")")
done
if grep -En '(^|[^[:alnum:]_])(sorry|sorryAx|admit|native_decide|axiom|constant|opaque|extern|implemented_by)([^[:alnum:]_]|$)' \
    "${proof_sources[@]}" >/dev/null; then
  fail 'Lean parity contains a proof or trust escape hatch'
fi

for theorem in \
  formal_parity_summary_matches_frozen_sounio \
  cd16_challenge_census_and_words_exact \
  frozen_parent_actions_admitted_closed_and_invertible \
  atlas_enumeration_and_nonmembership_exact \
  canonical_operator_seed_witness_exact \
  operator_seed_and_claim_bound_exact; do
  grep -Fq "theorem ${theorem}" "${ROOT}/${LEAN_REL}" ||
    fail "missing exported theorem: ${theorem}"
done
for theorem_fragment in \
  'formalParitySummary.challengePositive = 136 &&' \
  'formalParitySummary.challengeNegative = 120 &&' \
  'formalParitySummary.actionCount = 12 &&' \
  'formalParitySummary.parentActionsAdmitted &&' \
  'formalParitySummary.parentActionClosure &&' \
  'formalParitySummary.parentActionInverses &&' \
  'formalParitySummary.classCount = 14 &&' \
  'formalParitySummary.pairCount = 168 &&' \
  'formalParitySummary.pairReplayFailures = 0 &&' \
  'formalParitySummary.zeroResidualHits = 0 &&' \
  'formalParitySummary.bestClass = 8 &&' \
  'formalParitySummary.bestRepresentative = 13 &&' \
  'formalParitySummary.bestActionCode = 68674 &&' \
  'formalParitySummary.bestResidualNonzero = 96 &&' \
  'formalParitySummary.outcomeKind = 2 &&' \
  '!formalParitySummary.existingClassBridge &&' \
  'formalParitySummary.operatorSeedGenerated &&' \
  '!formalParitySummary.broadNovelty &&' \
  '!formalParitySummary.historicalNovelty &&' \
  '!formalParitySummary.priorityClaim &&' \
  '!formalParitySummary.claimReady'; do
  grep -Fq -- "${theorem_fragment}" "${ROOT}/${LEAN_REL}" ||
    fail "exported theorem statement drift: ${theorem_fragment}"
done
[[ "$(grep -Fc '#print axioms ' "${ROOT}/${AXIOM_AUDIT_REL}")" -eq 39 ]] ||
  fail 'axiom audit declaration count drift'

require_line "${ROOT}/${PARITY_OPEN_REL}" 'stage=PARITY_OPEN'
require_line "${ROOT}/${PARITY_OPEN_REL}" 'lean_status=OPEN_NOT_EXECUTED'
for expected in \
  'status=FORMAL_PARITY_COMPLETE' \
  'stage=PARITY_OPEN' \
  'producing_language=Lean_4' \
  'producing_role=FORMAL_PARITY' \
  "sounio_source_sha256=${SOUNIO_SHA256}" \
  "sounio_semantics_sha256=${SEMANTICS_SHA256}" \
  "source_manifest_sha256=${SOURCE_MANIFEST_SHA256}" \
  "result_sha256=${OLEAN_SHA256}" \
  'formal_obligations=39' \
  'formal_obligations_discharged=39' \
  'formal_parity_complete=true' \
  'proof_reduction=KERNEL_DECIDE_SHARDED' \
  'axiom_audit_theorems=39' \
  'axiom_closure=BOUNDED_LEAN_FOUNDATION' \
  'lean_foundational_axioms=propext,Quot.sound' \
  'unexpected_axiom_profiles=0' \
  'positive_bridge_branch_reconstructed=false' \
  'effect_parity_complete=false' \
  'material_parity_complete=false' \
  'semantic_write=false' \
  'expected_result_write=false' \
  'broad_novelty=false' \
  'historical_novelty=false' \
  'priority_claim=false' \
  'claim_ready=false'; do
  require_line "${ROOT}/${RECEIPT_REL}" "${expected}"
done
receipt_admitted "${ROOT}/${RECEIPT_REL}" ||
  fail 'formal receipt admission failed'

for expected in \
  'certificate_nodes=39' \
  'certificate_nodes_discharged=39' \
  'axiom_free_theorems=1' \
  'propext_only_theorems=36' \
  'propext_quot_sound_theorems=2' \
  'unexpected_axiom_profiles=0' \
  'challenge_positive=136' \
  'challenge_negative=120' \
  'atlas_pairs=168' \
  'pair_replay_failures=0' \
  'zero_residual_hits=0' \
  'frozen_outcome=OPERATOR_SEED' \
  'best_class=8' \
  'best_representative=13' \
  'best_action_code=68674' \
  'best_residual_nonzero=96' \
  'failure_outcome=NO_BRIDGE_NO_SEED' \
  'positive_bridge_branch_reconstructed=false' \
  'representative_roster_scope=BOUND_UNIQUE_AND_LT_48_ONLY' \
  'formal_parity_complete=true' \
  'claim_ready=false' \
  'python_processes_launched=0' \
  'rust_processes_launched=0'; do
  require_line "${ROOT}/${EVIDENCE_REL}" "${expected}"
done

lean_version="$(lean --version | sed -n '1p')"
lake_version="$(cd "${ROOT}/formal/lean4" && lake --version)"
lean_prefix="$(lean --print-prefix)"
require_hash "$(command -v lean)" "${LEAN_COMMAND_SHA256}"
require_hash "$(command -v lake)" "${LAKE_COMMAND_SHA256}"
require_hash "${lean_prefix}/bin/lean" "${LEAN_BINARY_SHA256}"
require_hash "${lean_prefix}/bin/lake" "${LAKE_BINARY_SHA256}"
toolchain_record="lean=${lean_version} lake=${lake_version}"
[[ "$(sha_text "${toolchain_record}")" == "${TOOLCHAIN_SHA256}" ]] ||
  fail 'Lean toolchain drift'
hardware_record="host=$(hostname) arch=$(uname -m) kernel=$(uname -s) $(uname -r) online_cpus=$(getconf _NPROCESSORS_ONLN) model=$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -n 1)"
[[ "$(sha_text "${hardware_record}")" == "${HARDWARE_SHA256}" ]] ||
  fail 'hardware drift'
command_record='cd formal/lean4 && lake build SounioPireusOperatorNoveltyFeedback'
[[ "$(sha_text "${command_record}")" == "${COMMAND_SHA256}" ]] ||
  fail 'build command drift'
axiom_audit_command_record='cd formal/lean4 && lake env lean SounioPireusOperatorNoveltyFeedbackAxiomAudit.lean'
[[ "$(sha_text "${axiom_audit_command_record}")" == \
    "${AXIOM_AUDIT_COMMAND_SHA256}" ]] || fail 'axiom audit command drift'

set +e
invalid_hash_output="$(sha_limbs 'not-a-sha256' 2>&1)"
invalid_hash_rc=$?
set -e
[[ "${invalid_hash_rc}" -eq 1 ]] || fail 'malformed SHA-256 did not fail closed'
[[ "${invalid_hash_output}" == \
  'pireus operator novelty formal parity: FAIL: invalid sha256 digest: not-a-sha256' ]] ||
  fail 'malformed SHA-256 refusal drift'
printf 'GUARDIAN_DISPATCH label=MALFORMED_SHA256 process_launched=false\n'

authorize PREEXEC \
  "$(parity_frame "${COMMAND_SHA256}" 4 0 "${SEMANTICS_SHA256}" zero)" \
  "${PREEXEC_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
authorize AXIOM_AUDIT_PREEXEC \
  "$(parity_frame "${AXIOM_AUDIT_COMMAND_SHA256}" 4 0 "${SEMANTICS_SHA256}" zero)" \
  "${AXIOM_AUDIT_PREEXEC_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
authorize WRONG_PARENT \
  "$(parity_frame "${COMMAND_SHA256}" 4 0 "${WRONG_PARENT}" zero)" \
  "${WRONG_PARENT_FRAME_SHA256}" 117 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=117 reason=parent-semantics-hash-mismatch next_stage=SEMANTICS_FROZEN'
printf 'GUARDIAN_DISPATCH label=WRONG_PARENT process_launched=false\n'
authorize LEAN_SEMANTIC_WRITE \
  "$(parity_frame "${COMMAND_SHA256}" 4 1 "${SEMANTICS_SHA256}" zero)" \
  "${LEAN_WRITE_FRAME_SHA256}" 113 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=SEMANTICS_FROZEN'
printf 'GUARDIAN_DISPATCH label=LEAN_SEMANTIC_WRITE process_launched=false\n'
authorize PYTHON_ORACLE "$(python_frame)" "${PYTHON_FRAME_SHA256}" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
printf 'GUARDIAN_DISPATCH label=PYTHON_ORACLE process_launched=false\n'
authorize CLAIM_PROMOTION "$(claim_promotion_frame)" \
  "${CLAIM_PROMOTION_FRAME_SHA256}" 122 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=122 reason=parity-receipt-missing next_stage=PARITY_OPEN'
printf 'GUARDIAN_DISPATCH label=CLAIM_PROMOTION process_launched=false\n'
authorize BAD_SEAL "$(bad_seal_frame)" "${BAD_SEAL_FRAME_SHA256}" 114 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=114 reason=expected-result-authority-required next_stage=PARITY_OPEN'
printf 'GUARDIAN_DISPATCH label=BAD_SEAL process_launched=false\n'

tmp_dir="$(mktemp -d /tmp/pireus-onf-formal-gate.XXXXXX)"
trap 'rm -rf "${tmp_dir}"' EXIT
sed "s/^source_03_sha256=.*/source_03_sha256=${WRONG_PARENT}/" \
  "${ROOT}/${SOURCE_MANIFEST_REL}" >"${tmp_dir}/tampered-manifest.v7"
if manifest_admitted "${tmp_dir}/tampered-manifest.v7" false; then
  fail 'tampered Lean source manifest was admitted'
fi
sed 's/^claim_ready=false$/claim_ready=true/' "${ROOT}/${RECEIPT_REL}" \
  >"${tmp_dir}/tampered-receipt.v7"
if receipt_admitted "${tmp_dir}/tampered-receipt.v7"; then
  fail 'claim promotion sabotage passed receipt admission'
fi
printf 'SABOTAGE manifest_hash=REFUSED receipt_flag_mutation=DETECTED\n'

parent_gate_output="$("${ROOT}/${PARENT_GATE_REL}")"
printf '%s\n' "${parent_gate_output}" | grep -Fqx -- \
  'pireus operator novelty feedback: STAGE_REACHED_NOT_A_CLAIM gate_mode=CONTENT_ADDRESSED_PARITY_OPEN_REPLAY stage=PARITY_OPEN operator_seed=true relative_scope=FINITE_QUOTIENT_ONLY formal=OPEN_NOT_EXECUTED effect=OPEN_NOT_EXECUTED material=OPEN_NOT_EXECUTED claim_promotion=DENIED claim_ready=false' ||
  fail 'Sounio authority parent gate terminal marker drift'

require_hash "${ROOT}/${SOURCE_MANIFEST_REL}" "${SOURCE_MANIFEST_SHA256}"
manifest_admitted "${ROOT}/${SOURCE_MANIFEST_REL}" true ||
  fail 'Lean source manifest drifted before build'
(cd "${ROOT}/formal/lean4" && lake build SounioPireusOperatorNoveltyFeedback)
require_hash "${ROOT}/${SOURCE_MANIFEST_REL}" "${SOURCE_MANIFEST_SHA256}"
manifest_admitted "${ROOT}/${SOURCE_MANIFEST_REL}" true ||
  fail 'Lean source manifest drifted during build'
require_hash "${ROOT}/${EVIDENCE_REL}" "${EVIDENCE_SHA256}"
require_hash "${ROOT}/${RECEIPT_REL}" "${RECEIPT_SHA256}"
receipt_admitted "${ROOT}/${RECEIPT_REL}" ||
  fail 'formal receipt drifted during build'
require_hash "${ROOT}/${OLEAN_REL}" "${OLEAN_SHA256}"
require_hash "${ROOT}/${ILEAN_REL}" "${ILEAN_SHA256}"

axiom_audit_output="$(
  cd "${ROOT}/formal/lean4" &&
    lake env lean SounioPireusOperatorNoveltyFeedbackAxiomAudit.lean
)"
audit_total="$(printf '%s\n' "${axiom_audit_output}" | wc -l)"
audit_empty="$(printf '%s\n' "${axiom_audit_output}" |
  grep -Fc ' does not depend on any axioms')"
audit_propext="$(printf '%s\n' "${axiom_audit_output}" |
  grep -Fc ' depends on axioms: [propext]')"
audit_quot="$(printf '%s\n' "${axiom_audit_output}" |
  grep -Fc ' depends on axioms: [propext, Quot.sound]')"
audit_unexpected="$((audit_total - audit_empty - audit_propext - audit_quot))"
[[ "${audit_total}" -eq 39 && "${audit_empty}" -eq 1 && \
   "${audit_propext}" -eq 36 && "${audit_quot}" -eq 2 && \
   "${audit_unexpected}" -eq 0 ]] || fail 'Lean axiom profile drift'
printf 'LEAN_AXIOM_AUDIT theorems=39 axiom_free=1 propext_only=36 propext_quot_sound=2 unexpected=0 closure=BOUNDED_LEAN_FOUNDATION\n'

authorize SEAL \
  "$(parity_frame "${COMMAND_SHA256}" 8 0 "${SEMANTICS_SHA256}" "${OLEAN_SHA256}")" \
  "${SEAL_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
require_hash "${ROOT}/${SOURCE_MANIFEST_REL}" "${SOURCE_MANIFEST_SHA256}"
require_hash "${ROOT}/${EVIDENCE_REL}" "${EVIDENCE_SHA256}"
require_hash "${ROOT}/${RECEIPT_REL}" "${RECEIPT_SHA256}"
receipt_admitted "${ROOT}/${RECEIPT_REL}" ||
  fail 'formal receipt drifted before terminal marker'

printf '%s\n' \
  'pireus operator novelty formal parity: STAGE_REACHED_NOT_A_CLAIM stage=PARITY_OPEN language=Lean4 role=FORMAL_PARITY certificate_nodes=39/39 atlas_pairs=168 zero_residual_hits=0 outcome=OPERATOR_SEED positive_bridge_branch=NOT_RECONSTRUCTED axiom_closure=BOUNDED_LEAN_FOUNDATION formal=COMPLETE effect=OPEN_NOT_EXECUTED material=OPEN_NOT_EXECUTED broad_novelty=false historical_novelty=false priority_claim=false claim_ready=false python_process_launched=false rust_process_launched=false'
