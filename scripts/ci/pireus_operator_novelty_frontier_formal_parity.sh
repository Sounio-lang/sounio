#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"

SOUNIO_REL='stdlib/hardware/pireus/operator_novelty_frontier.sio'
FREEZE_REL='tools/pireus/operator_novelty_frontier.freeze.v11'
PARITY_OPEN_REL='tools/pireus/operator_novelty_frontier.parity-open.v11'
LEAN_REL='formal/lean4/SounioPireusOperatorNoveltyFrontier.lean'
AUDIT_REL='formal/lean4/SounioPireusOperatorNoveltyFrontierAxiomAudit.lean'
LAKE_REL='formal/lean4/lakefile.lean'
EVIDENCE_REL='tools/pireus/evidence/operator_novelty_frontier_v11.lean.txt'
RECEIPT_REL='tools/pireus/operator_novelty_frontier.formal-parity.v11'
OLEAN_REL='formal/lean4/.lake/build/lib/lean/SounioPireusOperatorNoveltyFrontier.olean'
ILEAN_REL='formal/lean4/.lake/build/lib/lean/SounioPireusOperatorNoveltyFrontier.ilean'

SOUNIO_SHA256='9289cd504385e2f1f4eed095d82a963cf2e5e67124bf8d267d1bc6ccda7ac36b'
FREEZE_SHA256='b57decc8ff929640345e47edc931bdfa6cd06c738d3ff9591d3a460593dae242'
PARITY_OPEN_SHA256='f7cde0ed063d136bbef43cf9e820d734341f87717bb26e130a3643bc62fb31de'
SEMANTIC_SOURCE_MANIFEST_SHA256='a91157e621c2c569bfde51982e66bd46e360f606e7f95d59520732fc1644429a'
SEMANTICS_SHA256='f1e339ec7bc290f412d42bba3fa1ba609fd89947408ea422ab96026cce5883dc'
FORMAL_SOURCE_MANIFEST_SHA256='621cf62b393d11408236d62b6eaa14715cc0e2bda7216cd4e1a0735d2db9a969'
LEAN_SHA256='413ee964aedb1ff79ff4e9e9c4006875992641cc67319c3d9f44c40654ef40e4'
AUDIT_SHA256='3e51864dd1da7e7efa50dfdeec89cecc0525243728914bc5b4402676117b7a7d'
LAKE_SHA256='e440d3c1431f5ab8ef389e1c72044486a91428865afedf8fb187d7fc15cf125c'
EVIDENCE_SHA256='c69c4df062665eda35491f1f478e24634efed7caf0f266f352843c3f40b4fc6f'
RECEIPT_SHA256='b56b1f331879c2a8bbb70dc0adfc5ac61e21e922834c391ce4d815397a589d21'
OLEAN_SHA256='55463735e5930f47c10715c3dbcdfebf774911b91e302e129779234f89cf1a84'
ILEAN_SHA256='5f84b0137e9041dc3572e1c5891c83a06e41256929181fc489680fdf435c3fb7'
TOOLCHAIN_SHA256='03526d62a2416b90e4cad0c369f73bbed9f38f700e0182f220f511235548bf63'
LEAN_BINARY_SHA256='19d38963260cfb376f1aab0f0fbcf4e80ec25c8bd0ba3b1797d95141d56ec55a'
LAKE_BINARY_SHA256='19d38963260cfb376f1aab0f0fbcf4e80ec25c8bd0ba3b1797d95141d56ec55a'
HARDWARE_SHA256='ba649e3df628e482654987fc29b86489bfdfc9f69f6bfdc141f3a8d238ee33cd'
BUILD_COMMAND_SHA256='c921793061bcd0922519f6dde01c51f6d1b8feed082712648d51b683df2b18d1'
AUDIT_COMMAND_SHA256='6063e7e7bfc078637f06ab969de3703ec6c3e16b80bfdd89558229139ad75af0'
WRITE_COMMAND_SHA256='714a21e95676203ad8161b59b7f23f0970d0c6db8598f36ae1ff05fa79445f4c'
BUILD_FRAME_SHA256='5e88a5614bdac800c21c737a82aba06406e294cb909179cd8be159442358f7e1'
AUDIT_FRAME_SHA256='4c2254fda41a9db94f64124e4abbe2b34d592ef4450c04ecc3d9da680183bc94'
SEAL_FRAME_SHA256='1ed8007a087dd2cdb8bd7a14ef458d50e483137b5c73e95714fa2c6928c780cf'
WRITE_FRAME_SHA256='6e57ba142febe8dd9444bdb0cd7930a7213022287b3189e5543538f9d9e888a5'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
GUARDIAN_POLICY_SHA256='64bb0118793fe46dcb392abc1a9212eb15bd55047461576a3ef1a6cefa3f17da'
PYTHON_TOOLCHAIN_SHA256='497ce0938df96d9bf3c159472d251a946c2d6bd832220937d1b885f7759b05ba'
PYTHON_COMMAND_SHA256='72566473f0019fb50e65175a0ac019af5ad2f495ccfc30b664b7735791e968fc'
RUST_TOOLCHAIN_SHA256='b58640570ba9ffcdb2c2d241e4ce8ece9c7d75c6b1e59e308dee3e5f0e10b56d'
RUST_COMMAND_SHA256='0421c7020d972449b7a1f0492945c18a10be46f6a16ade7f8efad157e2e01b00'
ZERO='0 0 0 0 0 0 0 0'
FRAME_WORDS=82

fail() {
  printf 'pireus operator novelty frontier formal parity: FAIL: %s\n' "$*" >&2
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
  rg -Fqx -- "${expected}" "${path}" || fail "missing exact line in ${path}: ${expected}"
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
  [[ "$(sha_text "${frame}")" == "${expected_hash}" ]] || fail "${label}: frame hash drift"
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
    n3_novelty n4_novelty material_novelty historical_novelty priority_claim \
    claim_ready; do
    [[ "$(rg -c "^${key}=" "${path}")" -eq 1 ]] || return 1
  done
  rg -Fqx 'status=FORMAL_PARITY_COMPLETE' "${path}" &&
    rg -Fqx 'stage=PARITY_OPEN' "${path}" &&
    rg -Fqx 'formal_obligations=6' "${path}" &&
    rg -Fqx 'formal_obligations_discharged=6' "${path}" &&
    rg -Fqx 'formal_parity_complete=true' "${path}" &&
    rg -Fqx 'effect_parity_complete=false' "${path}" &&
    rg -Fqx 'material_parity_complete=false' "${path}" &&
    rg -Fqx 'n3_novelty=false' "${path}" &&
    rg -Fqx 'n4_novelty=false' "${path}" &&
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

cpu_model="$(lscpu | sed -n 's/^Model name:[[:space:]]*//p')"
hardware_record="host=$(hostname) arch=$(uname -m) kernel=$(uname -s) $(uname -r) online_cpus=$(getconf _NPROCESSORS_ONLN) model=${cpu_model}"
[[ "$(sha_text "${hardware_record}")" == "${HARDWARE_SHA256}" ]] ||
  fail 'formal hardware record drift'

require_line "${ROOT}/${FREEZE_REL}" "semantics_sha256=${SEMANTICS_SHA256}"
require_line "${ROOT}/${PARITY_OPEN_REL}" 'status=OPEN_NOT_EXECUTED'
require_line "${ROOT}/${PARITY_OPEN_REL}" 'stage=PARITY_OPEN'
require_line "${ROOT}/${PARITY_OPEN_REL}" 'lean_status=OPEN_NOT_EXECUTED'
require_line "${ROOT}/${PARITY_OPEN_REL}" 'claim_ready=false'
receipt_admitted "${ROOT}/${RECEIPT_REL}" || fail 'formal receipt admission failed'

if rg -n '(^|[^A-Za-z])(sorry|admit|axiom)([^A-Za-z]|$)' \
    "${ROOT}/${LEAN_REL}" "${ROOT}/${AUDIT_REL}" >/dev/null; then
  fail 'forbidden proof escape hatch found'
fi

set +e
invalid_hash_output="$(sha_limbs not-a-sha256 2>&1)"
invalid_hash_rc=$?
set -e
[[ "${invalid_hash_rc}" -eq 1 ]] || fail "malformed SHA-256 returned rc=${invalid_hash_rc}"
[[ "${invalid_hash_output}" == \
  'pireus operator novelty frontier formal parity: FAIL: invalid sha256 digest: not-a-sha256' ]] ||
  fail 'malformed SHA-256 did not fail closed'
printf 'GUARDIAN_DISPATCH label=MALFORMED_SHA256 process_launched=false\n'

wrong_parent='e1e339ec7bc290f412d42bba3fa1ba609fd89947408ea422ab96026cce5883dc'
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
  "$(authority_frame 3 4 2 2 1 0 0 0 1 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${BUILD_COMMAND_SHA256}" zero)" 119 \
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

authorize BUILD_PREEXEC \
  "$(authority_frame 3 4 2 2 1 0 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${BUILD_COMMAND_SHA256}" zero)" \
  "${BUILD_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
(cd "${ROOT}/formal/lean4" && lake build SounioPireusOperatorNoveltyFrontier)
require_hash "${ROOT}/${OLEAN_REL}" "${OLEAN_SHA256}"
require_hash "${ROOT}/${ILEAN_REL}" "${ILEAN_SHA256}"

authorize AXIOM_AUDIT_PREEXEC \
  "$(authority_frame 3 4 2 2 1 0 0 0 0 "${SEMANTICS_SHA256}" "${TOOLCHAIN_SHA256}" "${AUDIT_COMMAND_SHA256}" zero)" \
  "${AUDIT_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
axiom_output="$({
  cd "${ROOT}/formal/lean4"
  lake env lean -j 1 SounioPireusOperatorNoveltyFrontierAxiomAudit.lean 2>&1
})"
[[ "$(rg -c ' depends on axioms:' <<<"${axiom_output}")" -eq 7 ]] ||
  fail 'axiom audit theorem count drift'
[[ "$(rg -c 'native_decide.ax_1_1' <<<"${axiom_output}")" -eq 7 ]] ||
  fail 'native_decide axiom profile drift'
[[ "$(rg -c 'propext' <<<"${axiom_output}")" -eq 7 ]] ||
  fail 'propext axiom profile drift'
printf 'LEAN_AXIOM_AUDIT theorems=7 propext=7 native_decide=7 unexpected=0 closure=EXPLICIT_NATIVE_DECIDE_TRUST_BOUNDARY\n'

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
  'pireus operator novelty frontier formal parity: STAGE_REACHED_NOT_A_CLAIM stage=PARITY_OPEN language=Lean4 role=FORMAL_PARITY obligations=6/6 candidates=7200 atlas=6 separators=43200 action_map=EXPLICIT_PARTIAL quotient=7200:0:0:7200:7200 axiom_closure=EXPLICIT_NATIVE_DECIDE_TRUST_BOUNDARY formal=COMPLETE effect=OPEN_NOT_EXECUTED material=OPEN_NOT_EXECUTED n3=false n4=false historical_novelty=false priority_claim=false claim_ready=false python_process_launched=false rust_process_launched=false'
