#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"

SOUNIO_REL='stdlib/hardware/pireus/operator_discovery_engine.sio'
FREEZE_REL='tools/pireus/operator_discovery_engine.freeze.v10'
PARITY_OPEN_REL='tools/pireus/operator_discovery_engine.parity-open.v10'
LEAN_REL='formal/lean4/SounioPireusOperatorDiscoveryEngine.lean'
AUDIT_REL='formal/lean4/SounioPireusOperatorDiscoveryEngineAxiomAudit.lean'
LAKE_REL='formal/lean4/lakefile.lean'
EVIDENCE_REL='tools/pireus/evidence/operator_discovery_engine_v10.lean.txt'
RECEIPT_REL='tools/pireus/operator_discovery_engine.formal-parity.v10'
OLEAN_REL='formal/lean4/.lake/build/lib/lean/SounioPireusOperatorDiscoveryEngine.olean'
ILEAN_REL='formal/lean4/.lake/build/lib/lean/SounioPireusOperatorDiscoveryEngine.ilean'

SOUNIO_SHA256='919b6104cbce1c5f8643f5df88b9071305d3fee854f785ac63a883bc45f16117'
FREEZE_SHA256='9a83c9a4b920d41ee91bd7681f4e95ac11480d762185ec9ff003692d3c01d247'
PARITY_OPEN_SHA256='5f109404d2a2e8e56e6cff486f871e0961f843edd2e48e2feb5f5717d1d8d39d'
SEMANTICS_SHA256='2640bb928740ef03f5a42725f42c62735bc2121621bb3dfd4b4cdf3572003ec5'
SOURCE_MANIFEST_SHA256='9fc99dc31a61c66a5cd9e45f93e9604013397e86ecfddcd2cd67b01f28703894'
LEAN_SHA256='c428bb157ba23dff96389c46723a26e167c25cd87dfda442db45e3d66be14276'
AUDIT_SHA256='681e85efbcc97c096d4f079856ddff5bcd9ba6db76d52360c3313d09f516c373'
LAKE_SHA256='adfc368721a675a2373a05ea0f0bc282e8e8c40f1f46efa220a9dc9c6527de3b'
EVIDENCE_SHA256='0ee87df139114acd6dfc7735cc874e23f0b79def65fe25253ef1e374eab2fcee'
RECEIPT_SHA256='dddc85352de064baeee09da91917ecc3790ac5fd362ba29b4dc204d86addaa30'
OLEAN_SHA256='9d059e477791345217bd6a38a66eea422c1403e1e16b08a55c8084fa8987cf57'
ILEAN_SHA256='2b32b3b26541cdc40d96e8bddc3666ec2e67da2ab57283c6671e8ddf1a120384'
TOOLCHAIN_SHA256='03526d62a2416b90e4cad0c369f73bbed9f38f700e0182f220f511235548bf63'
LEAN_BINARY_SHA256='19d38963260cfb376f1aab0f0fbcf4e80ec25c8bd0ba3b1797d95141d56ec55a'
LAKE_BINARY_SHA256='19d38963260cfb376f1aab0f0fbcf4e80ec25c8bd0ba3b1797d95141d56ec55a'
HARDWARE_SHA256='ba649e3df628e482654987fc29b86489bfdfc9f69f6bfdc141f3a8d238ee33cd'
BUILD_COMMAND_SHA256='e4570f4724da654c98635dd9c26fb870d8696fa299c338f15c4b93e9280430e5'
AUDIT_COMMAND_SHA256='6934c3d9de80a14ce29d558fcbe4d89d9b807d5ed4acb57ea59ab72e38a99307'
BUILD_FRAME_SHA256='3a0948b29b79d3e000bfd83853cf7f527a1889182c76aecd9f160e5711093385'
AUDIT_FRAME_SHA256='50b8c5396129b597528301c707c4a06a856a84f41672029fb93a1c7bfdbf0234'
SEAL_FRAME_SHA256='1c7161e9683cc182831df6d45739547639d3ec0b99f954c86d99d49e44246647'
WRITE_FRAME_SHA256='58ea94caf4c00a1234e153d5c75800fe69f9ebc1c78b799597f136702c605d5b'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
GUARDIAN_POLICY_SHA256='64bb0118793fe46dcb392abc1a9212eb15bd55047461576a3ef1a6cefa3f17da'
PYTHON_TOOLCHAIN_SHA256='497ce0938df96d9bf3c159472d251a946c2d6bd832220937d1b885f7759b05ba'
PYTHON_COMMAND_SHA256='72566473f0019fb50e65175a0ac019af5ad2f495ccfc30b664b7735791e968fc'
ZERO='0 0 0 0 0 0 0 0'
FRAME_WORDS=82

fail() {
  printf 'pireus operator discovery formal parity: FAIL: %s\n' "$*" >&2
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

parity_frame() {
  local stage="$1" action="$2" language="$3" role="$4" policy="$5"
  local semantic_write="$6" expected_write="$7" parity_valid="$8"
  local review_promoted="$9" parent_hash="${10}" command_hash="${11}"
  local result_hash="${12}" toolchain_hash="${13}" result_limbs="${ZERO}"
  if [[ "${result_hash}" != zero ]]; then
    result_limbs="$(sha_limbs "${result_hash}")"
  fi
  printf '9020 %s %s %s %s %s %s %s %s %s 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${stage}" "${action}" "${language}" "${role}" "${policy}" \
    "${semantic_write}" "${expected_write}" "${parity_valid}" \
    "${review_promoted}" "$(sha_limbs "${SOURCE_MANIFEST_SHA256}")" \
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
  local label="$1" frame="$2" expected_rc="$3" expected="$4" hash
  hash="$(sha_text "${frame}")"
  authorize "${label}" "${frame}" "${hash}" "${expected_rc}" "${expected}"
  printf 'GUARDIAN_DISPATCH label=%s process_launched=false\n' "${label}"
}

receipt_admitted() {
  local path="$1" key
  for key in status stage formal_obligations formal_obligations_discharged \
    formal_parity_complete effect_parity_complete material_parity_complete \
    n3_novelty n4_novelty material_novelty historical_novelty priority_claim \
    claim_ready; do
    [[ "$(grep -c "^${key}=" "${path}")" -eq 1 ]] || return 1
  done
  grep -Fqx 'status=FORMAL_PARITY_COMPLETE' "${path}" &&
    grep -Fqx 'stage=PARITY_OPEN' "${path}" &&
    grep -Fqx 'formal_obligations=6' "${path}" &&
    grep -Fqx 'formal_obligations_discharged=6' "${path}" &&
    grep -Fqx 'formal_parity_complete=true' "${path}" &&
    grep -Fqx 'effect_parity_complete=false' "${path}" &&
    grep -Fqx 'material_parity_complete=false' "${path}" &&
    grep -Fqx 'n3_novelty=false' "${path}" &&
    grep -Fqx 'n4_novelty=false' "${path}" &&
    grep -Fqx 'material_novelty=false' "${path}" &&
    grep -Fqx 'historical_novelty=false' "${path}" &&
    grep -Fqx 'priority_claim=false' "${path}" &&
    grep -Fqx 'claim_ready=false' "${path}"
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
require_hash "${ROOT}/stdlib/coordination/loom_language_authority.sio" \
  "${GUARDIAN_POLICY_SHA256}"
[[ -x "${GUARDIAN}" ]] || fail 'native Loom Guardian unavailable'

source_manifest="$({
  cd "${ROOT}"
  sha256sum "${LEAN_REL}" "${AUDIT_REL}" "${LAKE_REL}"
})"
[[ "$(sha_text "${source_manifest}")" == "${SOURCE_MANIFEST_SHA256}" ]] ||
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
[[ "${invalid_hash_rc}" -eq 1 ]] ||
  fail "malformed SHA-256 returned rc=${invalid_hash_rc}"
[[ "${invalid_hash_output}" == \
  'pireus operator discovery formal parity: FAIL: invalid sha256 digest: not-a-sha256' ]] ||
  fail 'malformed SHA-256 did not fail closed'
printf 'GUARDIAN_DISPATCH label=MALFORMED_SHA256 process_launched=false\n'

wrong_parent='1640bb928740ef03f5a42725f42c62735bc2121621bb3dfd4b4cdf3572003ec5'
deny_without_dispatch PARENT_LAUNDERING \
  "$(parity_frame 3 4 2 2 1 0 0 0 0 "${wrong_parent}" \
    "${BUILD_COMMAND_SHA256}" zero "${TOOLCHAIN_SHA256}")" 117 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=117 reason=parent-semantics-hash-mismatch next_stage=SEMANTICS_FROZEN'
deny_without_dispatch LEAN_SEMANTIC_WRITE \
  "$(parity_frame 3 4 2 2 1 1 0 0 0 "${SEMANTICS_SHA256}" \
    "${BUILD_COMMAND_SHA256}" zero "${TOOLCHAIN_SHA256}")" 113 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=SEMANTICS_FROZEN'
deny_without_dispatch LEAN_EXPECTED_RESULT_WRITE \
  "$(parity_frame 3 4 2 2 1 0 1 0 0 "${SEMANTICS_SHA256}" \
    "${BUILD_COMMAND_SHA256}" zero "${TOOLCHAIN_SHA256}")" 114 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=114 reason=expected-result-authority-required next_stage=SEMANTICS_FROZEN'
deny_without_dispatch REVIEW_PROMOTION \
  "$(parity_frame 3 4 2 2 1 0 0 0 1 "${SEMANTICS_SHA256}" \
    "${BUILD_COMMAND_SHA256}" zero "${TOOLCHAIN_SHA256}")" 119 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=119 reason=review-promoted-to-authority next_stage=SEMANTICS_FROZEN'
deny_without_dispatch POLICY_MISSING \
  "$(parity_frame 3 4 2 2 0 0 0 0 0 "${SEMANTICS_SHA256}" \
    "${BUILD_COMMAND_SHA256}" zero "${TOOLCHAIN_SHA256}")" 101 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=101 reason=policy-missing next_stage=SEMANTICS_FROZEN'
deny_without_dispatch POLICY_TIMEOUT \
  "$(parity_frame 3 4 2 2 2 0 0 0 0 "${SEMANTICS_SHA256}" \
    "${BUILD_COMMAND_SHA256}" zero "${TOOLCHAIN_SHA256}")" 102 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=102 reason=policy-timeout next_stage=SEMANTICS_FROZEN'
deny_without_dispatch POLICY_ERROR \
  "$(parity_frame 3 4 2 2 3 0 0 0 0 "${SEMANTICS_SHA256}" \
    "${BUILD_COMMAND_SHA256}" zero "${TOOLCHAIN_SHA256}")" 103 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=103 reason=policy-error next_stage=SEMANTICS_FROZEN'
deny_without_dispatch PYTHON_ORACLE \
  "$(parity_frame 3 4 7 7 1 0 0 0 0 "${SEMANTICS_SHA256}" \
    "${PYTHON_COMMAND_SHA256}" zero "${PYTHON_TOOLCHAIN_SHA256}")" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'

authorize BUILD_PREEXEC \
  "$(parity_frame 3 4 2 2 1 0 0 0 0 "${SEMANTICS_SHA256}" \
    "${BUILD_COMMAND_SHA256}" zero "${TOOLCHAIN_SHA256}")" \
  "${BUILD_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
(cd "${ROOT}/formal/lean4" && lake build SounioPireusOperatorDiscoveryEngine)
require_hash "${ROOT}/${OLEAN_REL}" "${OLEAN_SHA256}"
require_hash "${ROOT}/${ILEAN_REL}" "${ILEAN_SHA256}"

authorize AXIOM_AUDIT_PREEXEC \
  "$(parity_frame 3 4 2 2 1 0 0 0 0 "${SEMANTICS_SHA256}" \
    "${AUDIT_COMMAND_SHA256}" zero "${TOOLCHAIN_SHA256}")" \
  "${AUDIT_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
axiom_output="$(
  cd "${ROOT}/formal/lean4" &&
    lake env lean -j 1 SounioPireusOperatorDiscoveryEngineAxiomAudit.lean 2>&1
)"
[[ "$(grep -c ' depends on axioms:' <<<"${axiom_output}")" -eq 6 ]] ||
  fail 'axiom audit theorem count drift'
[[ "$(grep -c 'native_decide.ax_1_1' <<<"${axiom_output}")" -eq 6 ]] ||
  fail 'native_decide axiom profile drift'
[[ "$(grep -c 'propext' <<<"${axiom_output}")" -eq 6 ]] ||
  fail 'propext axiom profile drift'
printf 'LEAN_AXIOM_AUDIT theorems=6 propext=6 native_decide=6 unexpected=0 closure=EXPLICIT_NATIVE_DECIDE_TRUST_BOUNDARY\n'

authorize SEAL \
  "$(parity_frame 4 8 2 2 1 0 0 1 0 "${SEMANTICS_SHA256}" \
    "${BUILD_COMMAND_SHA256}" "${OLEAN_SHA256}" "${TOOLCHAIN_SHA256}")" \
  "${SEAL_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
authorize RECEIPT_WRITE \
  "$(parity_frame 4 9 2 2 1 0 0 1 0 "${SEMANTICS_SHA256}" \
    "${BUILD_COMMAND_SHA256}" "${OLEAN_SHA256}" "${TOOLCHAIN_SHA256}")" \
  "${WRITE_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'

require_hash "${ROOT}/${EVIDENCE_REL}" "${EVIDENCE_SHA256}"
require_hash "${ROOT}/${RECEIPT_REL}" "${RECEIPT_SHA256}"
receipt_admitted "${ROOT}/${RECEIPT_REL}" || fail 'formal receipt drifted during gate'

printf '%s\n' \
  'pireus operator discovery formal parity: STAGE_REACHED_NOT_A_CLAIM stage=PARITY_OPEN language=Lean4 role=FORMAL_PARITY obligations=6/6 atlas_classes=3 actions=2 action_cells=49152 separators=272:0:0:257:272:0 collision_control=EXACT incomplete_control=EXACT law_spectrum=112:824 axiom_closure=EXPLICIT_NATIVE_DECIDE_TRUST_BOUNDARY formal=COMPLETE effect=OPEN_NOT_EXECUTED material=OPEN_NOT_EXECUTED n3=false n4=false historical_novelty=false priority_claim=false claim_ready=false python_process_launched=false rust_process_launched=false'
