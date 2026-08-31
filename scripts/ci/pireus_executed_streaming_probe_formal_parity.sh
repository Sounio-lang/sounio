#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN_CANDIDATE="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
GUARDIAN="$(readlink -f "${GUARDIAN_CANDIDATE}")"

SOUNIO_REL='stdlib/hardware/pireus/operator_orbit_canonicalization.sio'
FREEZE_REL='tools/pireus/operator_orbit_canonicalization.freeze.v13'
TRANSCRIPT_REL='tools/pireus/operator_orbit_canonicalization.first.v13'
PARENT_GATE_REL='scripts/ci/pireus_streaming_minimum_correspondence_formal_parity.sh'
PARENT_RECEIPT_REL='tools/pireus/streaming_minimum_correspondence.formal-parity.v13'
PARENT_EVIDENCE_REL='tools/pireus/evidence/streaming_minimum_correspondence_v13.formal-parity.txt'
BASE_REL='formal/lean4/SounioPireusExecutedStreamingProbe.lean'
CERTIFICATE_REL='formal/lean4/SounioPireusExecutedStreamingProbeCertificate.lean'
CHECK_REL='formal/lean4/SounioPireusExecutedStreamingProbeCheck.lean'
AUDIT_REL='formal/lean4/SounioPireusExecutedStreamingProbeAxiomAudit.lean'
LAKEFILE_REL='formal/lean4/lakefile.lean'
OFFLOAD_LOG_REL='.claude/llm_offload_log.md'
GATE_REL='scripts/ci/pireus_executed_streaming_probe_formal_parity.sh'
RECEIPT_REL='tools/pireus/executed_streaming_probe.formal-parity.v13'
EVIDENCE_REL='tools/pireus/evidence/executed_streaming_probe_v13.formal-parity.txt'

MATCHER_FREE_EXECUTABLE_COMMIT='73704f7afed6780c3a317b739cbd35fe94dbe395'
FIRST_EVIDENCE_COMMIT='22fbabe81cf365c0b542d8a425ec4c081f31e390'
FREEZE_COMMIT='aa9585e2c36e1e6580045b77904d8c5987799c4d'
PARENT_GATE_COMMIT='3ce522d76ed40d53e7bd5501a2ca84c029c9d721'
FORMAL_SOURCE_COMMIT='4a41eb10b8f94cb558af82b5fae63f54bba38224'
ARTIFACT_COMMIT='PENDING_ARTIFACT_COMMIT'
PRESEAL_GATE_SHA256='PENDING_PRESEAL_GATE_SHA256'
ARTIFACT_GATE_ANCHOR_COMMIT='PENDING_ARTIFACT_COMMIT'
ARTIFACT_GATE_ANCHOR_PRESEAL_SHA256='PENDING_PRESEAL_GATE_SHA256'

MATCHER_FREE_SOURCE_SHA256='3136968a83bbba18d56c543895d6bbd9530ccf6c59db78ac6b6f2fa3bd26c9e4'
FROZEN_MATCHER_SOURCE_SHA256='7ada1b17bf91fdb3f4c48877d2485f71a65bb4159d88cb7e4b288c77bfe3cdae'
FREEZE_SHA256='11893a34450729ff06ac40ade86c90decb7a6947daea3cc108cae17f73572f84'
TRANSCRIPT_SHA256='16af63f5e0f8aa7e5c899f4c395404b83fb402f6bbdb5f20dea2a3d10ad2e19f'
PARENT_GATE_SHA256='513674e0a9fc62cac80fc1b0045ef790f8a81def7c65b02122c44a2b0db4d44a'
PARENT_RECEIPT_SHA256='92bbde468d6e091739366b7417edaefa78a05d5d08b3175ac05db5053686da22'
PARENT_EVIDENCE_SHA256='f360f3fd3e966bc6b02c156b390a9538aad602e9d7bbe3209464028f7ad86f04'
BASE_SHA256='1dd92c1e70e7d9c01495c5c2e09ecc9578e6b18130f6ac06f5d18a09a023b9c4'
CERTIFICATE_SHA256='c8b71c21ad808cef1c4843f93842d149809131cd795b407658b40b1fa001889c'
CHECK_SHA256='2e40445f8815624f05a0aefcdccac99c77865c087bc7c0cb5da18155eb799ef0'
AUDIT_SHA256='11619fbffc517f5fc557c35eea4f5835f5df3be141b1155b98c64aa803bc1c14'
LAKEFILE_SHA256='7992ce727698567504989f963c46e89b0ba9d0cdf79b3ecb5859f2da831506b1'
FORMAL_SOURCE_REVIEW_LOG_SHA256='1a7ebf29f2550fc0da830b1daac198ee4f85463991ff79d7359e88702e987917'
RECEIPT_SHA256='256589f83f28c0638fccf3bdb6e04bfe8d625f3a941eeebf7feba88db07c0e9e'
EVIDENCE_SHA256='363ba50ff988f92c06e82881c8ab790545d0ec2ca60b2ad4cbd99293a3f9fa34'
SOUNIO_SEMANTICS_SHA256='0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c'
TOOLCHAIN_SHA256='a0786d46c580b27dad335f46e9c12a69cb6b59db438097f2ad4fe0d9eced1a6c'
HARDWARE_SHA256='e764b122c1b973fc54fe6c781bd92cc7c394f0dad4d895812f88f7d15af50d23'
LIVE_HARDWARE_MANIFEST_SHA256='ed98ba37afb72f73ed32b8d84fa17a221b5bb8483df454ffe870b65a913f1b7a'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
PYTHON_TOOLCHAIN_SHA256='422c73f42aff0794a916a210455ea8a5d2bbbd29430963b94c29966017abc517'
EXPECTED_RAW_HEX='0x00003e3c693c330f180f6200176654667f7c7c291c7f1ffe076f044864cb67d2'
EXPECTED_CANONICAL_HEX='0x00000004617a56057d2e6a13294d57496b0e7cb017b259955561265e4bda64e4'
ZERO='0 0 0 0 0 0 0 0'

BLOCK_RELS=(
  formal/lean4/SounioPireusExecutedStreamingProbeBlocks0.lean
  formal/lean4/SounioPireusExecutedStreamingProbeBlocks1.lean
  formal/lean4/SounioPireusExecutedStreamingProbeBlocks2.lean
  formal/lean4/SounioPireusExecutedStreamingProbeBlocks3.lean
  formal/lean4/SounioPireusExecutedStreamingProbeBlocks4.lean
  formal/lean4/SounioPireusExecutedStreamingProbeBlocks5.lean
  formal/lean4/SounioPireusExecutedStreamingProbeBlocks6.lean
  formal/lean4/SounioPireusExecutedStreamingProbeBlocks7.lean
)
BLOCK_SHA256=(
  39d1dd975d0433a7534c8f6785c2796e16d78a4d3169c47ba92c56d6c772c909
  3ea985bcd3adfa549f810b528b4150148a3e483579888f238652c2dfda12a160
  cac277183a09f6232539808a72704a9329ed85e53e0d61fe09e98003e608ed91
  1ebc756743edaad9634685c16bf02d89c9c269c8ea1d1501c83c3b83600df974
  3f8ce7d632064d600bd8a778270c698736e1dfd6cb0f6c56dd0abfa0791c717f
  f1263751d4074492958f5d4876721364c332f46ae3e5f7776e8eca1f14edd352
  551bb16d6bd153a144b935ef839694bcb498cd8e761648899cffa4405d699409
  d851c0927435c8aa8d1c99ce1620d59542b8fd82c3c5f1929756319f61f6ddd6
)

fail() {
  printf 'pireus executed streaming probe formal parity: FAIL: %s\n' "$*" >&2
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

record_value() {
  local path="$1" key="$2"
  awk -F= -v key="${key}" '
    $1 == key { count++; sub(/^[^=]*=/, ""); value=$0 }
    END { if (count != 1) exit 2; print value }
  ' "${path}" || fail "missing or duplicate record key ${key} in ${path}"
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
  local source_sha="$1" command_sha="$2"
  printf '9020 4 4 7 7 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${source_sha}")" "$(sha_limbs "${SOUNIO_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SOUNIO_SEMANTICS_SHA256}")" "$(sha_limbs "${PYTHON_TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" "$(sha_limbs "${command_sha}")" "${ZERO}" "${ZERO}"
}

authorize() {
  local label="$1" frame="$2" receipt_key="$3" decision frame_sha
  [[ "$(wc -w <<<"${frame}" | tr -d ' ')" -eq 82 ]] || fail "Guardian frame words: ${label}"
  frame_sha="$(sha_text "${frame}")"
  require_line "${ROOT}/${RECEIPT_REL}" "${receipt_key}=${frame_sha}"
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  [[ "${decision}" == 'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN' ]] ||
    fail "Guardian decision drift: ${label}: ${decision}"
  printf 'GUARDIAN_DECISION label=%s decision=%s\n' "${label}" "${decision}"
}

extract_bits() {
  local header="$1" path="$2"
  awk -v header="${header}" '
    $0 == header { seen++; active=1; next }
    active && /^ bits=[01]$/ { bits=bits substr($0, 7, 1); next }
    active && /^:[01]$/ { bits=bits substr($0, 2, 1); next }
    active { active=0 }
    END { if (seen != 1 || length(bits) != 256) exit 2; print bits }
  ' "${path}" || fail "invalid 256-bit transcript record: ${header}"
}

bits_to_hex() {
  local bits="$1"
  awk -v bits="${bits}" '
    BEGIN {
      hex["0000"]="0"; hex["0001"]="1"; hex["0010"]="2"; hex["0011"]="3";
      hex["0100"]="4"; hex["0101"]="5"; hex["0110"]="6"; hex["0111"]="7";
      hex["1000"]="8"; hex["1001"]="9"; hex["1010"]="a"; hex["1011"]="b";
      hex["1100"]="c"; hex["1101"]="d"; hex["1110"]="e"; hex["1111"]="f";
      if (length(bits) != 256) exit 2;
      printf "0x";
      for (i=1; i<=256; i+=4) {
        nibble=substr(bits, i, 4);
        if (!(nibble in hex)) exit 3;
        printf "%s", hex[nibble];
      }
      printf "\n";
    }
  ' || fail 'mechanical bit-to-hex encoding failed'
}

extract_admitted_field() {
  local key="$1" path="$2"
  awk -v key="${key}" '
    $0 == "PIREUS_POC_ADMITTED admitted=0" { seen++; active=1; next }
    active && $0 == "" { active=0 }
    active && index($0, " " key "=") == 1 {
      found++;
      value=substr($0, length(key) + 3)
    }
    END { if (seen != 1 || found != 1) exit 2; print value }
  ' "${path}" || fail "invalid admitted=0 field: ${key}"
}

cd "${ROOT}"
[[ "${ARTIFACT_COMMIT}" != "${ARTIFACT_GATE_ANCHOR_COMMIT}" ]] || fail 'artifact commit is not sealed'
[[ "${PRESEAL_GATE_SHA256}" != "${ARTIFACT_GATE_ANCHOR_PRESEAL_SHA256}" ]] || fail 'preseal gate hash is not sealed'

require_hash "${ROOT}/${SOUNIO_REL}" "${FROZEN_MATCHER_SOURCE_SHA256}"
require_hash "${ROOT}/${FREEZE_REL}" "${FREEZE_SHA256}"
require_hash "${ROOT}/${TRANSCRIPT_REL}" "${TRANSCRIPT_SHA256}"
require_hash "${ROOT}/${PARENT_GATE_REL}" "${PARENT_GATE_SHA256}"
require_hash "${ROOT}/${PARENT_RECEIPT_REL}" "${PARENT_RECEIPT_SHA256}"
require_hash "${ROOT}/${PARENT_EVIDENCE_REL}" "${PARENT_EVIDENCE_SHA256}"
require_hash "${ROOT}/${BASE_REL}" "${BASE_SHA256}"
require_hash "${ROOT}/${CERTIFICATE_REL}" "${CERTIFICATE_SHA256}"
require_hash "${ROOT}/${CHECK_REL}" "${CHECK_SHA256}"
require_hash "${ROOT}/${AUDIT_REL}" "${AUDIT_SHA256}"
require_hash "${ROOT}/${LAKEFILE_REL}" "${LAKEFILE_SHA256}"
require_hash "${ROOT}/${RECEIPT_REL}" "${RECEIPT_SHA256}"
require_hash "${ROOT}/${EVIDENCE_REL}" "${EVIDENCE_SHA256}"
require_hash "${GUARDIAN}" "${GUARDIAN_SHA256}"

for i in "${!BLOCK_RELS[@]}"; do
  require_hash "${ROOT}/${BLOCK_RELS[$i]}" "${BLOCK_SHA256[$i]}"
done

git -C "${ROOT}" merge-base --is-ancestor "${MATCHER_FREE_EXECUTABLE_COMMIT}" "${FIRST_EVIDENCE_COMMIT}" || fail 'executable does not precede first evidence'
git -C "${ROOT}" merge-base --is-ancestor "${FIRST_EVIDENCE_COMMIT}" "${FREEZE_COMMIT}" || fail 'first evidence does not precede freeze'
git -C "${ROOT}" merge-base --is-ancestor "${FREEZE_COMMIT}" "${PARENT_GATE_COMMIT}" || fail 'freeze does not precede parent parity gate'
git -C "${ROOT}" merge-base --is-ancestor "${PARENT_GATE_COMMIT}" "${FORMAL_SOURCE_COMMIT}" || fail 'parent parity gate does not precede executed probe source'
git -C "${ROOT}" merge-base --is-ancestor "${FORMAL_SOURCE_COMMIT}" "${ARTIFACT_COMMIT}" || fail 'formal source does not precede artifact seal'
git -C "${ROOT}" merge-base --is-ancestor "${ARTIFACT_COMMIT}" HEAD || fail 'artifact seal not in current history'

require_committed_hash "${MATCHER_FREE_EXECUTABLE_COMMIT}" "${SOUNIO_REL}" "${MATCHER_FREE_SOURCE_SHA256}"
require_committed_hash "${FIRST_EVIDENCE_COMMIT}" "${TRANSCRIPT_REL}" "${TRANSCRIPT_SHA256}"
require_committed_hash "${FREEZE_COMMIT}" "${FREEZE_REL}" "${FREEZE_SHA256}"
require_committed_hash "${FREEZE_COMMIT}" "${SOUNIO_REL}" "${FROZEN_MATCHER_SOURCE_SHA256}"
require_committed_hash "${PARENT_GATE_COMMIT}" "${PARENT_GATE_REL}" "${PARENT_GATE_SHA256}"
require_committed_hash "${PARENT_GATE_COMMIT}" "${PARENT_RECEIPT_REL}" "${PARENT_RECEIPT_SHA256}"
require_committed_hash "${PARENT_GATE_COMMIT}" "${PARENT_EVIDENCE_REL}" "${PARENT_EVIDENCE_SHA256}"
require_committed_hash "${FORMAL_SOURCE_COMMIT}" "${BASE_REL}" "${BASE_SHA256}"
require_committed_hash "${FORMAL_SOURCE_COMMIT}" "${CERTIFICATE_REL}" "${CERTIFICATE_SHA256}"
require_committed_hash "${FORMAL_SOURCE_COMMIT}" "${CHECK_REL}" "${CHECK_SHA256}"
require_committed_hash "${FORMAL_SOURCE_COMMIT}" "${AUDIT_REL}" "${AUDIT_SHA256}"
require_committed_hash "${FORMAL_SOURCE_COMMIT}" "${OFFLOAD_LOG_REL}" "${FORMAL_SOURCE_REVIEW_LOG_SHA256}"
for i in "${!BLOCK_RELS[@]}"; do
  require_committed_hash "${FORMAL_SOURCE_COMMIT}" "${BLOCK_RELS[$i]}" "${BLOCK_SHA256[$i]}"
done
require_committed_hash "${ARTIFACT_COMMIT}" "${RECEIPT_REL}" "${RECEIPT_SHA256}"
require_committed_hash "${ARTIFACT_COMMIT}" "${EVIDENCE_REL}" "${EVIDENCE_SHA256}"
require_committed_hash "${ARTIFACT_COMMIT}" "${GATE_REL}" "${PRESEAL_GATE_SHA256}"

normalized_live_gate_sha256="$(
  sed \
    -e "s/^ARTIFACT_COMMIT=.*/ARTIFACT_COMMIT='${ARTIFACT_GATE_ANCHOR_COMMIT}'/" \
    -e "s/^PRESEAL_GATE_SHA256=.*/PRESEAL_GATE_SHA256='${ARTIFACT_GATE_ANCHOR_PRESEAL_SHA256}'/" \
    "${ROOT}/${GATE_REL}" | sha256sum | cut -d' ' -f1
)"
[[ "${normalized_live_gate_sha256}" == "${PRESEAL_GATE_SHA256}" ]] || fail 'executing gate bytes are not the sealed two-field transformation'

semantics_sha256="$(
  sed -n '/^semantics_material_begin$/,/^semantics_material_end$/p' "${ROOT}/${FREEZE_REL}" |
    sed '1d;$d' | sha256sum | cut -d' ' -f1
)"
[[ "${semantics_sha256}" == "${SOUNIO_SEMANTICS_SHA256}" ]] || fail 'Sounio semantics digest drift'
require_line "${ROOT}/${FREEZE_REL}" "matcher_free_executable_commit=${MATCHER_FREE_EXECUTABLE_COMMIT}"
require_line "${ROOT}/${FREEZE_REL}" "matcher_free_module_sha256=${MATCHER_FREE_SOURCE_SHA256}"
require_line "${ROOT}/${FREEZE_REL}" "matcher_module_sha256=${FROZEN_MATCHER_SOURCE_SHA256}"
require_line "${ROOT}/${FREEZE_REL}" "first_transcript_sha256=${TRANSCRIPT_SHA256}"
require_line "${ROOT}/${FREEZE_REL}" 'source_changed_after_first_result=true'
require_line "${ROOT}/${FREEZE_REL}" 'source_change_scope=FROZEN_EXPECTED_COUNTERS_DIGESTS_AND_FALSE_BOUNDARY_MATCHER_ONLY'

source_manifest="$(printf '%s\n' \
  "matcher_free_source@${MATCHER_FREE_EXECUTABLE_COMMIT}=${MATCHER_FREE_SOURCE_SHA256}" \
  "frozen_matcher_source=${FROZEN_MATCHER_SOURCE_SHA256}" "freeze=${FREEZE_SHA256}" \
  "first_transcript@${FIRST_EVIDENCE_COMMIT}=${TRANSCRIPT_SHA256}" \
  "${BASE_REL}=${BASE_SHA256}" "${BLOCK_RELS[0]}=${BLOCK_SHA256[0]}" \
  "${BLOCK_RELS[1]}=${BLOCK_SHA256[1]}" "${BLOCK_RELS[2]}=${BLOCK_SHA256[2]}" \
  "${BLOCK_RELS[3]}=${BLOCK_SHA256[3]}" "${BLOCK_RELS[4]}=${BLOCK_SHA256[4]}" \
  "${BLOCK_RELS[5]}=${BLOCK_SHA256[5]}" "${BLOCK_RELS[6]}=${BLOCK_SHA256[6]}" \
  "${BLOCK_RELS[7]}=${BLOCK_SHA256[7]}" "${CERTIFICATE_REL}=${CERTIFICATE_SHA256}" \
  "${CHECK_REL}=${CHECK_SHA256}" "${AUDIT_REL}=${AUDIT_SHA256}")"
SOURCE_BUNDLE_SHA256="$(sha_text "${source_manifest}")"
require_line "${ROOT}/${RECEIPT_REL}" "formal_source_bundle_sha256=${SOURCE_BUNDLE_SHA256}"
require_line "${ROOT}/${EVIDENCE_REL}" "formal_source_bundle_sha256=${SOURCE_BUNDLE_SHA256}"

raw_bits="$(extract_bits 'PIREUS_POC_RAW_BITS admitted=0' "${ROOT}/${TRANSCRIPT_REL}")"
canonical_bits="$(extract_bits 'PIREUS_POC_CANON_BITS admitted=0' "${ROOT}/${TRANSCRIPT_REL}")"
raw_hex="$(bits_to_hex "${raw_bits}")"
canonical_hex="$(bits_to_hex "${canonical_bits}")"
matrix_code="$(extract_admitted_field matrix "${ROOT}/${TRANSCRIPT_REL}")"
swap="$(extract_admitted_field swap "${ROOT}/${TRANSCRIPT_REL}")"
gauge_word="$(extract_admitted_field gauge "${ROOT}/${TRANSCRIPT_REL}")"
[[ "${raw_hex}" == "${EXPECTED_RAW_HEX}" ]] || fail 'admitted=0 raw bits drift'
[[ "${canonical_hex}" == "${EXPECTED_CANONICAL_HEX}" ]] || fail 'admitted=0 canonical bits drift'
[[ "${matrix_code}" == '58475' ]] || fail 'admitted=0 matrix code drift'
[[ "${swap}" == '0' ]] || fail 'admitted=0 swap drift'
[[ "${gauge_word}" == '933' ]] || fail 'admitted=0 gauge word drift'
require_line "${ROOT}/${BASE_REL}" "  \"${MATCHER_FREE_SOURCE_SHA256}\""
require_line "${ROOT}/${BASE_REL}" "  \"${FROZEN_MATCHER_SOURCE_SHA256}\""
require_line "${ROOT}/${BASE_REL}" "  \"${TRANSCRIPT_SHA256}\""
require_line "${ROOT}/${BASE_REL}" "def admittedProbeMatrixCode : Nat := ${matrix_code}"
require_line "${ROOT}/${BASE_REL}" 'def admittedProbeSwap : Bool := false'
require_line "${ROOT}/${BASE_REL}" "def admittedProbeGaugeWord : Nat := ${gauge_word}"
require_line "${ROOT}/${BASE_REL}" "  ${raw_hex}"
require_line "${ROOT}/${BASE_REL}" "  ${canonical_hex}"
require_line "${ROOT}/${RECEIPT_REL}" "probe_raw_bits_hex=${raw_hex}"
require_line "${ROOT}/${RECEIPT_REL}" "probe_canonical_bits_hex=${canonical_hex}"
require_line "${ROOT}/${EVIDENCE_REL}" "probe_raw_bits_hex=${raw_hex}"
require_line "${ROOT}/${EVIDENCE_REL}" "probe_canonical_bits_hex=${canonical_hex}"

[[ "$(lean --version | head -1)" == 'Lean (version 4.33.1, x86_64-unknown-linux-gnu, commit 819816b2e0a3bf405af45ae5c7af2491d8f5bee6, Release)' ]] || fail 'Lean version drift'
[[ "$(lake --version)" == 'Lake version 5.0.0-src+819816b (Lean version 4.33.1)' ]] || fail 'Lake version drift'
[[ "$(uname -m)" == 'x86_64' ]] || fail 'architecture drift'
live_hardware_manifest="$(printf '%s\n' \
  'execution_route=LOCAL_XEON_WORKSPACE_CONTROL' \
  "execution_node=$(hostname)" "execution_architecture=$(uname -m)" \
  "execution_cpu=$(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1 | tr ' ' '_')" \
  "execution_logical_cpu_count=$(lscpu | sed -n 's/^CPU(s):[[:space:]]*//p' | head -1)" \
  "execution_socket_count=$(lscpu | sed -n 's/^Socket(s):[[:space:]]*//p' | head -1)" \
  "execution_cores_per_socket=$(lscpu | sed -n 's/^Core(s) per socket:[[:space:]]*//p' | head -1)" \
  "execution_threads_per_core=$(lscpu | sed -n 's/^Thread(s) per core:[[:space:]]*//p' | head -1)")"
[[ "$(sha_text "${live_hardware_manifest}")" == "${LIVE_HARDWARE_MANIFEST_SHA256}" ]] || fail 'live hardware manifest drift'

BUILD_COMMAND="$(record_value "${ROOT}/${EVIDENCE_REL}" build_command)"
CHECK_COMMAND="$(record_value "${ROOT}/${EVIDENCE_REL}" proof_check_command)"
AUDIT_COMMAND="$(record_value "${ROOT}/${EVIDENCE_REL}" axiom_audit_command)"
PYTHON_COMMAND="$(record_value "${ROOT}/${EVIDENCE_REL}" python_negative_command)"
BUILD_COMMAND_SHA256="$(sha_text "${BUILD_COMMAND}")"
CHECK_COMMAND_SHA256="$(sha_text "${CHECK_COMMAND}")"
AUDIT_COMMAND_SHA256="$(sha_text "${AUDIT_COMMAND}")"
PYTHON_COMMAND_SHA256="$(sha_text "${PYTHON_COMMAND}")"
require_line "${ROOT}/${RECEIPT_REL}" "build_command_sha256=${BUILD_COMMAND_SHA256}"
require_line "${ROOT}/${RECEIPT_REL}" "proof_check_command_sha256=${CHECK_COMMAND_SHA256}"
require_line "${ROOT}/${RECEIPT_REL}" "axiom_audit_command_sha256=${AUDIT_COMMAND_SHA256}"
require_line "${ROOT}/${RECEIPT_REL}" "python_command_sha256=${PYTHON_COMMAND_SHA256}"

[[ "$(grep -Ec '^theorem ' "${ROOT}/${BASE_REL}")" -eq 6 ]] || fail 'base theorem count drift'
[[ "$(grep -Ec '^def ' "${ROOT}/${BASE_REL}")" -eq 22 ]] || fail 'base definition count drift'
[[ "$(grep -Eh '^theorem ' "${BLOCK_RELS[@]/#/${ROOT}/}" | wc -l | tr -d ' ')" -eq 64 ]] || fail 'block theorem count drift'
[[ "$(grep -Ec '^theorem ' "${ROOT}/${CERTIFICATE_REL}")" -eq 13 ]] || fail 'certificate theorem count drift'
[[ "$(grep -Ec '^(def|structure) ' "${ROOT}/${CERTIFICATE_REL}")" -eq 4 ]] || fail 'certificate definition count drift'
[[ "$(grep -Ec '^example ' "${ROOT}/${CHECK_REL}")" -eq 11 ]] || fail 'proof-check obligation count drift'
[[ "$(grep -c '^#print axioms ' "${ROOT}/${AUDIT_REL}")" -eq 83 ]] || fail 'axiom audit inventory count drift'

for i in $(seq -w 0 63); do
  theorem_name="code_block_${i}_dominates"
  grep -Fqx "theorem ${theorem_name} :" "${ROOT}/formal/lean4/SounioPireusExecutedStreamingProbeBlocks$((10#${i}/8)).lean" || fail "missing block theorem: ${theorem_name}"
  grep -Fqx "#print axioms ${theorem_name}" "${ROOT}/${AUDIT_REL}" || fail "missing block theorem audit: ${theorem_name}"
done
require_line "${ROOT}/${CERTIFICATE_REL}" 'theorem all_code_block_views_count_is_40320 :'
require_line "${ROOT}/${CERTIFICATE_REL}" 'theorem admitted_probe_streaming_minimum_eq_packaged_canonical :'
require_line "${ROOT}/${CERTIFICATE_REL}" 'theorem admitted_probe_declared_canonical_eq_packaged_canonical :'
require_line "${ROOT}/${CERTIFICATE_REL}" '  , generalExecutedSounioStreamingEqualityProved := false'
require_line "${ROOT}/${CERTIFICATE_REL}" '  , formalParityClosed := false'
require_line "${ROOT}/${CERTIFICATE_REL}" '  , claimReady := false }'
[[ "$(grep -ho 'native_decide' "${ROOT}/${BASE_REL}" "${BLOCK_RELS[@]/#/${ROOT}/}" "${ROOT}/${CERTIFICATE_REL}" | wc -l | tr -d ' ')" -eq 71 ]] || fail 'source native_decide count drift'
[[ "$(grep -Eh '\bsorry\b|sorryAx' "${ROOT}/${BASE_REL}" "${BLOCK_RELS[@]/#/${ROOT}/}" "${ROOT}/${CERTIFICATE_REL}" "${ROOT}/${CHECK_REL}" || true)" == '' ]] || fail 'source sorry marker drift'

require_line "${ROOT}/${RECEIPT_REL}" 'native_decide_trust_assumed=true'
require_line "${ROOT}/${RECEIPT_REL}" 'native_decide_is_kernel_pure_claim=false'
require_line "${ROOT}/${RECEIPT_REL}" 'single_frozen_probe_execution_model_match=true'
require_line "${ROOT}/${RECEIPT_REL}" 'general_executed_sounio_streaming_equality=false'
require_line "${ROOT}/${RECEIPT_REL}" 'formal_parity_complete=false'
require_line "${ROOT}/${RECEIPT_REL}" 'claim_ready=false'
require_line "${ROOT}/${RECEIPT_REL}" 'spark_route_policy=KUBERNETES_ONLY'
require_line "${ROOT}/${RECEIPT_REL}" 'u250_declared_card_count=2'
require_line "${ROOT}/${RECEIPT_REL}" 'u250_installed_card_count=1'
require_line "${ROOT}/${RECEIPT_REL}" 'u250_pending_installation_card_count=1'
require_line "${ROOT}/${RECEIPT_REL}" 'u250_enumeration_failure_count=0'
require_line "${ROOT}/${RECEIPT_REL}" 'result_enum=V13_FROZEN_SOUNIO_PROBE0_MATCHES_LEAN_MODEL_NATIVE_TRUST_GENERAL_LINK_OPEN_NO_CLAIM'
[[ "$(awk '/=PENDING_/ { count++ } END { print count + 0 }' "${ROOT}/${RECEIPT_REL}" "${ROOT}/${EVIDENCE_REL}")" -eq 0 ]] || fail 'artifact placeholder remains'

python_frame="$(python_oracle_frame "${SOURCE_BUNDLE_SHA256}" "${PYTHON_COMMAND_SHA256}")"
require_line "${ROOT}/${RECEIPT_REL}" "python_preexec_frame_sha256=$(sha_text "${python_frame}")"
set +e
python_decision="$(printf '%s\n' "${python_frame}" | "${GUARDIAN}")"
python_rc=$?
set -e
[[ "${python_rc}" -eq 110 ]] || fail "Python oracle exit drift: ${python_rc}"
[[ "${python_decision}" == 'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=PARITY_OPEN' ]] || fail 'Python oracle decision drift'
[[ ! -e "${ROOT}/forbidden_executed_streaming_probe_oracle.py" ]] || fail 'forbidden Python oracle file exists'

negative_dir="$(mktemp -d /tmp/pireus-executed-streaming-negative.XXXXXX)"
trap 'rm -rf "${negative_dir}"' EXIT
build_frame="$(parity_frame "${SOURCE_BUNDLE_SHA256}" "${BUILD_COMMAND_SHA256}")"
set +e
(
  GUARDIAN=/bin/false
  authorize LOCAL_XEON_BUILD "${build_frame}" build_preexec_frame_sha256
  printf 'LEAN_STARTED\n' >"${negative_dir}/lean-started.txt"
) >"${negative_dir}/guardian-false.txt" 2>&1
negative_rc=$?
set -e
[[ "${negative_rc}" -eq 1 ]] || fail "Guardian override negative exit drift: ${negative_rc}"
[[ ! -e "${negative_dir}/lean-started.txt" ]] || fail 'Guardian override reached Lean'

authorize LOCAL_XEON_BUILD "${build_frame}" build_preexec_frame_sha256
set +e
build_output="$(bash -c "${BUILD_COMMAND}" 2>&1)"
build_rc=$?
set -e
[[ "${build_rc}" -eq 0 ]] || fail "Lean build exit drift: ${build_rc}"
[[ "${build_output}" != *'error:'* ]] || fail 'Lean build error marker drift'

check_frame="$(parity_frame "${CHECK_SHA256}" "${CHECK_COMMAND_SHA256}")"
authorize LOCAL_XEON_PROOF_CHECK "${check_frame}" proof_check_preexec_frame_sha256
check_output="$(bash -c "${CHECK_COMMAND}" 2>&1)"
[[ "$(count_occurrences '^warning:' "${check_output}")" -eq 0 ]] || fail 'proof check warning drift'

audit_frame="$(parity_frame "${AUDIT_SHA256}" "${AUDIT_COMMAND_SHA256}")"
authorize LOCAL_XEON_AXIOM_AUDIT "${audit_frame}" axiom_audit_preexec_frame_sha256
audit_output="$(bash -c "${AUDIT_COMMAND}" 2>&1)"
[[ "$(count_occurrences "^'SounioPireusExecutedStreamingProbe" "${audit_output}")" -eq 83 ]] || fail 'axiom report count drift'
[[ "$(count_occurrences 'depends on axioms:' "${audit_output}")" -eq 83 ]] || fail 'axiom-bearing count drift'
[[ "$(count_occurrences 'does not depend on any axioms' "${audit_output}")" -eq 0 ]] || fail 'axiom-free count drift'
[[ "$(count_occurrences 'propext' "${audit_output}")" -eq 81 ]] || fail 'propext count drift'
[[ "$(count_occurrences 'Classical.choice' "${audit_output}")" -eq 77 ]] || fail 'Classical.choice count drift'
[[ "$(count_occurrences 'Quot.sound' "${audit_output}")" -eq 81 ]] || fail 'Quot.sound count drift'
[[ "$(count_occurrences 'sorryAx' "${audit_output}")" -eq 0 ]] || fail 'sorryAx count drift'
axiom_blocks="$(tr '\n' ' ' <<<"${audit_output}" | grep -oE 'depends on axioms: \[[^]]+\]')"
[[ "$(grep -c '^depends on axioms:' <<<"${axiom_blocks}")" -eq 83 ]] || fail 'axiom parser coverage drift'
all_axioms="$(
  sed 's/^depends on axioms: \[//;s/\]$//' <<<"${axiom_blocks}" |
    tr ',' '\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | sort -u
)"
native_axioms="$(grep '_native\.native_decide\.ax_' <<<"${all_axioms}" || true)"
non_native_axioms="$(grep -v '_native\.native_decide\.ax_' <<<"${all_axioms}" || true)"
[[ "$(wc -l <<<"${native_axioms}" | tr -d ' ')" -eq 71 ]] || fail 'unique native_decide axiom count drift'
[[ "${non_native_axioms}" == $'Classical.choice\nQuot.sound\npropext' ]] || fail "unexpected non-native axiom set: ${non_native_axioms}"
for native_name in \
  admittedProbeWinnerEntry._native.native_decide.ax_1 \
  admitted_probe_canonical_value_matches_frozen_bits._native.native_decide.ax_1_1 \
  admitted_probe_winner_candidate_bits._native.native_decide.ax_1_1 \
  admitted_probe_winner_gauge_word._native.native_decide.ax_1_1 \
  all_code_block_views_eq_frozen_scan_action_views._native.native_decide.ax_1_1 \
  all_code_blocks_are_exactly_fin_range._native.native_decide.ax_1_1 \
  single_probe_closed_without_general_execution_promotion._native.native_decide.ax_1_1
do
  grep -Fqx "${native_name}" <<<"${native_axioms}" || fail "missing native axiom: ${native_name}"
done
for i in $(seq -w 0 63); do
  native_name="code_block_${i}_dominates._native.native_decide.ax_1_1"
  grep -Fqx "${native_name}" <<<"${native_axioms}" || fail "missing block native axiom: ${native_name}"
done

printf 'PIREUS_EXECUTED_STREAMING_PROBE_RESULT=V13_FROZEN_SOUNIO_PROBE0_MATCHES_LEAN_MODEL_NATIVE_TRUST_GENERAL_LINK_OPEN_NO_CLAIM stage=PARITY_OPEN semantic_authority=Sounio formal_language=Lean4 matcher_free_source_hash_bound=true frozen_matcher_source_hash_bound=true first_transcript_hash_bound=true transcript_probe=0 transcript_matrix=58475 transcript_swap=0 transcript_gauge=933 transcript_raw_bits=256 transcript_canonical_bits=256 lean_code_blocks=64 lean_code_block_size=1024 lean_gl4_entries=20160 lean_swap_views=40320 single_frozen_probe_execution_model_match=true general_executed_sounio_streaming_equality=false native_decide_trust_assumed=true native_decide_unique_axioms=71 axiom_reports=83 axiom_bearing=83 axiom_free=0 non_native_axiom_allowlist=Classical.choice,Quot.sound,propext unexpected_axioms=0 sorryax=0 python_dispatch=REFUSED_PREEXEC_E110 guardian_current_gate_false_exit=1 formal_parity_complete=false effect_parity_complete=false material_parity_complete=false spark_route_policy=KUBERNETES_ONLY spark_dispatches=0 dgx_dispatches=0 slurm_dispatches=0 u250_declared=2 u250_installed=1 u250_pending_installation=1 u250_enumeration_failures=0 llm_role=REVIEW_ONLY llm_confirmed_result=false novelty_confirmed=false claim_ready=false\n'
