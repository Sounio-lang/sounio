#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"

GARDEN_REL='tools/pireus/GARDEN_PIREUS_OPERATOR_NOVELTY_FEEDBACK_V7.md'
MODULE_REL='stdlib/hardware/pireus/operator_novelty_feedback.sio'
EXAMPLE_REL='examples/pireus_operator_novelty_feedback.sio'
TEST_REL='tests/stdlib/hardware/test_pireus_operator_novelty_feedback.sio'
CONTRACT_REL='tools/pireus/PIREUS_OPERATOR_NOVELTY_FEEDBACK_CONTRACT_V7.md'
FIRST_RECEIPT_REL='tools/pireus/operator_novelty_feedback.first.v7'
FIRST_DECISIONS_REL='tools/pireus/operator_novelty_feedback.guardian-decisions.v7'
FREEZE_REL='tools/pireus/operator_novelty_feedback.freeze.v7'
FREEZE_DECISIONS_REL='tools/pireus/operator_novelty_feedback.freeze-decisions.v7'
PARITY_REL='tools/pireus/operator_novelty_feedback.parity-open.v7'
FIRST_EVIDENCE_REL='tools/pireus/evidence/operator_novelty_feedback_v7.txt'
FIRST_TEST_REL='tools/pireus/evidence/operator_novelty_feedback_v7.test.txt'
FROZEN_EVIDENCE_REL='tools/pireus/evidence/operator_novelty_feedback_v7.frozen.txt'
FROZEN_TEST_REL='tools/pireus/evidence/operator_novelty_feedback_v7.test.frozen.txt'

GARDEN_COMMIT='0d725b95744669ec32d6f072c279efb3db366573'
EXECUTABLE_COMMIT='f4942de08530b76a1fe7427d4d60d47a69735d60'
FIRST_EVIDENCE_COMMIT='d2391bcc4d56cd4cc6c4e29dbab6520e0c0fd8f4'
MATCHER_COMMIT='396b01deb585971c5aaf1df629f8ead1a6bca6ab'
FREEZE_COMMIT='b31e2f7c28e3e320ad8e0bdef8e847b283f5220e'
FREEZE_GATE_COMMIT='c6430e3e849f3fd94dc5a024506fa53b58e3b09a'
PARITY_RECEIPT_COMMIT='eb51babec189fbdc94ec6c20cdf2b8144ca9a03f'

GARDEN_SHA256='7178bd7232b74fb7aa1662733a03a6c7e5f6fe18123a2233555f38a316e12cd9'
FIRST_SOURCE_SHA256='5ef81b3390e5acbee363edd77feb3a2f7c0daff99abc50e048f0f85c6d5491ce'
MODULE_SHA256='b73cc3fb6a905193a68a65eb6afd5d27da80395a0c38ae3772f9df56e8c8deaf'
EXAMPLE_SHA256='368a9b8e63ffb6d7e78f318ab14d3a2728cff64add97b90fce2357b101ccc843'
TEST_SHA256='d46fa88c2095fbbc5b36f8a7e960ef298933ffc499d3bfe0189e53379fd9e74c'
CONTRACT_SHA256='a8d0abd5cb0e09f657ec28fa7b78d63821f53055f90117f4b8dc3e7a98a80cdc'
FIRST_RECEIPT_SHA256='815956d81b4f7938c86556088ed50c6c2507df3d989abe12c5560d611f989d93'
FIRST_DECISIONS_SHA256='15069c09e114c26eed260a37455c7a6f4f6ff59f6dc05a2e177beb75b4d908ca'
FREEZE_SHA256='7293594eb7a881d1f89d9593b1cc19e3e611f99a491a4cd1146afe0a68cd623a'
FREEZE_DECISIONS_SHA256='39dd821ecb3e74d7748e711c687c34e4fb5eec62b761d539d57f21e4df941ade'
PARITY_SHA256='1ae8fe022071d12193624477f531595a789ffd05f97489e4ccd05d93cf78f7ef'
FIRST_EVIDENCE_SHA256='f3f26b92b7d9f70b1544af5f04ee6171173ab16d507e311ae54c104ec92e4720'
FIRST_TEST_SHA256='56662adc785712c0d32105ea3f5c0c8798a26a620fd32e1bb2b46a6f81e098bb'
FROZEN_EVIDENCE_SHA256='da2adf49188c1dcc1ca4c2a072f72f419705a8d2f12f34633ddd8a5e604998be'
FROZEN_TEST_SHA256='56662adc785712c0d32105ea3f5c0c8798a26a620fd32e1bb2b46a6f81e098bb'
SOURCE_MANIFEST_SHA256='081a036036c314d4fbddf87679f081d6d01ac766f80d2e78d9029c876880d6bc'
SEMANTICS_SHA256='a1be292392727cf515baf6d95a376d6060d56f9b807fc58d8998fbe23bdc7726'

QUOTIENT_PARENT_MODULE_SHA256='178663aa64bc44938afbe88874268d8078ee1d56e312add965d1470bb3b42ae0'
QUOTIENT_PARENT_SEMANTICS_SHA256='bd69d1d890506ebf90ff14fe5e3f2b653d5651b968d2fca7fe8ef9298a0c26c1'
QUOTIENT_PARENT_FREEZE_SHA256='973d620f30337378b760aa185ddbe9897bdd82ce18ee9e212756f519d1ed7181'
CHALLENGE_SOURCE_SHA256='7087649a5cfdb41a884aa9a2e1b0b64bbe2d25da3ca3cd1d54d5b70429854edb'
CHALLENGE_SEMANTICS_SHA256='9e92f256c25a774979a1b02cb10d5d39a1ae702ccec3a273044f91930f03a970'
CHALLENGE_RECEIPT_SHA256='daef832ee6370b656e93ae84c76ba6d17c98aaf5ad1dd86674dee27ba0f84346'

WRAPPER_SHA256='ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008'
RESOLVER_SHA256='a7e37545490745a58731933b0e07db69843cfeea30e739ed554b82d099ec3d84'
TOOLCHAIN_SHA256='6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2'
HARDWARE_SHA256='6c0cad13fd376aea694c4a7a73e603194713a938d6198c8ebddf16f3a1a75689'
GUARDIAN_SHA256='208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60'
GUARDIAN_POLICY_SHA256='64bb0118793fe46dcb392abc1a9212eb15bd55047461576a3ef1a6cefa3f17da'
GUARDIAN_SELFTEST_SHA256='c9c7f839fb262dbf616716e3c5f0601bb03cfbcbbcfe2fbe09bd0b39894e2a9f'
COMMAND_SHA256='42d64a15462379b0d1ef393fe16569e09f98afab6b6fe93d8f465829ccaa085d'
TEST_COMMAND_SHA256='8af85681f74183c82750b94f586203b3f8ffdf61e57f29bee50a420e814ecfb4'
CI_COMMAND_SHA256='bb672053251355f05b15c66ae51c8e6048c31f5bbfef93f025e0600637d5fafc'
PYTHON_TOOLCHAIN_SHA256='497ce0938df96d9bf3c159472d251a946c2d6bd832220937d1b885f7759b05ba'
PYTHON_COMMAND_SHA256='72566473f0019fb50e65175a0ac019af5ad2f495ccfc30b664b7735791e968fc'
RUST_TOOLCHAIN_SHA256='b58640570ba9ffcdb2c2d241e4ce8ece9c7d75c6b1e59e308dee3e5f0e10b56d'
RUST_COMMAND_SHA256='0421c7020d972449b7a1f0492945c18a10be46f6a16ade7f8efad157e2e01b00'

MAIN_PREEXEC_FRAME_SHA256='4cd030e1121799b2af8b9cda1e67c579e0ef5ab0264a29267b7ff00091df3471'
TEST_PREEXEC_FRAME_SHA256='5d3322c3ff3a110b5a23a92cdd295ba332e454abf4874bc9ce1bc673819afd0c'
MAIN_FREEZE_FRAME_SHA256='3f530d606af3c72fb0125e40f1d8527ee96130880399c4b4d2ff30795c32f9f6'
TEST_FREEZE_FRAME_SHA256='bd12a26b8c0aebd1aa58fe14645b46a63425d829b6e5bf17e7860a7ca07f15c4'
POLICY_MISSING_FRAME_SHA256='d179c54e57b641e31e93d0374dfeff8bc248d700c6899ae36c1cf57fc319bdc6'
POLICY_TIMEOUT_FRAME_SHA256='d20098c89541ec075a0d8d6821dc90699a13337a9df59b3883cc83c3bed1e9bf'
POLICY_ERROR_FRAME_SHA256='22b87e76f809d795618ae00a3c4a25eac651bc14130efe84b9f800cebb72407e'
PYTHON_FRAME_SHA256='f452882697d1a45bd14ab9ef3a60aab36bcfefa215686bc4b47768b2f2d61d27'
RUST_FRAME_SHA256='1f5e4fa4b436860aead50f1d4ff113f0fe27fd80d3ef3fb11e437fba836b27b9'
PARITY_PREFREEZE_FRAME_SHA256='ac6b7605d79dd02c7667d7d1746a8d6cec53df110ff5fd359b7440b1965b575f'
LLM_PROMOTION_FRAME_SHA256='e90798ec6a2f95b5149b6d38e6e9755f44109753d8cf4e44dfaa58d53d9ae013'
CPP_AUTHORITY_FRAME_SHA256='69249023325c211a3e720f8ebbd3b4c02935b3a71680ee6a9cced7bb11e63fa8'
CLAIM_PROMOTION_FRAME_SHA256='c965ce940368a66775d9d152431e7101fe5ad42a0f2528e633fb060fe92d96cc'
PARENT_LAUNDERING_FRAME_SHA256='67f5cf4e2c952b8f4fb0da0856d2ca66fe51cf470bae508c0919b8f3f53484cf'
PARITY_OPEN_FRAME_SHA256='db65372c10068a9e632e293014a365e19731c1d2a8f04549ec867a9376fb7675'
CI_FRAME_SHA256='0d8a23d851c583249e6e0b1a7914affea3b0ee9a6cb5df8527a6a5804794fb11'

FIRST_LINES=6127
FIRST_BYTES=86843
FROZEN_LINES=6130
FROZEN_BYTES=86887
TEST_LINES=2
TEST_BYTES=57
ZERO='0 0 0 0 0 0 0 0'
GUARDIAN_FRAME_SCHEMA=9020
GUARDIAN_FRAME_WORDS=82

fail() {
  printf 'pireus operator novelty feedback: FAIL: %s\n' "$*" >&2
  exit 1
}

sha_file() { sha256sum "$1" | cut -d' ' -f1; }
sha_text() { printf '%s\n' "$1" | sha256sum | cut -d' ' -f1; }

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

require_ancestor() {
  local ancestor="$1" descendant="$2"
  git -C "${ROOT}" merge-base --is-ancestor "${ancestor}" "${descendant}" ||
    fail "commit order drift: ${ancestor} !<= ${descendant}"
}

sha_limbs() {
  local hex="$1" out='' i part
  [[ "${hex}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid SHA-256: ${hex}"
  for ((i = 0; i < 8; i++)); do
    part="${hex:$((i * 8)):8}"
    out="${out}${out:+ }$((16#${part}))"
  done
  printf '%s' "${out}"
}

sounio_preexec_frame() {
  local command_hash="$1"
  printf '9020 1 2 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" "${ZERO}" \
    "$(sha_limbs "${QUOTIENT_PARENT_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${command_hash}")" "${ZERO}" "${ZERO}"
}

freeze_frame() {
  local policy="$1" command_hash="$2" result_hash="$3"
  printf '9020 2 3 1 1 %s 1 1 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${policy}" "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${QUOTIENT_PARENT_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${command_hash}")" \
    "$(sha_limbs "${result_hash}")" "${ZERO}"
}

forbidden_frame() {
  local language="$1" toolchain_hash="$2" command_hash="$3"
  printf '9020 3 4 %s 7 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${language}" "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${QUOTIENT_PARENT_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${toolchain_hash}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${command_hash}")" "${ZERO}" "${ZERO}"
}

claim_promotion_frame() {
  printf '9020 4 7 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${COMMAND_SHA256}")" \
    "$(sha_limbs "${FROZEN_EVIDENCE_SHA256}")" "${ZERO}"
}

parent_laundering_frame() {
  local wrong_parent='0d69d1d890506ebf90ff14fe5e3f2b653d5651b968d2fca7fe8ef9298a0c26c1'
  printf '9020 3 4 4 4 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${wrong_parent}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${COMMAND_SHA256}")" "${ZERO}" "${ZERO}"
}

parity_prefreeze_frame() {
  printf '9020 2 4 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${COMMAND_SHA256}")" "${ZERO}" "${ZERO}"
}

parity_open_frame() {
  # The parity child is bound to the frozen v7 semantics, hence self as parent.
  printf '9020 3 4 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${COMMAND_SHA256}")" "${ZERO}" "${ZERO}"
}

llm_promotion_frame() {
  printf '9020 3 5 6 6 1 0 0 0 1 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${COMMAND_SHA256}")" "${ZERO}" "${ZERO}"
}

cpp_authority_frame() {
  printf '9020 3 4 4 4 1 1 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${COMMAND_SHA256}")" \
    "$(sha_limbs "${FROZEN_EVIDENCE_SHA256}")" "${ZERO}"
}

ci_frame() {
  printf '9020 4 11 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${QUOTIENT_PARENT_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${CI_COMMAND_SHA256}")" \
    "$(sha_limbs "${FROZEN_EVIDENCE_SHA256}")" "${ZERO}"
}

authorize() {
  local label="$1" frame="$2" frame_hash="$3" expected_rc="$4" expected="$5"
  local decision rc words
  [[ "${frame%% *}" == "${GUARDIAN_FRAME_SCHEMA}" ]] ||
    fail "${label}: frame schema drift"
  words="$(wc -w <<<"${frame}" | tr -d ' ')"
  [[ "${words}" == "${GUARDIAN_FRAME_WORDS}" ]] ||
    fail "${label}: frame word count ${words} != ${GUARDIAN_FRAME_WORDS}"
  # sha_text hashes exactly the newline-terminated bytes sent to Guardian stdin.
  [[ "$(sha_text "${frame}")" == "${frame_hash}" ]] ||
    fail "${label}: frame hash drift"
  set +e
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  rc=$?
  set -e
  [[ "${rc}" == "${expected_rc}" ]] ||
    fail "${label}: Guardian rc ${rc} != ${expected_rc}"
  [[ "${decision}" == "${expected}" ]] ||
    fail "${label}: Guardian decision drift: ${decision}"
  printf 'GUARDIAN_DECISION label=%s frame_sha256=%s rc=%s %s\n' \
    "${label}" "${frame_hash}" "${rc}" "${decision}"
}

semantic_transcript_admitted() {
  local path="$1" frozen="$2" forge_block expected_forge
  grep -Fqx -- 'PIREUS_ONF_RESULT valid=1' "${path}" || return 1
  grep -Fqx -- ' error=0' "${path}" || return 1
  grep -Fqx -- ' failures=0' "${path}" || return 1
  grep -Fqx -- ' claim_ready=0' "${path}" || return 1
  grep -Fqx -- 'PIREUS_ONF_OUTCOME kind=2' "${path}" || return 1
  grep -Fqx -- ' bridge=0' "${path}" || return 1
  grep -Fqx -- ' seed=1' "${path}" || return 1
  grep -Fqx -- ' exhaustive_separation=1' "${path}" || return 1
  grep -Fqx -- ' best_class=8' "${path}" || return 1
  grep -Fqx -- ' best_action_code=68674' "${path}" || return 1
  grep -Fqx -- ' best_residual_nonzero=96' "${path}" || return 1
  grep -Fqx -- ' replay_checks=256' "${path}" || return 1
  grep -Fqx -- ' replay_failures=0' "${path}" || return 1
  grep -Fqx -- ' zero_hits=0' "${path}" || return 1
  grep -Fqx -- ' broad_novelty=0' "${path}" || return 1
  grep -Fqx -- ' historical_novelty=0' "${path}" || return 1
  grep -Fqx -- ' priority=0' "${path}" || return 1
  expected_forge=$'PIREUS_ONF_DIGEST name=forge value=2024183784\n:2544298388\n:1593048018\n:287379041\n:219866591\n:4108390960\n:1594301583\n:1652437146'
  forge_block="$(sed -n '/^PIREUS_ONF_DIGEST name=forge value=/,+7p' "${path}")"
  [[ "${forge_block}" == "${expected_forge}" ]] || return 1
  if [[ "${frozen}" == true ]]; then
    grep -Fqx -- 'PIREUS_ONF_FROZEN match=1' "${path}" || return 1
    grep -Fqx -- ' mismatch_code=0' "${path}" || return 1
  fi
}

transcript_admitted() {
  local path="$1" expected_hash="$2" expected_lines="$3" expected_bytes="$4" frozen="$5"
  [[ "$(sha_file "${path}")" == "${expected_hash}" ]] || return 1
  [[ "$(wc -l <"${path}" | tr -d ' ')" == "${expected_lines}" ]] || return 1
  [[ "$(wc -c <"${path}" | tr -d ' ')" == "${expected_bytes}" ]] || return 1
  semantic_transcript_admitted "${path}" "${frozen}"
}

cd "${ROOT}"
[[ -f "${ROOT}/AGENTS.md" && -f "${ROOT}/FOUNDER_INTENT.md" ]] ||
  fail 'repository root markers missing'

for pair in \
  "${GARDEN_REL}:${GARDEN_SHA256}" \
  "${MODULE_REL}:${MODULE_SHA256}" \
  "${EXAMPLE_REL}:${EXAMPLE_SHA256}" \
  "${TEST_REL}:${TEST_SHA256}" \
  "${CONTRACT_REL}:${CONTRACT_SHA256}" \
  "${FIRST_RECEIPT_REL}:${FIRST_RECEIPT_SHA256}" \
  "${FIRST_DECISIONS_REL}:${FIRST_DECISIONS_SHA256}" \
  "${FREEZE_REL}:${FREEZE_SHA256}" \
  "${FREEZE_DECISIONS_REL}:${FREEZE_DECISIONS_SHA256}" \
  "${PARITY_REL}:${PARITY_SHA256}" \
  "${FIRST_EVIDENCE_REL}:${FIRST_EVIDENCE_SHA256}" \
  "${FIRST_TEST_REL}:${FIRST_TEST_SHA256}" \
  "${FROZEN_EVIDENCE_REL}:${FROZEN_EVIDENCE_SHA256}" \
  "${FROZEN_TEST_REL}:${FROZEN_TEST_SHA256}" \
  "stdlib/hardware/pireus/operator_lowering_forge.sio:${QUOTIENT_PARENT_MODULE_SHA256}" \
  "tools/pireus/operator_lowering_forge.freeze.v6:${QUOTIENT_PARENT_FREEZE_SHA256}" \
  "stdlib/hardware/pireus/xor_lowering_legality.sio:${CHALLENGE_SOURCE_SHA256}" \
  "docs/research/pireus_xor_lowering_legality_semantics.md:${CHALLENGE_SEMANTICS_SHA256}" \
  "docs/research/receipts/pireus_xor_lowering_legality_20260827.md:${CHALLENGE_RECEIPT_SHA256}" \
  "bin/souc:${WRAPPER_SHA256}" \
  "scripts/lib/resolve_souc.sh:${RESOLVER_SHA256}" \
  "bin/souc-lean-single-x86_64:${TOOLCHAIN_SHA256}" \
  "stdlib/coordination/loom_language_authority.sio:${GUARDIAN_POLICY_SHA256}" \
  "scripts/ci/sounio_loom_language_authority_selftest.sh:${GUARDIAN_SELFTEST_SHA256}"
do
  require_hash "${ROOT}/${pair%%:*}" "${pair##*:}"
done
require_hash "${GUARDIAN}" "${GUARDIAN_SHA256}"

require_ancestor "${GARDEN_COMMIT}" "${EXECUTABLE_COMMIT}"
require_ancestor "${EXECUTABLE_COMMIT}" "${FIRST_EVIDENCE_COMMIT}"
require_ancestor "${FIRST_EVIDENCE_COMMIT}" "${MATCHER_COMMIT}"
require_ancestor "${MATCHER_COMMIT}" "${FREEZE_COMMIT}"
require_ancestor "${FREEZE_COMMIT}" "${FREEZE_GATE_COMMIT}"
require_ancestor "${FREEZE_GATE_COMMIT}" "${PARITY_RECEIPT_COMMIT}"
require_ancestor "${PARITY_RECEIPT_COMMIT}" HEAD

first_source_hash="$(git cat-file blob "${EXECUTABLE_COMMIT}:${MODULE_REL}" | sha256sum | cut -d' ' -f1)"
[[ "${first_source_hash}" == "${FIRST_SOURCE_SHA256}" ]] ||
  fail 'matcher-free historical source hash drift'
if git cat-file blob "${EXECUTABLE_COMMIT}:${MODULE_REL}" |
    grep -Fq 'pireus_operator_novelty_feedback_frozen_mismatch_code'; then
  fail 'historical executable contains post-result matcher'
fi
grep -Fq 'pireus_operator_novelty_feedback_frozen_mismatch_code' \
  "${ROOT}/${MODULE_REL}" || fail 'current module lacks frozen matcher'
for commit in "${MATCHER_COMMIT}" "${FREEZE_COMMIT}"; do
  committed_module_hash="$(git cat-file blob "${commit}:${MODULE_REL}" |
    sha256sum | cut -d' ' -f1)"
  [[ "${committed_module_hash}" == "${MODULE_SHA256}" ]] ||
    fail "matcher-bearing module hash drift at ${commit}"
  git cat-file blob "${commit}:${MODULE_REL}" |
    grep -Fq 'pireus_operator_novelty_feedback_frozen_mismatch_code' ||
    fail "matcher absent at ${commit}"
done
[[ "$(git cat-file blob "${PARITY_RECEIPT_COMMIT}:${PARITY_REL}" |
    sha256sum | cut -d' ' -f1)" == "${PARITY_SHA256}" ]] ||
  fail 'parity-open receipt object drift'

manifest_hash="$(sed -n '/^source_manifest_begin$/,/^source_manifest_end$/p' \
  "${ROOT}/${FREEZE_REL}" | sed '1d;$d' | sha256sum | cut -d' ' -f1)"
[[ "${manifest_hash}" == "${SOURCE_MANIFEST_SHA256}" ]] ||
  fail 'source manifest extraction drift'
semantics_hash="$(sed -n '/^semantics_material_begin$/,/^semantics_material_end$/p' \
  "${ROOT}/${FREEZE_REL}" | sed '1d;$d' | sha256sum | cut -d' ' -f1)"
[[ "${semantics_hash}" == "${SEMANTICS_SHA256}" ]] ||
  fail 'semantics extraction drift'

require_line "${ROOT}/${FIRST_RECEIPT_REL}" 'status=SOUNIO_EXECUTABLE'
require_line "${ROOT}/${FIRST_RECEIPT_REL}" 'first_executable_contains_result_matcher=false'
require_line "${ROOT}/${FIRST_RECEIPT_REL}" \
  "sounio_source_sha256=${FIRST_SOURCE_SHA256}"
require_line "${ROOT}/${FIRST_RECEIPT_REL}" \
  "toolchain_sha256=${TOOLCHAIN_SHA256}"
require_line "${ROOT}/${FIRST_RECEIPT_REL}" \
  "command_sha256=${COMMAND_SHA256}"
require_line "${ROOT}/${FIRST_RECEIPT_REL}" \
  "result_sha256=${FIRST_EVIDENCE_SHA256}"
require_line "${ROOT}/${FIRST_RECEIPT_REL}" 'outcome_kind=2'
require_line "${ROOT}/${FIRST_RECEIPT_REL}" 'operator_seed=true'
require_line "${ROOT}/${FIRST_RECEIPT_REL}" 'exhaustive_nonmembership_checks=168'
require_line "${ROOT}/${FIRST_RECEIPT_REL}" 'best_residual_nonzero=96'
require_line "${ROOT}/${FIRST_RECEIPT_REL}" 'broad_novelty=false'
require_line "${ROOT}/${FIRST_RECEIPT_REL}" 'historical_novelty=false'
require_line "${ROOT}/${FIRST_RECEIPT_REL}" 'claim_ready=false'
require_line "${ROOT}/${FREEZE_REL}" 'status=SEMANTICS_FROZEN'
require_line "${ROOT}/${FREEZE_REL}" "semantics_sha256=${SEMANTICS_SHA256}"
require_line "${ROOT}/${FREEZE_REL}" \
  "quotient_parent_semantics_sha256=${QUOTIENT_PARENT_SEMANTICS_SHA256}"
require_line "${ROOT}/${FREEZE_REL}" \
  "challenge_parent_semantics_sha256=${CHALLENGE_SEMANTICS_SHA256}"
require_line "${ROOT}/${FREEZE_REL}" 'first_result_is_exact_prefix=true'
require_line "${ROOT}/${FREEZE_REL}" 'operator_seed_generated=true'
require_line "${ROOT}/${FREEZE_REL}" 'admitted_actions=12'
require_line "${ROOT}/${FREEZE_REL}" 'operator_classes=14'
require_line "${ROOT}/${FREEZE_REL}" 'class_action_pairs=168'
require_line "${ROOT}/${FREEZE_REL}" 'exhaustive_nonmembership_checks=168'
require_line "${ROOT}/${FREEZE_REL}" 'zero_residual_hits=0'
require_line "${ROOT}/${FREEZE_REL}" 'scope_novelty=RELATIVE_FINITE_QUOTIENT_SEPARATION_ONLY'
require_line "${ROOT}/${FREEZE_REL}" 'formal_parity_open=false'
require_line "${ROOT}/${FREEZE_REL}" 'effect_parity_open=false'
require_line "${ROOT}/${FREEZE_REL}" 'material_parity_open=false'
require_line "${ROOT}/${FREEZE_REL}" 'claim_ready=false'
require_line "${ROOT}/${PARITY_REL}" 'status=OPEN_NOT_EXECUTED'
require_line "${ROOT}/${PARITY_REL}" 'stage=PARITY_OPEN'
require_line "${ROOT}/${PARITY_REL}" \
  "frozen_semantics_sha256=${SEMANTICS_SHA256}"
require_line "${ROOT}/${PARITY_REL}" \
  "opening_frame_sha256=${PARITY_OPEN_FRAME_SHA256}"
require_line "${ROOT}/${PARITY_REL}" \
  'opening_decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
require_line "${ROOT}/${PARITY_REL}" 'lean_status=OPEN_NOT_EXECUTED'
require_line "${ROOT}/${PARITY_REL}" 'koka_status=OPEN_NOT_EXECUTED'
require_line "${ROOT}/${PARITY_REL}" 'cpp_status=OPEN_NOT_EXECUTED'
require_line "${ROOT}/${PARITY_REL}" 'parity_processes_launched=0'
require_line "${ROOT}/${PARITY_REL}" 'scope_novelty_not_absolute=true'
require_line "${ROOT}/${PARITY_REL}" 'claim_ready=false'

transcript_admitted "${ROOT}/${FIRST_EVIDENCE_REL}" \
  "${FIRST_EVIDENCE_SHA256}" "${FIRST_LINES}" "${FIRST_BYTES}" false ||
  fail 'first transcript admission failed'
transcript_admitted "${ROOT}/${FROZEN_EVIDENCE_REL}" \
  "${FROZEN_EVIDENCE_SHA256}" "${FROZEN_LINES}" "${FROZEN_BYTES}" true ||
  fail 'frozen transcript admission failed'
cmp -n "${FIRST_BYTES}" "${ROOT}/${FIRST_EVIDENCE_REL}" \
  "${ROOT}/${FROZEN_EVIDENCE_REL}" || fail 'first transcript is not frozen prefix'

tmp_dir="$(mktemp -d /tmp/pireus-onf-gate.XXXXXX)"
trap 'rm -rf "${tmp_dir}"' EXIT
printf 'PIREUS_ONF_FROZEN match=1\n mismatch_code=0\n\n' \
  >"${tmp_dir}/expected-suffix.txt"
tail -c "+$((FIRST_BYTES + 1))" "${ROOT}/${FROZEN_EVIDENCE_REL}" \
  >"${tmp_dir}/actual-suffix.txt"
cmp "${tmp_dir}/expected-suffix.txt" "${tmp_dir}/actual-suffix.txt" ||
  fail 'frozen causal suffix drift'

sed '0,/^ best_residual_nonzero=96$/s// best_residual_nonzero=95/' \
  "${ROOT}/${FROZEN_EVIDENCE_REL}" >"${tmp_dir}/tampered-seed.txt"
if semantic_transcript_admitted "${tmp_dir}/tampered-seed.txt" true; then
  fail 'tampered seed passed semantic projection'
fi
sed '0,/^:1652437146$/s//:1652437147/' \
  "${ROOT}/${FROZEN_EVIDENCE_REL}" >"${tmp_dir}/tampered-digest.txt"
if semantic_transcript_admitted "${tmp_dir}/tampered-digest.txt" true; then
  fail 'tampered forge digest passed semantic projection'
fi
sed 's/^PIREUS_ONF_OUTCOME kind=2$/PIREUS_ONF_OUTCOME kind=1/' \
  "${ROOT}/${FROZEN_EVIDENCE_REL}" >"${tmp_dir}/tampered-kind.txt"
if semantic_transcript_admitted "${tmp_dir}/tampered-kind.txt" true; then
  fail 'tampered outcome kind passed semantic projection'
fi
sed 's/^ best_class=8$/ best_class=7/' \
  "${ROOT}/${FROZEN_EVIDENCE_REL}" >"${tmp_dir}/tampered-class.txt"
if semantic_transcript_admitted "${tmp_dir}/tampered-class.txt" true; then
  fail 'tampered best class passed semantic projection'
fi
sed 's/^ best_action_code=68674$/ best_action_code=68675/' \
  "${ROOT}/${FROZEN_EVIDENCE_REL}" >"${tmp_dir}/tampered-action.txt"
if semantic_transcript_admitted "${tmp_dir}/tampered-action.txt" true; then
  fail 'tampered action passed semantic projection'
fi
sed 's/^ replay_checks=256$/ replay_checks=255/g' \
  "${ROOT}/${FROZEN_EVIDENCE_REL}" >"${tmp_dir}/tampered-replay.txt"
if semantic_transcript_admitted "${tmp_dir}/tampered-replay.txt" true; then
  fail 'tampered replay count passed semantic projection'
fi
sed 's/^ broad_novelty=0$/ broad_novelty=1/' \
  "${ROOT}/${FROZEN_EVIDENCE_REL}" >"${tmp_dir}/tampered-broad.txt"
if semantic_transcript_admitted "${tmp_dir}/tampered-broad.txt" true; then
  fail 'tampered broad novelty flag passed semantic projection'
fi
sed 's/^ claim_ready=0$/ claim_ready=1/g' \
  "${ROOT}/${FROZEN_EVIDENCE_REL}" >"${tmp_dir}/tampered-claim.txt"
if semantic_transcript_admitted "${tmp_dir}/tampered-claim.txt" true; then
  fail 'tampered claim flag passed semantic projection'
fi
sed -e '/^semantics_material_begin$/,/^semantics_material_end$/s/^outcome_kind=2$/outcome_kind=1/' \
  "${ROOT}/${FREEZE_REL}" \
  >"${tmp_dir}/tampered-freeze.txt"
tampered_semantics_hash="$(sed -n \
  '/^semantics_material_begin$/,/^semantics_material_end$/p' \
  "${tmp_dir}/tampered-freeze.txt" | sed '1d;$d' | sha256sum | cut -d' ' -f1)"
[[ "${tampered_semantics_hash}" != "${SEMANTICS_SHA256}" ]] ||
  fail 'semantics tamper did not change hash'
cp "${ROOT}/${MODULE_REL}" "${tmp_dir}/tampered-source.sio"
printf '\n' >>"${tmp_dir}/tampered-source.sio"
[[ "$(sha_file "${tmp_dir}/tampered-source.sio")" != "${MODULE_SHA256}" ]] ||
  fail 'source tamper did not change hash'

selftest_output="$("${ROOT}/scripts/ci/sounio_loom_language_authority_selftest.sh")"
printf '%s\n' "${selftest_output}" | grep -Fqx -- \
  'sounio-loom-language-authority-selftest: PASS language=Sounio cases=33 python=refused rust=refused policy_missing=refused llm_promotion=refused parent_laundering=refused ocaml_realization=admitted ocaml_prefreeze=refused ocaml_parent_laundering=refused ocaml_guardian=admitted ocaml_parity=refused cpp_bootstrap=admitted malformed=refused sabotage_python_rule=admits' ||
  fail 'Guardian selftest terminal drift'
printf '%s\n' \
  'GUARDIAN_SELFTEST_NOTE sabotage_python_rule=admits means the deliberately weakened fixture was detected as admitting Python; production policy remains hash-bound and denies it'

main_command_record='SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pireus_operator_novelty_feedback.sio'
test_command_record='SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/stdlib/hardware/test_pireus_operator_novelty_feedback.sio'
ci_command_record='bash scripts/ci/pireus_operator_novelty_feedback.sh'
[[ "$(sha_text "${main_command_record}")" == "${COMMAND_SHA256}" ]] ||
  fail 'main command record drift'
[[ "$(sha_text "${test_command_record}")" == "${TEST_COMMAND_SHA256}" ]] ||
  fail 'test command record drift'
[[ "$(sha_text "${ci_command_record}")" == "${CI_COMMAND_SHA256}" ]] ||
  fail 'CI command record drift'

authorize CI_PREEXEC "$(ci_frame)" \
  "${CI_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'

authorize FROZEN_REPLAY_PREEXEC \
  "$(sounio_preexec_frame "${COMMAND_SHA256}")" \
  "${MAIN_PREEXEC_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE'
if ! timeout 600 env SOUNIO_SOUC_ENGINE=lean_single \
    ./bin/souc run examples/pireus_operator_novelty_feedback.sio \
    >"${tmp_dir}/main.txt"; then
  fail 'frozen Sounio replay failed or timed out'
fi
transcript_admitted "${tmp_dir}/main.txt" "${FROZEN_EVIDENCE_SHA256}" \
  "${FROZEN_LINES}" "${FROZEN_BYTES}" true ||
  fail 'live frozen replay drift'

authorize FROZEN_TEST_PREEXEC \
  "$(sounio_preexec_frame "${TEST_COMMAND_SHA256}")" \
  "${TEST_PREEXEC_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE'
if ! timeout 600 env SOUNIO_SOUC_ENGINE=lean_single \
    ./bin/souc run tests/stdlib/hardware/test_pireus_operator_novelty_feedback.sio \
    >"${tmp_dir}/test.txt"; then
  fail 'frozen Sounio structural test failed or timed out'
fi
[[ "$(sha_file "${tmp_dir}/test.txt")" == "${FROZEN_TEST_SHA256}" ]] ||
  fail 'live frozen test hash drift'
[[ "$(wc -l <"${tmp_dir}/test.txt" | tr -d ' ')" == "${TEST_LINES}" ]] ||
  fail 'live frozen test line count drift'
[[ "$(wc -c <"${tmp_dir}/test.txt" | tr -d ' ')" == "${TEST_BYTES}" ]] ||
  fail 'live frozen test byte count drift'
require_line "${tmp_dir}/test.txt" \
  'pireus operator novelty feedback structural failures: 0'

authorize FREEZE_SEAL \
  "$(freeze_frame 1 "${COMMAND_SHA256}" "${FROZEN_EVIDENCE_SHA256}")" \
  "${MAIN_FREEZE_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'
authorize TEST_FREEZE_SEAL \
  "$(freeze_frame 1 "${TEST_COMMAND_SHA256}" "${FROZEN_TEST_SHA256}")" \
  "${TEST_FREEZE_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'
authorize POLICY_MISSING \
  "$(freeze_frame 0 "${COMMAND_SHA256}" "${FROZEN_EVIDENCE_SHA256}")" \
  "${POLICY_MISSING_FRAME_SHA256}" 101 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=101 reason=policy-missing next_stage=SOUNIO_EXECUTABLE'
authorize POLICY_TIMEOUT \
  "$(freeze_frame 2 "${COMMAND_SHA256}" "${FROZEN_EVIDENCE_SHA256}")" \
  "${POLICY_TIMEOUT_FRAME_SHA256}" 102 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=102 reason=policy-timeout next_stage=SOUNIO_EXECUTABLE'
authorize POLICY_ERROR \
  "$(freeze_frame 3 "${COMMAND_SHA256}" "${FROZEN_EVIDENCE_SHA256}")" \
  "${POLICY_ERROR_FRAME_SHA256}" 103 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=103 reason=policy-error next_stage=SOUNIO_EXECUTABLE'
authorize PYTHON_ORACLE \
  "$(forbidden_frame 7 "${PYTHON_TOOLCHAIN_SHA256}" "${PYTHON_COMMAND_SHA256}")" \
  "${PYTHON_FRAME_SHA256}" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
printf 'GUARDIAN_DISPATCH label=PYTHON_ORACLE process_launched=false\n'
authorize RUST_ORACLE \
  "$(forbidden_frame 8 "${RUST_TOOLCHAIN_SHA256}" "${RUST_COMMAND_SHA256}")" \
  "${RUST_FRAME_SHA256}" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
printf 'GUARDIAN_DISPATCH label=RUST_ORACLE process_launched=false\n'
authorize PARITY_PREFREEZE "$(parity_prefreeze_frame)" \
  "${PARITY_PREFREEZE_FRAME_SHA256}" 112 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=112 reason=wrong-stage next_stage=SOUNIO_EXECUTABLE'
authorize PARITY_OPEN "$(parity_open_frame)" \
  "${PARITY_OPEN_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
authorize LLM_PROMOTION "$(llm_promotion_frame)" \
  "${LLM_PROMOTION_FRAME_SHA256}" 119 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=119 reason=review-promoted-to-authority next_stage=SEMANTICS_FROZEN'
authorize CPP_AUTHORITY "$(cpp_authority_frame)" \
  "${CPP_AUTHORITY_FRAME_SHA256}" 113 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=SEMANTICS_FROZEN'
authorize CLAIM_PROMOTION "$(claim_promotion_frame)" \
  "${CLAIM_PROMOTION_FRAME_SHA256}" 122 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=122 reason=parity-receipt-missing next_stage=PARITY_OPEN'
printf 'GUARDIAN_DISPATCH label=CLAIM_PROMOTION process_launched=false\n'
authorize PARENT_LAUNDERING "$(parent_laundering_frame)" \
  "${PARENT_LAUNDERING_FRAME_SHA256}" 117 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=117 reason=parent-semantics-hash-mismatch next_stage=SEMANTICS_FROZEN'
printf 'GUARDIAN_DISPATCH label=PARENT_LAUNDERING process_launched=false\n'

require_hash "${ROOT}/${MODULE_REL}" "${MODULE_SHA256}"
printf '%s\n' \
  'pireus operator novelty feedback: STAGE_REACHED_NOT_A_CLAIM gate_mode=CONTENT_ADDRESSED_PARITY_OPEN_REPLAY stage=PARITY_OPEN operator_seed=true relative_scope=FINITE_QUOTIENT_ONLY formal=OPEN_NOT_EXECUTED effect=OPEN_NOT_EXECUTED material=OPEN_NOT_EXECUTED claim_promotion=DENIED claim_ready=false'
