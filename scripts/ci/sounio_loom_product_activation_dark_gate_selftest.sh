#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_PARENT="$ROOT_DIR/tools/loom/_build"
mkdir -p "$TEST_PARENT"
TEST_ROOT="$(mktemp -d "$TEST_PARENT/product-activation-dark.XXXXXX")"
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
RUNTIME="$TEST_ROOT/resident-v5"
PROJECTION="$ROOT_DIR/tools/loom/kernel_peer_activation_capsule.current.v1"
PYTHON_PATH="$(command -v python3 || true)"

cleanup() {
  if [[ "${SOUNIO_LOOM_KEEP_TEST_ROOT:-0}" != 1 ]]; then
    rm -rf "$TEST_ROOT"
  fi
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-product-activation-dark-gate-selftest: FAIL: %s test_root=%s\n' \
    "$*" "$TEST_ROOT" >&2
  exit 1
}

projection_code() {
  local label="$1" line rest
  line="$(grep -m1 "^CASE label=${label} EXPECT code=" "$PROJECTION")"
  [[ -n "$line" ]] || fail "Sounio projection omitted $label"
  rest="${line#* EXPECT code=}"
  printf '%s' "${rest%% *}"
}

output_field() {
  local output="$1" key="$2" token
  for token in $output; do
    if [[ "$token" == "$key="* ]]; then
      printf '%s' "${token#*=}"
      return 0
    fi
  done
  fail "probe field is missing: $key"
}

probe() {
  local tag="$1" deadline_ms="$2"
  shift 2
  SOUNIO_LOOM_HOOK_TEST_MODE=1 \
    SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_RUNTIME="$RUNTIME" \
    SOUNIO_LOOM_ACTIVATION_DARK_LOG="$TEST_ROOT/$tag.dark.tsv" \
    SOUNIO_LOOM_SUBPROCESS_MEMBRANE_LOG="$TEST_ROOT/$tag.membrane.tsv" \
    SOUNIO_LOOM_RESIDENT_RECEIPT_LOG="$TEST_ROOT/$tag.resident.tsv" \
    "$LOOM" subprocess-membrane-probe --root "$ROOT_DIR" \
      --cwd "$ROOT_DIR" --scope "$TEST_ROOT/scope" \
      --deadline-ms "$deadline_ms" -- "$@"
}

expect_probe() {
  local tag="$1" expected_rc="$2" marker="$3" deadline_ms="$4"
  shift 4
  local output rc
  set +e
  output="$(probe "$tag" "$deadline_ms" "$@" 2>&1)"
  rc=$?
  set -e
  [[ "$rc" -eq "$expected_rc" ]] ||
    fail "$tag rc=$rc expected=$expected_rc output=$output"
  [[ "$output" == *"$marker"* ]] ||
    fail "$tag omitted $marker: $output"
  printf '%s' "$output"
}

expect_pre_spawn_refusal() {
  local tag="$1" variable="$2" path="$3" marker="$4"
  local output rc
  set +e
  output="$(env SOUNIO_LOOM_HOOK_TEST_MODE=1 \
    SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_RUNTIME="$RUNTIME" \
    SOUNIO_LOOM_ACTIVATION_DARK_LOG="$TEST_ROOT/$tag.dark.tsv" \
    SOUNIO_LOOM_SUBPROCESS_MEMBRANE_LOG="$TEST_ROOT/$tag.membrane.tsv" \
    SOUNIO_LOOM_RESIDENT_RECEIPT_LOG="$TEST_ROOT/$tag.resident.tsv" \
    "$variable=$path" "$LOOM" subprocess-membrane-probe \
      --root "$ROOT_DIR" --cwd "$ROOT_DIR" --scope "$TEST_ROOT/scope" \
      --deadline-ms 2000 -- /usr/bin/true 2>&1)"
  rc=$?
  set -e
  [[ "$rc" -eq 1 && "$output" == *"$marker"* ]] ||
    fail "$tag was not refused before spawn: rc=$rc output=$output"
  [[ ! -e "$TEST_ROOT/$tag.resident.tsv" && \
     ! -e "$TEST_ROOT/$tag.dark.tsv" && \
     ! -e "$TEST_ROOT/$tag.membrane.tsv" ]] ||
    fail "$tag reached a resident, dark receipt, or legacy membrane"
}

[[ -n "$PYTHON_PATH" && -x "$PYTHON_PATH" ]] ||
  fail 'python3 path is required for the deliberate non-execution control'

SOUNIO_LOOM_KERNEL_PEER_ACTIVATION_CURRENT_OUTPUT="$PROJECTION" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_peer_activation_capsule_current_frame.sh" \
  >/dev/null
SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_OUTPUT="$RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_resident_membrane_v5.sh" >/dev/null
(
  flock -x 9
  dune build --root "$ROOT_DIR/tools/loom" src/loom.exe >/dev/null
) 9>"$ROOT_DIR/tools/loom/_build/.dune-build.lock"
mkdir -p "$TEST_ROOT/scope"

current_code="$(projection_code current_material)"
seal_code="$(projection_code seal)"

positive="$(expect_probe positive 0 'kind=1 exit=0' 15000 /usr/bin/true)"
[[ "$positive" == *"authority=resident-Sounio-v5"* && \
   "$positive" == *"activation_authority=Sounio"* && \
   "$positive" == *"activation_code=$current_code"* && \
   "$positive" == *"activation_capsule_state=EMPTY"* && \
   "$positive" == *"activation_mode=dark"* && \
   "$positive" == *"activation_authorizing=false"* && \
   "$positive" == *"production_activation=false"* && \
   "$positive" == *"decision_code=0"* ]] ||
  fail "benign path lost product dark-gate evidence: $positive"

positive_pid="$(output_field "$positive" authority_pid)"
positive_generation="$(output_field "$positive" authority_generation_sha256)"
[[ "$positive_pid" =~ ^[1-9][0-9]*$ && \
   "$positive_generation" =~ ^[0-9a-f]{64}$ ]] ||
  fail 'benign path emitted a malformed resident identity'
[[ "$(grep -c '^schema=loom-product-activation-dark-decision-v1' \
      "$TEST_ROOT/positive.dark.tsv")" -eq 1 ]] ||
  fail 'benign path did not emit exactly one dark decision'
grep -Fq $'\tdecision=DENY\tcode='"$current_code"$'\tauthorizing=false\tproduction_activation=false\t' \
  "$TEST_ROOT/positive.dark.tsv" || fail 'current projection receipt was not a dark DENY'
grep -Fq $'\tprojection_label=current_material\tcapsule_state_after=EMPTY\t' \
  "$TEST_ROOT/positive.dark.tsv" || fail 'current projection mutated the affine capsule'
grep -Fq $'\tauthority_generation_sha256='"$positive_generation"$'\tauthority_pid='"$positive_pid"$'\t' \
  "$TEST_ROOT/positive.dark.tsv" || fail 'dark receipt lost resident identity'
for event in START PEER_ACTIVATION_CAPSULE EFFECT_CLOSURE EFFECT STOP; do
  grep -Fq $'\tevent='"$event"$'\t' "$TEST_ROOT/positive.resident.tsv" ||
    fail "single resident omitted $event"
done
[[ "$(grep -c $'\tevent=START\t' "$TEST_ROOT/positive.resident.tsv")" -eq 1 && \
   "$(grep -c $'\tevent=STOP\t' "$TEST_ROOT/positive.resident.tsv")" -eq 1 ]] ||
  fail 'benign path did not use exactly one resident lifecycle'
while IFS= read -r receipt; do
  [[ "$receipt" == *$'\tgeneration_sha256='"$positive_generation"$'\t'* && \
     "$receipt" == *$'\tpid='"$positive_pid"$'\t'* ]] ||
    fail "resident identity drifted inside product probe: $receipt"
done < "$TEST_ROOT/positive.resident.tsv"
[[ -s "$TEST_ROOT/positive.membrane.tsv" ]] ||
  fail 'legacy action 9023 did not continue after the nonauthorizing dark DENY'

python_sentinel="$TEST_ROOT/python-executed"
python_output="$(expect_probe python 126 'decision_code=410' 15000 \
  "$PYTHON_PATH" -c "open('$python_sentinel', 'w').write('BAD')")"
[[ ! -e "$python_sentinel" && \
   "$python_output" == *"activation_code=$current_code"* ]] ||
  fail 'deliberate Python oracle attempt executed or bypassed the dark projection'

sabotage_sentinel="$TEST_ROOT/sabotage-executed"
set +e
sabotage_output="$(SOUNIO_LOOM_ACTIVATION_DARK_LABEL=seal \
  probe sabotage 15000 /bin/sh -c "touch '$sabotage_sentinel'" 2>&1)"
sabotage_rc=$?
set -e
[[ "$sabotage_rc" -eq 1 && \
   "$sabotage_output" == *'activation-dark-unexpected-allow'* ]] ||
  fail "positive causal sabotage was not stopped by the product gate: $sabotage_output"
[[ ! -e "$sabotage_sentinel" && ! -e "$TEST_ROOT/sabotage.membrane.tsv" ]] ||
  fail 'causal sabotage reached the legacy membrane or materialized an effect'
grep -Fq $'\tdecision=ALLOW\tcode='"$seal_code"$'\tauthorizing=false\tproduction_activation=false\t' \
  "$TEST_ROOT/sabotage.dark.tsv" || fail 'causal sabotage did not record Sounio ALLOW'
grep -Fq $'\tprojection_label=seal\tcapsule_state_after=SEALED\t' \
  "$TEST_ROOT/sabotage.dark.tsv" || fail 'causal sabotage did not exercise the affine transition'
grep -Fq $'\tevent=PEER_ACTIVATION_CAPSULE\t' "$TEST_ROOT/sabotage.resident.tsv" ||
  fail 'causal sabotage did not reach action 9031'

tampered_projection="$TEST_ROOT/projection-tampered"
cp "$PROJECTION" "$tampered_projection"
printf '\n' >> "$tampered_projection"
expect_pre_spawn_refusal projection-tamper \
  SOUNIO_LOOM_ACTIVATION_DARK_PROJECTION "$tampered_projection" \
  activation-dark-projection-hash-mismatch

for spec in \
  action:SOUNIO_LOOM_ACTIVATION_DARK_ACTION_MANIFEST:kernel_peer_activation_capsule_authority.freeze.v1:activation-dark-action-manifest-hash-mismatch \
  operational:SOUNIO_LOOM_ACTIVATION_DARK_OPERATIONAL_MANIFEST:kernel_peer_activation_capsule.runtime.v1:activation-dark-operational-manifest-hash-mismatch \
  resident:SOUNIO_LOOM_ACTIVATION_DARK_RESIDENT_MANIFEST:resident_membrane.runtime.v5:activation-dark-resident-manifest-hash-mismatch; do
  tag="${spec%%:*}"
  rest="${spec#*:}"
  variable="${rest%%:*}"
  rest="${rest#*:}"
  source="${rest%%:*}"
  marker="${rest#*:}"
  tampered="$TEST_ROOT/$tag-tampered"
  cp "$ROOT_DIR/tools/loom/$source" "$tampered"
  printf '\n' >> "$tampered"
  expect_pre_spawn_refusal "$tag-tamper" "$variable" "$tampered" "$marker"
done

receipt_sentinel="$TEST_ROOT/receipt-failure-executed"
set +e
receipt_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_RUNTIME="$RUNTIME" \
  SOUNIO_LOOM_ACTIVATION_DARK_LOG="$TEST_ROOT" \
  SOUNIO_LOOM_SUBPROCESS_MEMBRANE_LOG="$TEST_ROOT/receipt-failure.membrane.tsv" \
  SOUNIO_LOOM_RESIDENT_RECEIPT_LOG="$TEST_ROOT/receipt-failure.resident.tsv" \
  "$LOOM" subprocess-membrane-probe --root "$ROOT_DIR" --cwd "$ROOT_DIR" \
    --scope "$TEST_ROOT/scope" --deadline-ms 15000 -- /bin/sh -c \
    "touch '$receipt_sentinel'" 2>&1)"
receipt_rc=$?
set -e
[[ "$receipt_rc" -eq 1 && ! -e "$receipt_sentinel" && \
   ! -e "$TEST_ROOT/receipt-failure.membrane.tsv" ]] ||
  fail "receipt failure did not fail closed before launch: rc=$receipt_rc output=$receipt_output"

sounio_gate="$(bash "$ROOT_DIR/scripts/ci/sounio_loom_kernel_peer_activation_capsule_authority_selftest.sh")"
[[ "$sounio_gate" == *'causal_sabotage=ALLOWx9'* && \
   "$sounio_gate" == *'authority_laundering=DENY508'* ]] ||
  fail 'Sounio authority sabotage no longer attributes the semantic refusals'

if rg -n 'DENY50[2-9]|DENY510|ALLOW code=0 reason=allow' \
  "$ROOT_DIR/tools/loom/src/loom_peer_activation_capsule.ml" \
  "$ROOT_DIR/tools/loom/src/loom_membrane.ml" \
  "$ROOT_DIR/tools/loom/src/loom.ml" >/dev/null; then
  fail 'OCaml copied a Sounio semantic expected-result string'
fi

printf '%s\n' \
  "sounio-loom-product-activation-dark-gate-selftest: PASS semantic_authority=Sounio operational_realization=OCaml product_path=loom-probe resident=Sounio-v5 resident_model=single-Sounio-pid current_material=DENY${current_code}+PRODUCT_CONTINUES capsule_state=EMPTY python_oracle=ATTEMPTED+DENY410+NOT_EXECUTED causal_sabotage=ALLOW${seal_code}+PRODUCT_GATE_REFUSAL+NO_EFFECT+NO_9023 projection_tamper=refused-before-spawn action_manifest_tamper=refused-before-spawn operational_manifest_tamper=refused-before-spawn resident_manifest_tamper=refused-before-spawn receipt_failure=refused-before-launch receipts=hash-bound activation_mode=dark authorizing=false production_activation=false live_material=false expected_results_encoded_in_ocaml=false python_executed=false rust_executed=false"
