#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_PARENT="$ROOT_DIR/tools/loom/_build"
mkdir -p "$TEST_PARENT"
TEST_ROOT="$(mktemp -d "$TEST_PARENT/peer-activation-capsule-ocaml.XXXXXX")"
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
RUNTIME="$TEST_ROOT/resident-v5"
REFERENCE="$TEST_ROOT/action-9031"
RECEIPTS="$TEST_ROOT/resident.tsv"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-kernel-peer-activation-capsule-ocaml-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_OUTPUT="$RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_resident_membrane_v5.sh" >/dev/null
SOUNIO_LOOM_KERNEL_PEER_ACTIVATION_CAPSULE_OUTPUT="$REFERENCE" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_peer_activation_capsule_authority.sh" >/dev/null
(
  flock -x 9
  dune build --root "$ROOT_DIR/tools/loom" src/loom.exe >/dev/null
) 9>"$ROOT_DIR/tools/loom/_build/.dune-build.lock"

fixtures="$(printf '1\n' | "$REFERENCE")"
fixture_line() {
  local label="$1"
  printf '%s\n' "$fixtures" | grep -m1 "^CASE label=${label} "
}

fixture_code() {
  local label="$1" line
  line="$(fixture_line "$label")"
  printf '%s\n' "$line" | sed -n 's/.* EXPECT code=\([0-9][0-9]*\) FRAME .*/\1/p'
}

write_fixture() {
  local label="$1" target="$2" line frame
  line="$(fixture_line "$label")"
  frame="${line#* FRAME }"
  [[ -n "$frame" && "$frame" != "$line" ]] || fail "missing Sounio fixture $label"
  printf '%s\n' "$frame" > "$target"
}

write_fixture seal "$TEST_ROOT/seal.frame"
write_fixture consume "$TEST_ROOT/consume.frame"
write_fixture extinguish "$TEST_ROOT/extinguish.frame"
write_fixture poison "$TEST_ROOT/poison.frame"
write_fixture current_material "$TEST_ROOT/current.frame"
write_fixture python_oracle "$TEST_ROOT/python.frame"

seal_code="$(fixture_code seal)"
consume_code="$(fixture_code consume)"
extinguish_code="$(fixture_code extinguish)"
poison_code="$(fixture_code poison)"
current_code="$(fixture_code current_material)"
python_code="$(fixture_code python_oracle)"
for code in "$seal_code" "$consume_code" "$extinguish_code" "$poison_code" \
  "$current_code" "$python_code"; do
  [[ -n "$code" ]] || fail 'Sounio fixture omitted an expected decision code'
done

probe() {
  local mode="$1" seal="$2"
  shift 2
  SOUNIO_LOOM_HOOK_TEST_MODE=1 \
    SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_RUNTIME="$RUNTIME" \
    SOUNIO_LOOM_RESIDENT_RECEIPT_LOG="$RECEIPTS" \
    "$LOOM" peer-activation-capsule-probe --root "$ROOT_DIR" --mode "$mode" \
      --seal "$seal" --deadline-ms 15000 "$@"
}

current_output="$(probe current "$TEST_ROOT/current.frame")"
[[ "$current_output" == *"codes=$current_code state=EMPTY poisoned=false"* && \
  "$current_output" == *'deny_preserved=true'* ]] ||
  fail "current parent chain did not preserve EMPTY: $current_output"

python_output="$(probe python "$TEST_ROOT/python.frame")"
[[ "$python_output" == *"codes=$python_code state=EMPTY poisoned=false"* && \
  "$python_output" == *'deny_preserved=true'* ]] ||
  fail "Python oracle laundering did not preserve EMPTY: $python_output"

happy_output="$(probe happy "$TEST_ROOT/seal.frame" \
  --consume "$TEST_ROOT/consume.frame" \
  --extinguish "$TEST_ROOT/extinguish.frame")"
[[ "$happy_output" == *"codes=$seal_code,$consume_code,$extinguish_code state=EXTINCT poisoned=false"* && \
  "$happy_output" == *'sequence=3 '* ]] ||
  fail "happy affine lifecycle did not become EXTINCT exactly once: $happy_output"

deny_output="$(probe deny-preserves "$TEST_ROOT/seal.frame" \
  --deny "$TEST_ROOT/current.frame" --consume "$TEST_ROOT/consume.frame" \
  --extinguish "$TEST_ROOT/extinguish.frame")"
[[ "$deny_output" == *"codes=$current_code,$seal_code,$consume_code,$extinguish_code state=EXTINCT poisoned=false"* && \
  "$deny_output" == *'deny_preserved=true'* ]] ||
  fail "Sounio DENY mutated or burned the capsule: $deny_output"

poison_output="$(probe poison "$TEST_ROOT/seal.frame" \
  --consume "$TEST_ROOT/consume.frame" --poison "$TEST_ROOT/poison.frame")"
[[ "$poison_output" == *"codes=$seal_code,$consume_code,$poison_code state=POISONED poisoned=true"* ]] ||
  fail "typed poison did not irreversibly retire the capsule: $poison_output"

for mode in replay mismatch timeout eof; do
  output="$(probe "$mode" "$TEST_ROOT/seal.frame" \
    --consume "$TEST_ROOT/consume.frame")"
  [[ "$output" == *"codes=$seal_code state=POISONED poisoned=true control_refused=true reuse_refused=true"* ]] ||
    fail "$mode did not poison and refuse reuse: $output"
done

grep -Fq $'\tevent=PEER_ACTIVATION_CAPSULE\t' "$RECEIPTS" ||
  fail 'peer-activation receipt is missing'
grep -Fq $'\tparent_9031_manifest_sha256=f2da55138bcfe5a8a2c65ebd79c1e534f152b33af5c6cc3d1f2b4eb3b4af6e7e\t' \
  "$RECEIPTS" || fail 'receipt omitted frozen action 9031'
grep -Fq $'\tresident_manifest_sha256=b3cf8c1e0524be35fc67b2b5a779bad9a9291195d65dc82dbc87595396fb5353\t' \
  "$RECEIPTS" || fail 'receipt omitted frozen resident v5 manifest'

tampered_manifest="$TEST_ROOT/action-9031.freeze.v1"
cp "$ROOT_DIR/tools/loom/kernel_peer_activation_capsule_authority.freeze.v1" \
  "$tampered_manifest"
printf '\n' >> "$tampered_manifest"
set +e
manifest_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_RUNTIME="$RUNTIME" \
  SOUNIO_LOOM_KERNEL_PEER_ACTIVATION_CAPSULE_MANIFEST="$tampered_manifest" \
  "$LOOM" peer-activation-capsule-probe --root "$ROOT_DIR" --mode current \
    --seal "$TEST_ROOT/current.frame" --deadline-ms 2000 2>&1)"
manifest_rc=$?
set -e
[[ "$manifest_rc" -eq 1 && \
  "$manifest_output" == *'peer-activation-capsule-manifest-hash-mismatch'* ]] ||
  fail "action 9031 manifest tamper did not fail before spawn: $manifest_output"

tampered_resident_manifest="$TEST_ROOT/resident-v5.runtime"
cp "$ROOT_DIR/tools/loom/resident_membrane.runtime.v5" \
  "$tampered_resident_manifest"
printf '\n' >> "$tampered_resident_manifest"
set +e
resident_manifest_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_RUNTIME="$RUNTIME" \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_MANIFEST="$tampered_resident_manifest" \
  "$LOOM" peer-activation-capsule-probe --root "$ROOT_DIR" --mode current \
    --seal "$TEST_ROOT/current.frame" --deadline-ms 2000 2>&1)"
resident_manifest_rc=$?
set -e
[[ "$resident_manifest_rc" -eq 1 && \
  "$resident_manifest_output" == *'resident-runtime-v5-manifest-hash-mismatch'* ]] ||
  fail "resident v5 manifest tamper did not fail before spawn: $resident_manifest_output"

tampered_runtime="$TEST_ROOT/resident-v5-tampered"
cp "$RUNTIME" "$tampered_runtime"
printf '\n' >> "$tampered_runtime"
chmod 0755 "$tampered_runtime"
set +e
runtime_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_RUNTIME="$tampered_runtime" \
  "$LOOM" peer-activation-capsule-probe --root "$ROOT_DIR" --mode current \
    --seal "$TEST_ROOT/current.frame" --deadline-ms 2000 2>&1)"
runtime_rc=$?
set -e
[[ "$runtime_rc" -eq 1 && \
  "$runtime_output" == *'resident-runtime-hash-mismatch'* ]] ||
  fail "resident v5 runtime tamper did not fail before spawn: $runtime_output"

sounio_authority_gate="$(bash "$ROOT_DIR/scripts/ci/sounio_loom_kernel_peer_activation_capsule_authority_selftest.sh")"
[[ "$sounio_authority_gate" == *'causal_sabotage=ALLOWx9'* && \
  "$sounio_authority_gate" == *'authority_laundering=DENY508'* ]] ||
  fail 'frozen Sounio causal sabotage no longer attributes the laundering refusal'

if rg -n 'DENY50[2-9]|DENY510|ALLOW code=0 reason=allow' \
  "$ROOT_DIR/tools/loom/src/loom_peer_activation_capsule.ml" \
  "$ROOT_DIR/tools/loom/src/loom_resident.ml" \
  "$ROOT_DIR/tools/loom/src/loom.ml" >/dev/null; then
  fail 'OCaml copied a semantic expected-result string'
fi

printf '%s\n' \
  'sounio-loom-kernel-peer-activation-capsule-ocaml-selftest: PASS semantic_authority=Sounio operational_realization=OCaml resident=Sounio-v5 resident_model=single-Sounio-pid lifecycle=EMPTY-SEALED-CONSUMED-EXTINCT-POISONED happy=ALLOWx3 poison=ALLOWx3 current_material=DENY502+STATE_PRESERVED python_oracle=DENY508+STATE_PRESERVED semantic_deny=STATE_PRESERVED+RECOVERY_ALLOWED replay=POISON+REUSE_REFUSED mismatch=POISON+REUSE_REFUSED timeout=POISON+REUSE_REFUSED eof=POISON+REUSE_REFUSED affirmative_absence=required-before-EXTINCT receipts=hash-bound action_manifest_tamper=refused-before-spawn resident_manifest_tamper=refused-before-spawn runtime_tamper=refused-before-spawn causal_sabotage=ALLOWx9 ocaml_expected_results=absent same_uid_peer_isolation=true capsule_material=false production_activation=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false python_executed=false rust_executed=false'
