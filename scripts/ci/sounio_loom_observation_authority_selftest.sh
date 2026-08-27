#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
CANARY="$ROOT_DIR/scripts/ci/sounio_loom_pod_replay_canary.sh"
NORMAL_LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
NORMAL_ADAPTER="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-continuity-runtime"
MODULE="$ROOT_DIR/stdlib/coordination/loom_continuity.sio"
BUILD_ADAPTER="$ROOT_DIR/scripts/dev/build_sounio_loom_continuity_adapter.sh"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-observation-authority.XXXXXX")"
KEY_ROOT="$TEST_ROOT/keys"
AUTHORITY_SOCKET="$TEST_ROOT/journal-authority.sock"
AUTHORITY_STATE="$TEST_ROOT/journal-authority-state"
PANE_ID='loom-pod-replay:terminal'
LANE='pane-6c6f6f6d2d706f642d7265706c61793a7465726d696e616c'
AUTHORITY_EPOCH=1
AUTHORITY_PID=''

fail() {
  printf 'sounio-loom-observation-authority-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

field() {
  local name="$1" value="$2"
  sed -n "s/.* ${name}=\([^ ]*\).*/\1/p" <<< "$value"
}

json_string_value() {
  local path="$1" key="$2"
  sed -n "s/.*\"${key}\":\"\([^\"]*\)\".*/\1/p" "$path" | head -1
}

key_value() {
  local key="$1" path="$2" value
  value="$(sed -n "s/^${key}=//p" "$path")"
  [[ -n "$value" && "$value" != *$'\n'* ]] || \
    fail "$path must contain exactly one $key"
  printf '%s\n' "$value"
}

principal_id() {
  openssl pkey -pubin -in "$1" -outform DER 2>/dev/null | \
    sha256sum | awk '{print $1}'
}

domain_digest() {
  local domain="$1" value="$2"
  printf 'loom-continuity-fact-digest-v1\0%s\0%s' "$domain" "$value" | \
    sha256sum | awk '{print $1}'
}

compact_token() {
  local domain="$1" value="$2" digest prefix
  digest="$(printf '%s\0%s' "$domain" "$value" | sha256sum | awk '{print $1}')"
  prefix="${digest:0:15}"
  printf '%s\n' "$((16#$prefix + 1))"
}

kill_generation() {
  local root="$1" loom="$2" status pid bridge_file bridge_pid
  status="$(
    SOUNIO_COORD_RUNTIME_MODE=local "$loom" status \
      --state-dir "$root/loom" --cwd "$ROOT_DIR" \
      --agent beagle-workbench --lane "$LANE" 2>/dev/null || true
  )"
  for pid in "$(field daemon_pid "$status")" \
    "$(field guardian_pid "$status")" "$(field harness_pid "$status")"; do
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] && kill -9 "$pid" 2>/dev/null || true
  done
  for bridge_file in "$root"/bridge-*.pid; do
    [[ -f "$bridge_file" ]] || continue
    bridge_pid="$(cat "$bridge_file")"
    [[ "$bridge_pid" =~ ^[1-9][0-9]*$ ]] && \
      kill -9 "$bridge_pid" 2>/dev/null || true
  done
}

cleanup() {
  local root
  for root in "$TEST_ROOT"/journal-* "$TEST_ROOT"/revocation; do
    [[ -d "$root" ]] && kill_generation "$root" "$NORMAL_LOOM" || true
    [[ -d "$root" && -x "${MUTANT_LOOM:-}" ]] && \
      kill_generation "$root" "$MUTANT_LOOM" || true
  done
  [[ "$AUTHORITY_PID" =~ ^[1-9][0-9]*$ ]] && \
    kill "$AUTHORITY_PID" 2>/dev/null || true
  [[ "$AUTHORITY_PID" =~ ^[1-9][0-9]*$ ]] && \
    wait "$AUTHORITY_PID" 2>/dev/null || true
  if [[ "${SOUNIO_LOOM_TEST_KEEP:-0}" == 1 ]]; then
    printf 'sounio-loom-observation-authority-selftest: kept %s\n' "$TEST_ROOT" >&2
    return
  fi
  chmod -R u+w "$TEST_ROOT" 2>/dev/null || true
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

run_phase() {
  local root="$1" uid="$2" phase="$3" loom="$4"
  env \
    POD_UID="$uid" \
    POD_NAME=loom-observation-authority-0 \
    SOUNIO_CANARY_SOURCE_ROOT="$ROOT_DIR" \
    SOUNIO_LOOM_POD_CANARY_ROOT="$root" \
    SOUNIO_LOOM_BINARY="$loom" \
    SOUNIO_LOOM_REQUIRE_SIGNED_RECEIPTS=1 \
    SOUNIO_LOOM_REQUIRE_OBSERVATION_AUTHORITY=1 \
    SOUNIO_LOOM_SIGNING_KEY="$KEY_ROOT/signer-private.pem" \
    SOUNIO_LOOM_VERIFY_KEY="$KEY_ROOT/signer-public.pem" \
    SOUNIO_LOOM_OBSERVER_VERIFY_KEY="$KEY_ROOT/observer-public.pem" \
    SOUNIO_LOOM_JOURNAL_AUTHORITY_SOCKET="$AUTHORITY_SOCKET" \
    SOUNIO_LOOM_JOURNAL_AUTHORITY_VERIFY_KEY="$KEY_ROOT/journal-public.pem" \
    SOUNIO_LOOM_JOURNAL_AUTHORITY_EPOCH="$AUTHORITY_EPOCH" \
    SOUNIO_LOOM_CONTINUITY_ADAPTER="$NORMAL_ADAPTER" \
    bash "$CANARY" "$phase"
}

receipt_path() {
  local root="$1" matches=()
  mapfile -t matches < <(
    find "$root/loom" -name sounio-continuity.receipt -type f -print
  )
  [[ "${#matches[@]}" -eq 1 ]] || \
    fail "$root contains ${#matches[@]} predecessor receipts instead of one"
  printf '%s\n' "${matches[0]}"
}

generation_count() {
  find "$1/loom" -name sounio-continuity.receipt -type f -print \
    2>/dev/null | wc -l
}

bind_canary_to_subject_receipt() {
  local root="$1" receipt="$2" state="$root/phase.env" temporary digest
  digest="$(sha256sum "$receipt" | awk '{print $1}')"
  temporary="$state.$$"
  awk -v digest="$digest" '
    /^receipt_one=/ { print "receipt_one=" digest; changed=1; next }
    { print }
    END { if (changed != 1) exit 42 }
  ' "$state" > "$temporary" || fail 'could not bind canary state to forged receipt'
  mv "$temporary" "$state"
}

rewrite_semantic_journal() {
  local journal="$1" temporary="$journal.rewritten" changed=0
  local seq old_previous old_hash utc kind payload context epoch principal signature
  local source_previous previous hash
  source_previous="$(printf '0%.0s' {1..64})"
  previous="$source_previous"
  : > "$temporary"
  while IFS=$'\t' read -r seq old_previous old_hash utc kind payload context \
      epoch principal signature; do
    [[ -n "$seq" ]] || continue
    [[ -n "$signature" ]] || fail 'authority journal did not contain ten fields'
    if [[ "$changed" -eq 0 && \
      ( "$kind" == OBSERVER_ATTACHED || "$kind" == WAKE ) ]]; then
      payload="${payload}00"
      changed=1
    fi
    [[ "$old_previous" == "$source_previous" ]] || \
      fail "source journal chain was discontinuous at sequence $seq"
    hash="$(printf '%s\t%s\t%s\t%s\t%s' \
      "$seq" "$previous" "$utc" "$kind" "$payload" | \
      sha256sum | awk '{print $1}')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$seq" "$previous" "$hash" "$utc" "$kind" "$payload" \
      "$context" "$epoch" "$principal" "$signature" >> "$temporary"
    source_previous="$old_hash"
    previous="$hash"
  done < "$journal"
  [[ "$changed" -eq 1 ]] || fail 'no semantically free event was available to rewrite'
  mv "$temporary" "$journal"
  REWRITTEN_SEMANTIC_HEAD="$previous"
}

assert_internal_hash_chain() {
  local journal="$1" seq previous hash utc kind payload context epoch principal signature
  local expected_previous expected_hash
  expected_previous="$(printf '0%.0s' {1..64})"
  while IFS=$'\t' read -r seq previous hash utc kind payload context epoch principal signature; do
    [[ "$previous" == "$expected_previous" ]] || \
      fail "rewritten chain has a previous-hash mismatch at sequence $seq"
    expected_hash="$(printf '%s\t%s\t%s\t%s\t%s' \
      "$seq" "$previous" "$utc" "$kind" "$payload" | \
      sha256sum | awk '{print $1}')"
    [[ "$hash" == "$expected_hash" ]] || \
      fail "rewritten chain has an event-hash mismatch at sequence $seq"
    expected_previous="$hash"
  done < "$journal"
}

forge_receipt_v3() {
  local receipt="$1" facts key_id adapter_sha256 verdict payload signature
  local facts_sha256 payload_sha256 signature_base64 semantic_digest
  local generation_digest fingerprint_digest guardian_digest guardian_head fields=()
  local decision_generation decision_fingerprint
  facts="$(key_value facts "$receipt")"
  read -r -a fields <<< "$facts"
  [[ "${#fields[@]}" -eq 15 ]] || fail 'genesis receipt did not contain 15 facts'
  fields[4]="$(compact_token semantic-head "$REWRITTEN_SEMANTIC_HEAD")"
  guardian_head="$(awk -F '\t' 'NF { head=$3 } END { print head }' \
    "$(dirname "$receipt")/guardian.tsv")"
  fields[5]="$(compact_token guardian-head "$guardian_head")"
  facts="${fields[*]}"
  key_id="$(key_value key_id "$receipt")"
  adapter_sha256="$(key_value adapter_sha256 "$receipt")"
  generation_digest="$(key_value decision_generation_sha256 "$receipt")"
  fingerprint_digest="$(key_value decision_generation_fingerprint_sha256 "$receipt")"
  decision_generation="$(key_value decision_generation "$receipt")"
  decision_fingerprint="$(key_value decision_generation_fingerprint "$receipt")"
  guardian_digest="$(domain_digest guardian-head "$guardian_head")"
  semantic_digest="$(domain_digest semantic-head "$REWRITTEN_SEMANTIC_HEAD")"
  facts_sha256="$(printf '%s\n' "$facts" | sha256sum | awk '{print $1}')"
  verdict='SOUNIO_CONTINUITY_ACCEPT schema=loom-native-continuity-v2 authenticity=ed25519'
  payload="$TEST_ROOT/signer-payload-$RANDOM.txt"
  signature="$TEST_ROOT/signer-signature-$RANDOM.bin"
  {
    printf 'schema=loom-native-continuity-signed-payload-v2\n'
    printf 'algorithm=ed25519\nkey_id=%s\n' "$key_id"
    printf 'adapter_sha256=%s\nfacts_sha256=%s\nfacts=%s\n' \
      "$adapter_sha256" "$facts_sha256" "$facts"
    printf 'fact_digest_schema=loom-continuity-fact-digest-v1\n'
    printf 'decision_generation=%s\n' "$decision_generation"
    printf 'decision_generation_fingerprint=%s\n' "$decision_fingerprint"
    printf 'decision_semantic_head=%s\n' "$REWRITTEN_SEMANTIC_HEAD"
    printf 'decision_guardian_head=%s\n' "$guardian_head"
    printf 'decision_generation_sha256=%s\n' "$generation_digest"
    printf 'decision_generation_fingerprint_sha256=%s\n' "$fingerprint_digest"
    printf 'decision_semantic_head_sha256=%s\n' "$semantic_digest"
    printf 'decision_guardian_head_sha256=%s\n' "$guardian_digest"
    printf 'verdict=%s\n' "$verdict"
  } > "$payload"
  payload_sha256="$(sha256sum "$payload" | awk '{print $1}')"
  openssl pkeyutl -sign -rawin -inkey "$KEY_ROOT/signer-private.pem" \
    -in "$payload" -out "$signature"
  signature_base64="$(openssl base64 -A -in "$signature")"
  {
    printf 'schema=loom-native-continuity-receipt-v3\n'
    printf 'algorithm=ed25519\nkey_id=%s\n' "$key_id"
    printf 'adapter_sha256=%s\nfacts_sha256=%s\nfacts=%s\n' \
      "$adapter_sha256" "$facts_sha256" "$facts"
    printf 'fact_digest_schema=loom-continuity-fact-digest-v1\n'
    printf 'decision_generation=%s\n' "$decision_generation"
    printf 'decision_generation_fingerprint=%s\n' "$decision_fingerprint"
    printf 'decision_semantic_head=%s\n' "$REWRITTEN_SEMANTIC_HEAD"
    printf 'decision_guardian_head=%s\n' "$guardian_head"
    printf 'decision_generation_sha256=%s\n' "$generation_digest"
    printf 'decision_generation_fingerprint_sha256=%s\n' "$fingerprint_digest"
    printf 'decision_semantic_head_sha256=%s\n' "$semantic_digest"
    printf 'decision_guardian_head_sha256=%s\n' "$guardian_digest"
    printf 'verdict=%s\n' "$verdict"
    printf 'signed_payload_sha256=%s\nsignature_base64=%s\n' \
      "$payload_sha256" "$signature_base64"
  } > "$receipt"
}

forge_measurement_v2() {
  local root="$1" uid="$2" receipt="$3" generation_dir
  local generation fingerprint semantic_head guardian_head semantic_digest guardian_digest
  local generation_digest fingerprint_digest semantic_journal_digest guardian_journal_digest
  local descriptor_digest journal_principal journal_epoch journal_checkpoint
  local observer_key_id observer_principal subject_key_id subject_principal
  local subject_receipt_digest subject_facts_digest adapter_digest output payload signature
  local payload_digest signature_base64
  generation_dir="$(dirname "$receipt")"
  generation="$(key_value instance_id "$generation_dir/session.state")"
  fingerprint="$(json_string_value "$root/spawn-$uid.json" generationFingerprint)"
  semantic_head="$REWRITTEN_SEMANTIC_HEAD"
  guardian_head="$(awk -F '\t' 'NF { head=$3 } END { print head }' \
    "$generation_dir/guardian.tsv")"
  generation_digest="$(domain_digest generation "$generation")"
  fingerprint_digest="$(domain_digest generation-fingerprint "$fingerprint")"
  semantic_digest="$(domain_digest semantic-head "$semantic_head")"
  guardian_digest="$(domain_digest guardian-head "$guardian_head")"
  semantic_journal_digest="$(sha256sum "$generation_dir/journal.tsv" | awk '{print $1}')"
  guardian_journal_digest="$(sha256sum "$generation_dir/guardian.tsv" | awk '{print $1}')"
  descriptor_digest="$(sha256sum "$generation_dir/session.state" | awk '{print $1}')"
  journal_principal="$(awk -F '\t' 'NF { print $9; exit }' "$generation_dir/journal.tsv")"
  journal_epoch="$(awk -F '\t' 'NF { print $8; exit }' "$generation_dir/journal.tsv")"
  journal_checkpoint="$(printf 'loom-journal-authority-checkpoint-v1\0%s\0%s\0%s\0%s\0%s' \
    "$journal_principal" "$journal_epoch" "$semantic_journal_digest" \
    "$guardian_journal_digest" "$descriptor_digest" | sha256sum | awk '{print $1}')"
  observer_key_id="$(sha256sum "$KEY_ROOT/observer-public.pem" | awk '{print $1}')"
  observer_principal="$(principal_id "$KEY_ROOT/observer-public.pem")"
  subject_key_id="$(key_value key_id "$receipt")"
  subject_principal="$(principal_id "$KEY_ROOT/signer-public.pem")"
  subject_receipt_digest="$(sha256sum "$receipt" | awk '{print $1}')"
  subject_facts_digest="$(key_value facts_sha256 "$receipt")"
  adapter_digest="$(sha256sum "$NORMAL_ADAPTER" | awk '{print $1}')"
  output="$generation_dir/sounio-continuity.observer-attestation"
  payload="$TEST_ROOT/observer-payload-$RANDOM.txt"
  signature="$TEST_ROOT/observer-signature-$RANDOM.bin"
  {
    printf 'schema=loom-independent-measurement-payload-v2\nalgorithm=ed25519\n'
    printf 'observer_key_id=%s\nobserver_principal_id=%s\n' \
      "$observer_key_id" "$observer_principal"
    printf 'subject_signer_key_id=%s\nsubject_principal_id=%s\n' \
      "$subject_key_id" "$subject_principal"
    printf 'subject_receipt_sha256=%s\nsubject_facts_sha256=%s\nsubject_adapter_sha256=%s\n' \
      "$subject_receipt_digest" "$subject_facts_digest" "$adapter_digest"
    printf 'measurement_source=write-authorized-generation-artifacts-v2\n'
    printf 'measured_generation=%s\nmeasured_generation_fingerprint=%s\n' \
      "$generation" "$fingerprint"
    printf 'measured_semantic_head=%s\nmeasured_guardian_head=%s\n' \
      "$semantic_head" "$guardian_head"
    printf 'measured_generation_sha256=%s\n' "$generation_digest"
    printf 'measured_generation_fingerprint_sha256=%s\n' "$fingerprint_digest"
    printf 'measured_semantic_head_sha256=%s\n' "$semantic_digest"
    printf 'measured_guardian_head_sha256=%s\n' "$guardian_digest"
    printf 'semantic_journal_sha256=%s\nguardian_journal_sha256=%s\n' \
      "$semantic_journal_digest" "$guardian_journal_digest"
    printf 'descriptor_sha256=%s\n' "$descriptor_digest"
    printf 'journal_authority_principal_id=%s\njournal_authority_epoch=%s\n' \
      "$journal_principal" "$journal_epoch"
    printf 'journal_authority_checkpoint_sha256=%s\n' "$journal_checkpoint"
    printf 'observation=write-authorized-generation-measurement\n'
  } > "$payload"
  payload_digest="$(sha256sum "$payload" | awk '{print $1}')"
  openssl pkeyutl -sign -rawin -inkey "$KEY_ROOT/observer-private.pem" \
    -in "$payload" -out "$signature"
  signature_base64="$(openssl base64 -A -in "$signature")"
  {
    sed '1s/payload/attestation/' "$payload"
    printf 'signed_payload_sha256=%s\nsignature_base64=%s\n' \
      "$payload_digest" "$signature_base64"
  } > "$output"
}

make_loom_mutant() {
  local source="$TEST_ROOT/loom-mutant" mutation_count
  cp -a "$ROOT_DIR/tools/loom" "$source"
  mutation_count="$(rg -c '^let journal_authority_signature_is_valid public_key directory payload signature =$' \
    "$source/src/loom.ml")"
  [[ "$mutation_count" -eq 1 ]] || \
    fail "expected one journal signature verifier target, got $mutation_count"
  awk '
    BEGIN { changed=0; skip=0 }
    !changed && $0 == "let journal_authority_signature_is_valid public_key directory payload signature =" {
      print "let journal_authority_signature_is_valid _public_key _directory _payload _signature ="
      print "  let _unused_verifier = journal_verify_with_key in"
      print "  true"
      changed=1
      skip=1
      next
    }
    skip { skip=0; next }
    { print }
    END { if (changed != 1) exit 42 }
  ' "$source/src/loom.ml" > "$source/src/loom.ml.mutated" || \
    fail 'could not apply journal-signature-verifier sabotage'
  mv "$source/src/loom.ml.mutated" "$source/src/loom.ml"
  dune build --root "$source" >/dev/null
  MUTANT_LOOM="$source/_build/default/src/loom.exe"
  [[ -x "$MUTANT_LOOM" ]] || fail 'mutant Loom executable was not built'
}

measure_generation() {
  local root="$1" uid="$2" loom="$3" receipt generation output
  receipt="$(receipt_path "$root")"
  generation="$(json_string_value "$root/spawn-$uid.json" loomInstanceId)"
  output="$(dirname "$receipt")/sounio-continuity.observer-attestation"
  env \
    SOUNIO_LOOM_REQUIRE_OBSERVATION_AUTHORITY=1 \
    SOUNIO_LOOM_JOURNAL_AUTHORITY_SOCKET="$AUTHORITY_SOCKET" \
    SOUNIO_LOOM_JOURNAL_AUTHORITY_VERIFY_KEY="$KEY_ROOT/journal-public.pem" \
    SOUNIO_LOOM_JOURNAL_AUTHORITY_EPOCH="$AUTHORITY_EPOCH" \
    "$loom" measure-continuity-generation \
      --state-dir "$root/loom" --pane-id "$PANE_ID" \
      --generation "$generation" --receipt "$receipt" \
      --subject-public-key "$KEY_ROOT/signer-public.pem" \
      --observer-private-key "$KEY_ROOT/observer-private.pem" \
      --observer-public-key "$KEY_ROOT/observer-public.pem" \
      --out "$output" --adapter "$NORMAL_ADAPTER" >/dev/null
  [[ -s "$output" ]] || fail 'normal authority measurement was not created'
}

prepare_rewritten_witness() {
  local root="$1" uid="$2" loom="$3" phase receipt generation journal
  phase="$(run_phase "$root" "$uid" phase-one "$loom")"
  [[ "$phase" == *CANARY_PHASE_ONE* ]] || fail 'authority genesis failed'
  receipt="$(receipt_path "$root")"
  generation="$(json_string_value "$root/spawn-$uid.json" loomInstanceId)"
  [[ -n "$generation" ]] || fail 'authority genesis omitted generation identity'
  kill_generation "$root" "$loom"
  journal="$(dirname "$receipt")/journal.tsv"
  rewrite_semantic_journal "$journal"
  assert_internal_hash_chain "$journal"
  forge_receipt_v3 "$receipt"
  bind_canary_to_subject_receipt "$root" "$receipt"
  forge_measurement_v2 "$root" "$uid" "$receipt"
}

observation_authority_frame() {
  local measured_semantic_start="${1:-21}" values=(
    9005 1002 1101 1201 1301 1401 1501 1
    2101 2201 2301 2401 2101 2201 2301 2401
  )
  local start offset
  for start in 1 11 21 31 1 11 "$measured_semantic_start" 31; do
    for offset in 0 1 2 3 4 5 6 7; do values+=("$((start + offset))"); done
  done
  printf '%s' "${values[*]}"
}

make_digest_mutant() {
  local mutated="$TEST_ROOT/loom_continuity_digest_mutated.sio"
  local mutant="$TEST_ROOT/sounio-loom-continuity-digest-mutant"
  awk '
    BEGIN { in_function=0; skip_body=0; changed=0 }
    $0 == "fn full_digest_vectors_agree(" { in_function=1; print; next }
    in_function && !skip_body {
      print
      if ($0 == ") -> bool {") { print "    true"; skip_body=1; changed++ }
      next
    }
    skip_body { if ($0 == "}") { print; in_function=0; skip_body=0 }; next }
    { print }
    END { if (changed != 1) exit 42 }
  ' "$MODULE" > "$mutated" || fail 'could not apply full-digest sabotage'
  SOUNIO_LOOM_CONTINUITY_PREBUILT= \
  SOUNIO_LOOM_CONTINUITY_MODULE="$mutated" \
  SOUNIO_LOOM_CONTINUITY_OUTPUT="$mutant" "$BUILD_ADAPTER" >/dev/null
  printf '%s\n' "$mutant"
}

run_digest_treatment() {
  local frame output rc=0
  frame="$(observation_authority_frame 22)"
  set +e
  output="$(printf '%s\n' "$frame" | "$NORMAL_ADAPTER")"
  rc=$?
  set -e
  [[ "$rc" -eq 42 && "$output" == \
    'SOUNIO_CONTINUITY_REFUSE reason=observation-authority-admission' ]] || \
    fail "full-digest alias witness was not refused: rc=$rc output=$output"
  printf '%s\n' \
    'SOUNIO_LOOM_FULL_DIGEST_TREATMENT=PASS compact_aliases=identical full_semantic_digest=divergent sounio=refused'
}

run_digest_control() {
  local mutant frame output
  mutant="$(make_digest_mutant)"
  frame="$(observation_authority_frame 22)"
  output="$(printf '%s\n' "$frame" | "$mutant")" || \
    fail 'targeted full-digest sabotage did not admit the alias witness'
  [[ "$output" == \
    'SOUNIO_CONTINUITY_PRESPAWN_ACCEPT schema=loom-native-pre-spawn-v3 authority=three-principals+full-sha256-agreement' ]] || \
    fail "full-digest control returned a non-canonical verdict: $output"
  printf '%s\n' \
    'SOUNIO_LOOM_FULL_DIGEST_CONTROL=PASS intervention=full_digest_vectors_agree-always-true alias_witness=admitted'
}

run_journal_treatment() {
  local root="$TEST_ROOT/journal-treatment" before after output rc=0
  prepare_rewritten_witness "$root" journal-treatment-one "$NORMAL_LOOM"
  before="$(generation_count "$root")"
  set +e
  output="$(run_phase "$root" journal-treatment-two phase-two "$NORMAL_LOOM" 2>&1)"
  rc=$?
  set -e
  [[ "$rc" -ne 0 && "$output" == *sounio-journal-authority-event-signature-invalid* ]] || \
    fail "rewritten journal was not refused by its stale authority signature: rc=$rc output=$output"
  after="$(generation_count "$root")"
  [[ "$after" -eq "$before" ]] || fail 'journal treatment created a successor receipt'
  printf '%s\n' \
    'SOUNIO_LOOM_JOURNAL_AUTHORITY_TREATMENT=PASS hash_chain=valid signer_receipt=resigned observer_measurement=resigned journal_signature=stale refused_before_spawn=1 successor_created=0'
}

run_positive() {
  local root="$TEST_ROOT/journal-positive" phase spawn
  phase="$(run_phase "$root" journal-positive-one phase-one "$NORMAL_LOOM")"
  [[ "$phase" == *CANARY_PHASE_ONE* ]] || fail 'positive authority genesis failed'
  kill_generation "$root" "$NORMAL_LOOM"
  measure_generation "$root" journal-positive-one "$NORMAL_LOOM"
  phase="$(run_phase "$root" journal-positive-two phase-two "$NORMAL_LOOM")"
  [[ "$phase" == *CANARY_PHASE_TWO* ]] || fail 'positive authority successor failed'
  spawn="$root/spawn-journal-positive-two.json"
  grep -q '"sounioPolicyObservationAuthorityVerified":true' "$spawn" || \
    fail 'positive successor omitted observation-authority status'
  [[ "$(generation_count "$root")" -eq 2 ]] || \
    fail 'positive authority control did not create one successor'
  printf '%s\n' \
    'SOUNIO_LOOM_OBSERVATION_AUTHORITY_POSITIVE=PASS journals=write-authorized measurement=independent full_digest=agreed successor_created=1'
}

run_journal_control() {
  local root="$TEST_ROOT/journal-control" phase spawn
  [[ -x "${MUTANT_LOOM:-}" ]] || make_loom_mutant
  prepare_rewritten_witness "$root" journal-control-one "$MUTANT_LOOM"
  phase="$(run_phase "$root" journal-control-two phase-two "$MUTANT_LOOM")"
  [[ "$phase" == *CANARY_PHASE_TWO* ]] || \
    fail 'targeted journal verifier sabotage did not admit the control successor'
  spawn="$root/spawn-journal-control-two.json"
  grep -q '"sounioPolicyObservationAuthorityVerified":true' "$spawn" || \
    fail 'control successor omitted observation-authority status'
  grep -q '"sounioPolicyFullDigestAgreementVerified":true' "$spawn" || \
    fail 'control successor omitted full-digest status'
  [[ "$(generation_count "$root")" -eq 2 ]] || \
    fail 'journal control did not create exactly one successor receipt'
  printf '%s\n' \
    'SOUNIO_LOOM_JOURNAL_AUTHORITY_CONTROL=PASS intervention=journal_authority_signature_is_valid-always-true same_rewritten_witness=admitted successor_created=1'
}

run_revocation() {
  local root="$TEST_ROOT/revocation" phase receipt journal output rc=0
  phase="$(run_phase "$root" revocation-one phase-one "$NORMAL_LOOM")"
  [[ "$phase" == *CANARY_PHASE_ONE* ]] || fail 'revocation genesis failed'
  receipt="$(receipt_path "$root")"
  journal="$(dirname "$receipt")/journal.tsv"
  kill_generation "$root" "$NORMAL_LOOM"
  set +e
  output="$(env \
    SOUNIO_LOOM_REQUIRE_JOURNAL_AUTHORITY=1 \
    SOUNIO_LOOM_JOURNAL_AUTHORITY_VERIFY_KEY="$KEY_ROOT/journal-public.pem" \
    SOUNIO_LOOM_JOURNAL_AUTHORITY_EPOCH="$AUTHORITY_EPOCH" \
    SOUNIO_LOOM_JOURNAL_AUTHORITY_REVOKED_EPOCHS="$AUTHORITY_EPOCH" \
    "$NORMAL_LOOM" verify-journal --journal "$journal" 2>&1)"
  rc=$?
  set -e
  [[ "$rc" -ne 0 && "$output" == *sounio-journal-authority-epoch-revoked* ]] || \
    fail "revoked journal epoch was not refused: rc=$rc output=$output"
  printf '%s\n' \
    'SOUNIO_LOOM_JOURNAL_AUTHORITY_REVOCATION=PASS active_epoch=1 revoked_epoch=1 verification=refused'
}

authority_request() {
  local context="$1" sequence="$2" previous="$3" event_hash="$4"
  printf 'SOUNIO_JOURNAL_AUTHORITY_V1\tSIGN\t%s\t%s\t%s\t%s\n' \
    "$context" "$sequence" "$previous" "$event_hash" | \
    socat -t 5 - "UNIX-CONNECT:$AUTHORITY_SOCKET"
}

run_authority_state() {
  local context zero head_one head_two first retry refusal state temporary tampered
  context="$(printf 'observation-authority-state-control' | sha256sum | awk '{print $1}')"
  zero="$(printf '0%.0s' {1..64})"
  head_one="$(printf 'authority-event-one' | sha256sum | awk '{print $1}')"
  head_two="$(printf 'authority-event-two' | sha256sum | awk '{print $1}')"
  first="$(authority_request "$context" 1 "$zero" "$head_one")"
  [[ "$first" == $'OK\tSIGNED\t'* ]] || fail "first authority request failed: $first"
  retry="$(authority_request "$context" 1 "$zero" "$head_one")"
  [[ "$retry" == "$first" ]] || fail 'exact authority retry changed its signature'
  refusal="$(authority_request "$context" 3 "$head_one" "$head_two")"
  [[ "$refusal" == $'REFUSE\tnon-monotonic-event' ]] || \
    fail "authority accepted a sequence gap: $refusal"
  state="$AUTHORITY_STATE/$context.state"
  temporary="$state.tampered"
  awk '
    /^signature_base64=/ { print "signature_base64=AAAAAAAA"; changed=1; next }
    { print }
    END { if (changed != 1) exit 42 }
  ' "$state" > "$temporary" || fail 'could not tamper authority state signature'
  mv "$temporary" "$state"
  tampered="$(authority_request "$context" 1 "$zero" "$head_one")"
  [[ "$tampered" == $'REFUSE\tstored-state-signature-invalid' ]] || \
    fail "authority trusted a tampered durable state: $tampered"
  printf '%s\n' \
    'SOUNIO_LOOM_JOURNAL_AUTHORITY_STATE=PASS exact_retry=idempotent sequence_gap=refused stored_signature_tamper=refused file_and_directory_fsync=enabled'
}

initialize() {
  command -v openssl >/dev/null || fail 'OpenSSL is required'
  command -v socat >/dev/null || fail 'socat is required for authority protocol controls'
  mkdir -p "$KEY_ROOT" "$AUTHORITY_STATE"
  local role
  for role in signer observer journal; do
    openssl genpkey -algorithm ED25519 -out "$KEY_ROOT/$role-private.pem"
    openssl pkey -in "$KEY_ROOT/$role-private.pem" -pubout \
      -out "$KEY_ROOT/$role-public.pem"
    chmod 600 "$KEY_ROOT/$role-private.pem"
  done
  "$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null
  "$NORMAL_LOOM" journal-authority-serve --socket "$AUTHORITY_SOCKET" \
    --state-dir "$AUTHORITY_STATE" \
    --private-key "$KEY_ROOT/journal-private.pem" \
    --public-key "$KEY_ROOT/journal-public.pem" \
    --epoch "$AUTHORITY_EPOCH" > "$TEST_ROOT/journal-authority.log" 2>&1 &
  AUTHORITY_PID=$!
  local attempt
  for attempt in $(seq 1 200); do
    [[ -S "$AUTHORITY_SOCKET" ]] && break
    kill -0 "$AUTHORITY_PID" 2>/dev/null || \
      fail "journal authority exited before readiness: $(cat "$TEST_ROOT/journal-authority.log")"
    sleep 0.02
  done
  [[ -S "$AUTHORITY_SOCKET" ]] || fail 'journal authority socket did not become ready'
  "$NORMAL_LOOM" journal-authority-status --socket "$AUTHORITY_SOCKET" >/dev/null
}

initialize
case "${1:-all}" in
  positive) run_positive ;;
  authority-state) run_authority_state ;;
  digest-treatment) run_digest_treatment ;;
  digest-control) run_digest_control ;;
  journal-treatment) run_journal_treatment ;;
  journal-control) run_journal_control ;;
  revocation) run_revocation ;;
  all)
    run_authority_state
    run_positive
    run_digest_treatment
    run_digest_control
    run_journal_treatment
    run_journal_control
    run_revocation
    ;;
  *)
    fail 'usage: sounio_loom_observation_authority_selftest.sh [authority-state|positive|digest-treatment|digest-control|journal-treatment|journal-control|revocation|all]'
    ;;
esac
