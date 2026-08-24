#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
CANARY="$ROOT_DIR/scripts/ci/sounio_loom_pod_replay_canary.sh"
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
ADAPTER="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-continuity-runtime"
MODULE="$ROOT_DIR/stdlib/coordination/loom_continuity.sio"
BUILD_ADAPTER="$ROOT_DIR/scripts/dev/build_sounio_loom_continuity_adapter.sh"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-journal-quorum.XXXXXX")"
STATE_DIR="$TEST_ROOT/loom"
EPOCH=1
PANE_ID='loom-pod-replay:terminal'
BEAGLE_LANE='pane-6c6f6f6d2d706f642d7265706c61793a7465726d696e616c'
AUTHORITY_PIDS=()
LOOM_ACTIVE=0

fail() {
  printf 'sounio-loom-journal-quorum-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

cleanup() {
  [[ ! -d "$TEST_ROOT/e2e" ]] || kill_generation "$TEST_ROOT/e2e" || true
  if [[ "$LOOM_ACTIVE" == 1 ]]; then
    env "${AUTHORITY_ENV[@]}" "$LOOM" stop --state-dir "$STATE_DIR" \
      --cwd "$ROOT_DIR" --agent quorum-test --lane journal >/dev/null 2>&1 || true
  fi
  local pid
  for pid in "${AUTHORITY_PIDS[@]}"; do
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] && kill "$pid" 2>/dev/null || true
  done
  for pid in "${AUTHORITY_PIDS[@]}"; do
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] && wait "$pid" 2>/dev/null || true
  done
  if [[ "${SOUNIO_LOOM_TEST_KEEP:-0}" == 1 ]]; then
    printf 'sounio-loom-journal-quorum-selftest: kept %s\n' "$TEST_ROOT" >&2
    return
  fi
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

field() {
  local name="$1" value="$2"
  sed -n "s/.* ${name}=\([^ ]*\).*/\1/p" <<< "$value"
}

json_string_value() {
  local path="$1" key="$2"
  sed -n "s/.*\"${key}\":\"\([^\"]*\)\".*/\1/p" "$path" | head -1
}

kv_value() {
  local path="$1" key="$2"
  sed -n "s/^${key}=//p" "$path" | head -1
}

kill_generation() {
  local root="$1" status pid bridge_file bridge_pid
  status="$(
    env "${AUTHORITY_ENV[@]}" "$LOOM" status \
      --state-dir "$root/loom" --cwd "$ROOT_DIR" \
      --agent beagle-workbench --lane "$BEAGLE_LANE" 2>/dev/null || true
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

receipt_path() {
  local root="$1" matches=()
  mapfile -t matches < <(find "$root/loom" \
    -name sounio-continuity.receipt -type f -print)
  [[ "${#matches[@]}" -eq 1 ]] || \
    fail "$root contains ${#matches[@]} predecessor receipts instead of one"
  printf '%s\n' "${matches[0]}"
}

generation_count() {
  find "$1/loom" -name sounio-continuity.receipt -type f -print \
    2>/dev/null | wc -l
}

mkdir -p "$TEST_ROOT/keys"
for index in 1 2 3; do
  openssl genpkey -algorithm Ed25519 \
    -out "$TEST_ROOT/keys/authority-$index-private.pem" >/dev/null 2>&1
  openssl pkey -in "$TEST_ROOT/keys/authority-$index-private.pem" -pubout \
    -out "$TEST_ROOT/keys/authority-$index-public.pem" >/dev/null 2>&1
  chmod 0600 "$TEST_ROOT/keys/authority-$index-private.pem"
  "$LOOM" journal-authority-serve \
    --socket "$TEST_ROOT/authority-$index.sock" \
    --state-dir "$TEST_ROOT/authority-$index-state" \
    --private-key "$TEST_ROOT/keys/authority-$index-private.pem" \
    --public-key "$TEST_ROOT/keys/authority-$index-public.pem" \
    --epoch "$EPOCH" >"$TEST_ROOT/authority-$index.log" 2>&1 &
  AUTHORITY_PIDS+=("$!")
done

for role in signer observer; do
  openssl genpkey -algorithm Ed25519 \
    -out "$TEST_ROOT/keys/$role-private.pem" >/dev/null 2>&1
  openssl pkey -in "$TEST_ROOT/keys/$role-private.pem" -pubout \
    -out "$TEST_ROOT/keys/$role-public.pem" >/dev/null 2>&1
  chmod 0600 "$TEST_ROOT/keys/$role-private.pem"
done

for index in 1 2 3; do
  for _ in $(seq 1 100); do
    [[ -S "$TEST_ROOT/authority-$index.sock" ]] && break
    kill -0 "${AUTHORITY_PIDS[$((index - 1))]}" 2>/dev/null || \
      fail "authority $index exited before readiness"
    sleep 0.05
  done
  [[ -S "$TEST_ROOT/authority-$index.sock" ]] || \
    fail "authority $index did not create its socket"
  "$LOOM" journal-authority-status \
    --socket "$TEST_ROOT/authority-$index.sock" >/dev/null
done

AUTHORITY_ENV=(
  SOUNIO_LOOM_REQUIRE_JOURNAL_AUTHORITY=1
  SOUNIO_LOOM_JOURNAL_AUTHORITY_QUORUM=2
  SOUNIO_LOOM_JOURNAL_AUTHORITY_EPOCH="$EPOCH"
  SOUNIO_LOOM_JOURNAL_AUTHORITY_1_SOCKET="$TEST_ROOT/authority-1.sock"
  SOUNIO_LOOM_JOURNAL_AUTHORITY_1_VERIFY_KEY="$TEST_ROOT/keys/authority-1-public.pem"
  SOUNIO_LOOM_JOURNAL_AUTHORITY_2_SOCKET="$TEST_ROOT/authority-2.sock"
  SOUNIO_LOOM_JOURNAL_AUTHORITY_2_VERIFY_KEY="$TEST_ROOT/keys/authority-2-public.pem"
  SOUNIO_LOOM_JOURNAL_AUTHORITY_3_SOCKET="$TEST_ROOT/authority-3.sock"
  SOUNIO_LOOM_JOURNAL_AUTHORITY_3_VERIFY_KEY="$TEST_ROOT/keys/authority-3-public.pem"
)

run_canary_phase() {
  local root="$1" uid="$2" phase="$3"
  env "${AUTHORITY_ENV[@]}" \
    POD_UID="$uid" \
    POD_NAME=loom-journal-quorum-0 \
    SOUNIO_CANARY_SOURCE_ROOT="$ROOT_DIR" \
    SOUNIO_LOOM_POD_CANARY_ROOT="$root" \
    SOUNIO_LOOM_BINARY="$LOOM" \
    SOUNIO_LOOM_REQUIRE_SIGNED_RECEIPTS=1 \
    SOUNIO_LOOM_REQUIRE_OBSERVATION_AUTHORITY=1 \
    SOUNIO_LOOM_SIGNING_KEY="$TEST_ROOT/keys/signer-private.pem" \
    SOUNIO_LOOM_VERIFY_KEY="$TEST_ROOT/keys/signer-public.pem" \
    SOUNIO_LOOM_OBSERVER_VERIFY_KEY="$TEST_ROOT/keys/observer-public.pem" \
    SOUNIO_LOOM_CONTINUITY_ADAPTER="$ADAPTER" \
    bash "$CANARY" "$phase"
}

measure_generation() {
  local root="$1" uid="$2" receipt generation output generation_dir
  local semantic_journal guardian_journal descriptor
  local semantic_digest guardian_digest descriptor_digest principal_set
  local required epoch expected_checkpoint
  receipt="$(receipt_path "$root")"
  generation="$(json_string_value "$root/spawn-$uid.json" loomInstanceId)"
  [[ -n "$generation" ]] || fail 'quorum genesis omitted its generation identity'
  output="$(dirname "$receipt")/sounio-continuity.observer-attestation"
  env "${AUTHORITY_ENV[@]}" \
    SOUNIO_LOOM_REQUIRE_OBSERVATION_AUTHORITY=1 \
    "$LOOM" measure-continuity-generation \
      --state-dir "$root/loom" --pane-id "$PANE_ID" \
      --generation "$generation" --receipt "$receipt" \
      --subject-public-key "$TEST_ROOT/keys/signer-public.pem" \
      --observer-private-key "$TEST_ROOT/keys/observer-private.pem" \
      --observer-public-key "$TEST_ROOT/keys/observer-public.pem" \
      --out "$output" --adapter "$ADAPTER" >/dev/null
  [[ -s "$output" ]] || fail 'quorum authority measurement was not created'
  grep -q '^schema=loom-independent-measurement-attestation-v3$' "$output" || \
    fail 'quorum authority measurement did not use attestation v3'
  grep -q '^journal_authority_required_quorum=2$' "$output" || \
    fail 'quorum authority measurement omitted its threshold'
  generation_dir="$(dirname "$(find "$root/loom" \
    -path "*/generations/$generation/journal.tsv" -type f -print -quit)")"
  semantic_journal="$generation_dir/journal.tsv"
  guardian_journal="$generation_dir/guardian.tsv"
  descriptor="$generation_dir/session.state"
  [[ -f "$semantic_journal" && -f "$guardian_journal" && -f "$descriptor" ]] || \
    fail 'could not locate the measured generation artifacts'
  semantic_digest="$(sha256sum "$semantic_journal" | awk '{ print $1 }')"
  guardian_digest="$(sha256sum "$guardian_journal" | awk '{ print $1 }')"
  descriptor_digest="$(sha256sum "$descriptor" | awk '{ print $1 }')"
  [[ "$(kv_value "$output" semantic_journal_sha256)" == "$semantic_digest" ]] || \
    fail 'measurement does not bind the literal semantic quorum certificates'
  [[ "$(kv_value "$output" guardian_journal_sha256)" == "$guardian_digest" ]] || \
    fail 'measurement does not bind the literal Guardian quorum certificates'
  principal_set="$(kv_value "$output" journal_authority_principal_id)"
  required="$(kv_value "$output" journal_authority_required_quorum)"
  epoch="$(kv_value "$output" journal_authority_epoch)"
  expected_checkpoint="$({
    printf '%s\0' 'loom-journal-authority-quorum-checkpoint-v1' "$principal_set" \
      "$epoch" "$required" "$semantic_digest" "$guardian_digest"
    printf '%s' "$descriptor_digest"
  } | sha256sum | awk '{ print $1 }')"
  [[ "$(kv_value "$output" journal_authority_checkpoint_sha256)" == \
    "$expected_checkpoint" ]] || \
    fail 'checkpoint does not commit the exact quorum-bearing journal bytes'
}

# Exercise the complete predecessor -> independent measurement -> total process
# loss -> pre-spawn typed admission -> successor path.
e2e_root="$TEST_ROOT/e2e"
phase="$(run_canary_phase "$e2e_root" quorum-one phase-one)"
[[ "$phase" == *CANARY_PHASE_ONE* ]] || fail 'quorum genesis failed'
kill_generation "$e2e_root"
measure_generation "$e2e_root" quorum-one
phase="$(run_canary_phase "$e2e_root" quorum-two phase-two)"
[[ "$phase" == *CANARY_PHASE_TWO* ]] || fail 'quorum successor failed'
e2e_spawn="$e2e_root/spawn-quorum-two.json"
grep -q '"sounioPolicyJournalAuthorityQuorumVerified":true' "$e2e_spawn" || \
  fail 'quorum successor omitted its native Sounio quorum status'
grep -q '"sounioPolicyJournalAuthorityRequiredQuorum":2' "$e2e_spawn" || \
  fail 'quorum successor omitted the required threshold'
[[ "$(generation_count "$e2e_root")" -eq 2 ]] || \
  fail 'quorum path did not create exactly one successor'
kill_generation "$e2e_root"

env "${AUTHORITY_ENV[@]}" "$LOOM" start --state-dir "$STATE_DIR" \
  --agent quorum-test --lane journal --session-id quorum-test \
  --cwd "$ROOT_DIR" -- /bin/sh -c 'printf "QUORUM_READY\n"; sleep 30' >/dev/null
LOOM_ACTIVE=1

status=''
for _ in $(seq 1 100); do
  status="$(env "${AUTHORITY_ENV[@]}" "$LOOM" status \
    --state-dir "$STATE_DIR" --cwd "$ROOT_DIR" \
    --agent quorum-test --lane journal 2>/dev/null || true)"
  [[ "$status" == *'state=active'* ]] && break
  sleep 0.05
done
[[ "$status" == *'state=active'* ]] || fail "Loom did not become active: $status"

# One authority can disappear without surrendering journal progress.
kill "${AUTHORITY_PIDS[2]}"
wait "${AUTHORITY_PIDS[2]}" 2>/dev/null || true
AUTHORITY_PIDS[2]=0
env "${AUTHORITY_ENV[@]}" "$LOOM" stop --state-dir "$STATE_DIR" \
  --cwd "$ROOT_DIR" --agent quorum-test --lane journal >/dev/null
LOOM_ACTIVE=0

mapfile -t journals < <(find "$STATE_DIR" \
  \( -name journal.tsv -o -name guardian.tsv \) -type f -print | sort)
[[ "${#journals[@]}" -eq 2 ]] || \
  fail "expected two retained journals, found ${#journals[@]}"

for journal in "${journals[@]}"; do
  verify_command=verify-journal
  [[ "$(basename "$journal")" == guardian.tsv ]] && \
    verify_command=verify-guardian-journal
  env "${AUTHORITY_ENV[@]}" "$LOOM" "$verify_command" \
    --journal "$journal" >/dev/null
  awk -F '\t' '
    NF != 16 || $9 != "quorum-v1" || $10 != 2 { exit 41 }
    { valid = ($12 != "-") + ($14 != "-") + ($16 != "-") }
    valid < 2 { exit 42 }
    END { if (NR == 0) exit 43 }
  ' "$journal" || fail "journal omitted a valid 2-of-3 certificate: $journal"
done

semantic_journal="$(find "$STATE_DIR" -name journal.tsv -type f -print -quit)"
generation_name="$(basename "$(dirname "$semantic_journal")")"
session_name="$(basename "$(dirname "$(dirname "$(dirname "$semantic_journal")")")")"
forged_journal="$TEST_ROOT/forged/$session_name/generations/$generation_name/journal.tsv"
mkdir -p "$(dirname "$forged_journal")"
awk -F '\t' 'BEGIN { OFS="\t" } { $14="-"; $16="-"; print }' \
  "$semantic_journal" > "$forged_journal"

set +e
runtime_treatment="$(env "${AUTHORITY_ENV[@]}" "$LOOM" verify-journal \
  --journal "$forged_journal" 2>&1)"
runtime_treatment_rc=$?
set -e
[[ "$runtime_treatment_rc" -ne 0 && \
  "$runtime_treatment" == *'quorum-unsatisfied'* ]] || \
  fail "single-share journal was not refused: rc=$runtime_treatment_rc output=$runtime_treatment"

# The adapter witness holds every other fact fixed. Only the minimum verified
# share count changes from two to one.
digest_frame='1 2 3 4 5 6 7 8 11 12 13 14 15 16 17 18 21 22 23 24 25 26 27 28 31 32 33 34 35 36 37 38'
frame_prefix='9006 101 102 103 104 105 106 2'
frame_suffix='107 108 1 201 202 203 204 201 202 203 204'
positive_frame="$frame_prefix 2 $frame_suffix $digest_frame $digest_frame"
treatment_frame="$frame_prefix 1 $frame_suffix $digest_frame $digest_frame"

positive="$(printf '%s\n' "$positive_frame" | "$ADAPTER")"
[[ "$positive" == *'loom-native-pre-spawn-v4'* ]] || \
  fail "2-of-3 positive control was not admitted: $positive"

set +e
treatment="$(printf '%s\n' "$treatment_frame" | "$ADAPTER" 2>&1)"
treatment_rc=$?
set -e
[[ "$treatment_rc" -eq 42 && \
  "$treatment" == 'SOUNIO_CONTINUITY_REFUSE reason=journal-quorum-admission' ]] || \
  fail "single-share typed treatment was not refused: rc=$treatment_rc output=$treatment"

mutant_module="$TEST_ROOT/loom_continuity_quorum_sabotage.sio"
awk '
  /^fn journal_quorum_is_satisfied\(/ {
    inside=1
    print "fn journal_quorum_is_satisfied("
    print "    required_quorum: i64,"
    print "    min_valid_signatures: i64,"
    print ") -> bool {"
    print "    true"
    print "}"
    next
  }
  inside && /^}$/ { inside=0; next }
  inside { next }
  { print }
' "$MODULE" > "$mutant_module"
[[ "$(grep -c '^fn journal_quorum_is_satisfied(' "$mutant_module")" -eq 1 ]] || \
  fail 'could not create the targeted Sounio quorum sabotage'

mutant_adapter="$TEST_ROOT/sounio-loom-continuity-quorum-sabotage"
SOUNIO_LOOM_CONTINUITY_MODULE="$mutant_module" \
SOUNIO_LOOM_CONTINUITY_OUTPUT="$mutant_adapter" \
  bash "$BUILD_ADAPTER" >/dev/null
control="$(printf '%s\n' "$treatment_frame" | "$mutant_adapter")"
[[ "$control" == *'loom-native-pre-spawn-v4'* ]] || \
  fail "removing only journal_quorum_is_satisfied did not admit the same witness: $control"

# Reusing one principal in two configured slots fails before any certificate
# can be interpreted as independent custody.
collapsed_env=("${AUTHORITY_ENV[@]}")
collapsed_env[8]="SOUNIO_LOOM_JOURNAL_AUTHORITY_3_VERIFY_KEY=$TEST_ROOT/keys/authority-2-public.pem"
set +e
collapsed="$(env "${collapsed_env[@]}" "$LOOM" verify-journal \
  --journal "$semantic_journal" 2>&1)"
collapsed_rc=$?
set -e
[[ "$collapsed_rc" -ne 0 && "$collapsed" == *'principals-not-disjoint'* ]] || \
  fail "collapsed quorum principals were not refused: rc=$collapsed_rc output=$collapsed"

printf '%s\n' \
  'SOUNIO_LOOM_JOURNAL_QUORUM_POSITIVE=PASS successor_created=1 native_frame=9006 measurement=attestation-v3 certificate_binding=journal-tsv-sha256 required=2 min_valid=2 one_member_loss=survived'
printf '%s\n' \
  'SOUNIO_LOOM_JOURNAL_QUORUM_TREATMENT=PASS unchanged_single_share=refused_before_spawn runtime_replay=refused successor_created=0'
printf '%s\n' \
  'SOUNIO_LOOM_JOURNAL_QUORUM_CONTROL=PASS intervention=journal_quorum_is_satisfied-always-true unchanged_single_share=admitted'
printf '%s\n' \
  'sounio-loom-journal-quorum-selftest: PASS runtime=2-of-3 successor=admitted-via-9006 measurement=attestation-v3 certificate-binding=journal-tsv-sha256 one-member-loss=survived single-share=refused pre-spawn=refused sabotage=admitted collapsed-principal=refused'
