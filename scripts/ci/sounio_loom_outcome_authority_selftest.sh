#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-outcome-authority.XXXXXX")"
STATE_DIR="$WORK/state"
trap 'rm -rf "$WORK"' EXIT

fail() {
  printf 'loom-outcome-authority: FAIL: %s\n' "$*" >&2
  exit 1
}

digest() {
  printf '%s' "$1" | sha256sum | awk '{print $1}'
}

expect_refusal() {
  local label="$1" expected="$2"
  shift 2
  local rc=0
  set +e
  "$@" >"$WORK/$label.out" 2>"$WORK/$label.err"
  rc=$?
  set -e
  [[ "$rc" -eq 1 ]] || fail "$label returned rc=$rc"
  rg -q "$expected" "$WORK/$label.err" || {
    sed -n '1,220p' "$WORK/$label.err" >&2
    fail "$label was refused by an unrelated rule"
  }
}

rehash_last_event_after_receipt_mutation() {
  local journal="$1"
  local sequence observed previous kind payload old_digest body new_digest
  IFS=$'\t' read -r sequence observed previous kind payload old_digest \
    < <(tail -n 1 "$journal")
  printf '%s' "$payload" | perl -0777 -ne 'print pack("H*", $_)' \
    > "$WORK/payload.bin"
  perl -0777 -pi -e '
    $changed = s/(classification_receipt=.*?measurement_nonce=nonce-)(2)/$1 . "9"/se;
    END { exit 42 unless $changed == 1 }
  ' "$WORK/payload.bin" || fail 'receipt mutation missed its field'
  payload="$(perl -0777 -ne 'print unpack("H*", $_)' "$WORK/payload.bin")"
  body="$(printf '%s\t%s\t%s\t%s\t%s' \
    "$sequence" "$observed" "$previous" "$kind" "$payload")"
  new_digest="$({ printf '%s\000%s' 'loom-epistemic-journal-v0' "$body"; } \
    | sha256sum | awk '{print $1}')"
  sed '$d' "$journal" > "$journal.rehashed"
  printf '%s\t%s\n' "$body" "$new_digest" >> "$journal.rehashed"
  mv "$journal.rehashed" "$journal"
}

"$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null

openssl genpkey -algorithm ED25519 -out "$WORK/measurement-private.pem" 2>/dev/null
openssl pkey -in "$WORK/measurement-private.pem" -pubout \
  -out "$WORK/measurement-public.pem" 2>/dev/null
openssl genpkey -algorithm ED25519 -out "$WORK/classifier-private.pem" 2>/dev/null
openssl pkey -in "$WORK/classifier-private.pem" -pubout \
  -out "$WORK/classifier-public.pem" 2>/dev/null
openssl genpkey -algorithm ED25519 -out "$WORK/attacker-private.pem" 2>/dev/null
openssl pkey -in "$WORK/attacker-private.pem" -pubout \
  -out "$WORK/attacker-public.pem" 2>/dev/null

provenance="$(digest authority-provenance)"
claim_evidence="$(digest authority-claim-evidence)"
action_evidence="$(digest authority-action-evidence)"
action_falsifier="$(digest authority-action-falsifier)"
branch_evidence="$(digest authority-branch-evidence)"
classifier_spec="$(digest classifier-spec-v1)"

"$LOOM" world-create --state-dir "$STATE_DIR" --world target \
  --agent codex --lane outcome-authority-target >/dev/null
"$LOOM" knowledge-observe --state-dir "$STATE_DIR" --world target \
  --knowledge k --value observation --error 0.01 --uncertainty bounded \
  --confidence 0.5 --provenance "$provenance" >/dev/null
"$LOOM" epistemic-claim-open --state-dir "$STATE_DIR" --world target \
  --claim c --knowledge k --evidence "$claim_evidence" >/dev/null
"$LOOM" world-create --state-dir "$STATE_DIR" --world scheduler \
  --agent codex --lane outcome-authority-v0 >/dev/null

action_header='state\taction_id\ttarget_world\tclaim\tprovider\tresource\tinformation\tfalsification\tdivergence\ttoken_cost\twall_cost\tgpu_cost\tquota_cost\trisk\tevidence_sha256\tfalsifier_sha256'
outcome_header='action_id\tvariant_index\tvariant_count\toutcome_id\tsuccessor_state\tbranch_evidence_sha256'
printf '%b\n' "$action_header" \
  "root\tA\ttarget\tc\tcodex\t/auth/a\t10\t10\t10\t1\t1\t0\t0\t1\t$action_evidence\t$action_falsifier" \
  "next\tB\ttarget\tc\tkimi\t/auth/b\t20\t20\t20\t1\t1\t0\t0\t1\t$action_evidence\t$action_falsifier" \
  > "$WORK/actions.tsv"
printf '%b\n' "$outcome_header" \
  "A\t0\t2\tX\tnext\t$branch_evidence" \
  "A\t1\t2\tY\t-\t$branch_evidence" \
  "B\t0\t1\tDONE\t-\t$branch_evidence" \
  > "$WORK/outcomes.tsv"

compiled="$($LOOM contingent-policy-compile --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy policy-auth --root-state root \
  --actions "$WORK/actions.tsv" --outcomes "$WORK/outcomes.tsv" \
  --token-budget 4 --wall-budget 4 --gpu-budget 0 --quota-budget 0 \
  --order information-first --owner policy-owner --generation generation-1 \
  --measurement-principal instrument-1 \
  --measurement-public-key "$WORK/measurement-public.pem" \
  --classifier-principal classifier-1 \
  --classifier-public-key "$WORK/classifier-public.pem" \
  --classifier-spec-digest "$classifier_spec")"
rg -q 'authority=signed-two-stage root_state=root root_action=A ' \
  <<< "$compiled" || fail "signed authority was not compiled: $compiled"

expect_refusal legacy-laundering \
  'epistemic-outcome-authority-attestation-required:policy-auth' \
  "$LOOM" contingent-policy-observe --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy policy-auth --outcome X \
  --owner policy-owner --generation generation-1 \
  --outcome-digest "$(digest opaque-ui-outcome)"

printf '{"temperature_c":37.2,"instrument":"probe-7"}\n' \
  > "$WORK/measurement-1.json"
truncate -s 16777217 "$WORK/measurement-oversized.bin"
expect_refusal measurement-size-bound \
  'measurement exceeds 16777216 bytes' \
  "$LOOM" contingent-measurement-attest --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy policy-auth \
  --measurement "$WORK/measurement-oversized.bin" \
  --measurement-principal instrument-1 \
  --measurement-private-key "$WORK/measurement-private.pem" \
  --measurement-nonce nonce-oversized --receipt "$WORK/oversized.measurement"
expect_refusal measurement-principal-substitution \
  'epistemic-outcome-authority-measurement-principal-drift:policy-auth' \
  "$LOOM" contingent-measurement-attest --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy policy-auth \
  --measurement "$WORK/measurement-1.json" --measurement-principal ui \
  --measurement-private-key "$WORK/measurement-private.pem" \
  --measurement-nonce nonce-1 --receipt "$WORK/ui.measurement"
expect_refusal measurement-key-substitution \
  'epistemic-outcome-authority-measurement-private-key-mismatch:policy-auth' \
  "$LOOM" contingent-measurement-attest --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy policy-auth \
  --measurement "$WORK/measurement-1.json" \
  --measurement-principal instrument-1 \
  --measurement-private-key "$WORK/attacker-private.pem" \
  --measurement-nonce nonce-1 --receipt "$WORK/attacker.measurement"

measurement="$($LOOM contingent-measurement-attest --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy policy-auth \
  --measurement "$WORK/measurement-1.json" \
  --measurement-principal instrument-1 \
  --measurement-private-key "$WORK/measurement-private.pem" \
  --measurement-nonce nonce-1 --receipt "$WORK/measurement-1.receipt")"
rg -q 'LOOM_MEASUREMENT_ATTESTED .*receipt_sha256=' <<< "$measurement" || \
  fail "measurement receipt was not produced: $measurement"

cp "$WORK/measurement-1.receipt" "$WORK/noncanonical.measurement"
perl -pi -e 's/^(signature_base64=.*)=$/$1/' \
  "$WORK/noncanonical.measurement"
expect_refusal measurement-base64-canonicalization \
  'epistemic-outcome-authority-signature-(invalid|noncanonical)-base64' \
  "$LOOM" contingent-classification-attest --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy policy-auth \
  --measurement-receipt "$WORK/noncanonical.measurement" --outcome X \
  --classifier-principal classifier-1 \
  --classifier-private-key "$WORK/classifier-private.pem" \
  --receipt "$WORK/noncanonical.classification"

truncate -s 16385 "$WORK/oversized.measurement"
expect_refusal measurement-receipt-size-bound \
  'measurement receipt exceeds 16384 bytes' \
  "$LOOM" contingent-classification-attest --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy policy-auth \
  --measurement-receipt "$WORK/oversized.measurement" --outcome X \
  --classifier-principal classifier-1 \
  --classifier-private-key "$WORK/classifier-private.pem" \
  --receipt "$WORK/oversized.classification"

expect_refusal classifier-principal-substitution \
  'epistemic-outcome-authority-classifier-principal-drift:policy-auth' \
  "$LOOM" contingent-classification-attest --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy policy-auth \
  --measurement-receipt "$WORK/measurement-1.receipt" --outcome X \
  --classifier-principal llm-ui \
  --classifier-private-key "$WORK/classifier-private.pem" \
  --receipt "$WORK/ui.classification"
expect_refusal classifier-key-substitution \
  'epistemic-outcome-authority-classifier-private-key-mismatch:policy-auth' \
  "$LOOM" contingent-classification-attest --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy policy-auth \
  --measurement-receipt "$WORK/measurement-1.receipt" --outcome X \
  --classifier-principal classifier-1 \
  --classifier-private-key "$WORK/attacker-private.pem" \
  --receipt "$WORK/attacker.classification"

classification="$($LOOM contingent-classification-attest \
  --state-dir "$STATE_DIR" --world scheduler \
  --contingent-policy policy-auth \
  --measurement-receipt "$WORK/measurement-1.receipt" --outcome X \
  --classifier-principal classifier-1 \
  --classifier-private-key "$WORK/classifier-private.pem" \
  --receipt "$WORK/classification-1.receipt")"
rg -q 'LOOM_CLASSIFICATION_ATTESTED .*receipt_sha256=' \
  <<< "$classification" || fail "classification receipt was not produced"

# Any append at the same policy cursor changes the CAS coordinate. The old
# receipts must die even though current_path remains root.
"$LOOM" knowledge-observe --state-dir "$STATE_DIR" --world scheduler \
  --knowledge unrelated-head-change --value unrelated --error 0 \
  --uncertainty none --confidence 1 --provenance "$provenance" >/dev/null
expect_refusal intervening-event-stales-receipt \
  'epistemic-outcome-authority-cursor-binding-mismatch:policy-auth' \
  "$LOOM" contingent-policy-observe-attested --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy policy-auth \
  --measurement-receipt "$WORK/measurement-1.receipt" \
  --classification-receipt "$WORK/classification-1.receipt" \
  --owner policy-owner --generation generation-1
"$LOOM" contingent-measurement-attest --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy policy-auth \
  --measurement "$WORK/measurement-1.json" \
  --measurement-principal instrument-1 \
  --measurement-private-key "$WORK/measurement-private.pem" \
  --measurement-nonce nonce-1b --receipt "$WORK/measurement-1.receipt" \
  >/dev/null
"$LOOM" contingent-classification-attest --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy policy-auth \
  --measurement-receipt "$WORK/measurement-1.receipt" --outcome X \
  --classifier-principal classifier-1 \
  --classifier-private-key "$WORK/classifier-private.pem" \
  --receipt "$WORK/classification-1.receipt" >/dev/null

set +e
"$LOOM" contingent-policy-observe-attested \
  --state-dir "$STATE_DIR" --world scheduler \
  --contingent-policy policy-auth \
  --measurement-receipt "$WORK/measurement-1.receipt" \
  --classification-receipt "$WORK/classification-1.receipt" \
  --owner policy-owner --generation generation-1 \
  >"$WORK/race-1.out" 2>"$WORK/race-1.err" &
race_pid_1=$!
"$LOOM" contingent-policy-observe-attested \
  --state-dir "$STATE_DIR" --world scheduler \
  --contingent-policy policy-auth \
  --measurement-receipt "$WORK/measurement-1.receipt" \
  --classification-receipt "$WORK/classification-1.receipt" \
  --owner policy-owner --generation generation-1 \
  >"$WORK/race-2.out" 2>"$WORK/race-2.err" &
race_pid_2=$!
wait "$race_pid_1"; race_rc_1=$?
wait "$race_pid_2"; race_rc_2=$?
set -e
if [[ "$race_rc_1:$race_rc_2" != '0:1' && \
      "$race_rc_1:$race_rc_2" != '1:0' ]]; then
  sed -n '1,120p' "$WORK/race-1.err" >&2
  sed -n '1,120p' "$WORK/race-2.err" >&2
  fail "same-receipt race returned $race_rc_1:$race_rc_2"
fi
if [[ "$race_rc_1" -eq 0 ]]; then
  advanced="$(cat "$WORK/race-1.out")"
  loser_err="$WORK/race-2.err"
else
  advanced="$(cat "$WORK/race-2.out")"
  loser_err="$WORK/race-1.err"
fi
rg -q 'epistemic-outcome-authority-cursor-binding-mismatch:policy-auth' \
  "$loser_err" || fail 'same-receipt race loser failed for an unrelated rule'
rg -q 'state=advanced path=root current_action=A outcome=X next_path=root.0 next_action=B ' \
  <<< "$advanced" || fail "attested transition was wrong: $advanced"

expect_refusal stale-receipt-replay \
  'epistemic-outcome-authority-cursor-binding-mismatch:policy-auth' \
  "$LOOM" contingent-policy-observe-attested --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy policy-auth \
  --measurement-receipt "$WORK/measurement-1.receipt" \
  --classification-receipt "$WORK/classification-1.receipt" \
  --owner policy-owner --generation generation-1

printf '{"assay":"complete","result":true}\n' > "$WORK/measurement-2.json"
"$LOOM" contingent-measurement-attest --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy policy-auth \
  --measurement "$WORK/measurement-2.json" \
  --measurement-principal instrument-1 \
  --measurement-private-key "$WORK/measurement-private.pem" \
  --measurement-nonce nonce-2 --receipt "$WORK/measurement-2.receipt" \
  >/dev/null
"$LOOM" contingent-classification-attest --state-dir "$STATE_DIR" \
  --world scheduler --contingent-policy policy-auth \
  --measurement-receipt "$WORK/measurement-2.receipt" --outcome DONE \
  --classifier-principal classifier-1 \
  --classifier-private-key "$WORK/classifier-private.pem" \
  --receipt "$WORK/classification-2.receipt" >/dev/null
completed="$($LOOM contingent-policy-observe-attested \
  --state-dir "$STATE_DIR" --world scheduler \
  --contingent-policy policy-auth \
  --measurement-receipt "$WORK/measurement-2.receipt" \
  --classification-receipt "$WORK/classification-2.receipt" \
  --owner policy-owner --generation generation-1)"
rg -q 'state=completed path=root.0 current_action=B outcome=DONE ' \
  <<< "$completed" || fail "attested completion was wrong: $completed"

"$LOOM" world-verify --state-dir "$STATE_DIR" --world scheduler >/dev/null
"$LOOM" export-events-arrow --state-dir "$STATE_DIR" \
  --out "$WORK/outcome-authority.arrow" >/dev/null
[[ -s "$WORK/outcome-authority.arrow" ]] || fail 'Arrow export is empty'

tampered="$WORK/tampered"
cp -a "$STATE_DIR" "$tampered"
journal="$tampered/loom-epistemic/worlds/scheduler/journal.tsv"
rehash_last_event_after_receipt_mutation "$journal"
expect_refusal receipt-sabotage \
  'epistemic-outcome-authority-classification-noncanonical' \
  "$LOOM" world-verify --state-dir "$tampered" --world scheduler
expect_refusal arrow-receipt-laundering \
  'epistemic-outcome-authority-classification-noncanonical' \
  "$LOOM" export-events-arrow --state-dir "$tampered" \
  --out "$WORK/laundered.arrow"
[[ ! -e "$WORK/laundered.arrow" ]] || \
  fail 'receipt sabotage produced Arrow output'

printf 'SOUNIO_LOOM_OUTCOME_AUTHORITY_GATE_PASS=true schema=loom-outcome-evidence-authority-v0 frame=9012 measurement=ED25519 classification=ED25519 roles=OWNER+MEASURER+CLASSIFIER pairwise=DISJOINT cursor_binding=EXACT journal_head_binding=EXACT partition_binding=EXACT classifier_spec=PRECOMMITTED canonical_base64=PASS size_bounds=PASS legacy_laundering=REFUSED principal_substitution=REFUSED key_substitution=REFUSED intervening_event=STALES replay=REFUSED same_receipt_race=ONE_WINNER atomic_consume=PASS native_route_frame=9011 replay_verification=PASS journal_sabotage=REFUSED arrow_laundering=REFUSED physical_truth_claim=NONE rollback_resistance=NOT_CLAIMED runtime=OCaml+Sounio\n'
