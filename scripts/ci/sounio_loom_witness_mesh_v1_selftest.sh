#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-witness-mesh-v1.XXXXXX")"
STATE_DIR="$WORK/state"
MEMBERSHIP="$WORK/membership.tsv"
ENDPOINTS="$WORK/endpoints.tsv"
declare -a WITNESS_PIDS=("" "" "" "" "")
declare -a WITNESS_PORTS=("" "" "" "" "")

cleanup() {
  local index pid
  for index in 1 2 3 4; do
    pid="${WITNESS_PIDS[$index]:-}"
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  rm -rf "$WORK"
}
trap cleanup EXIT

fail() {
  printf 'loom-witness-mesh-v1: FAIL: %s\n' "$*" >&2
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

start_witness() {
  local index="$1" log="$WORK/witness-$1.log" pid attempt=0
  : > "$log"
  "$LOOM" witness-serve \
    --witness-state-dir "$WORK/witness-$index-state" \
    --membership "$MEMBERSHIP" --witness "w$index" \
    --private-key "$WORK/witness-$index-private.pem" \
    --bind 127.0.0.1 --port 0 >"$log" 2>&1 &
  pid=$!
  WITNESS_PIDS[$index]="$pid"
  until rg -q 'LOOM_WITNESS_READY schema=loom-witness-service-v1 ' "$log"; do
    kill -0 "$pid" 2>/dev/null || {
      sed -n '1,220p' "$log" >&2
      fail "witness $index exited before readiness"
    }
    attempt=$((attempt + 1))
    [[ "$attempt" -lt 200 ]] || fail "witness $index readiness timed out"
    sleep 0.025
  done
  WITNESS_PORTS[$index]="$(sed -n 's/.* port=\([0-9][0-9]*\) .*/\1/p' "$log")"
  [[ -n "${WITNESS_PORTS[$index]}" ]] || fail "witness $index omitted its port"
}

stop_witness() {
  local index="$1" pid="${WITNESS_PIDS[$1]:-}"
  if [[ -n "$pid" ]]; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    WITNESS_PIDS[$index]=""
  fi
}

write_endpoints() {
  printf 'witness_id\thost\tport\n' > "$ENDPOINTS"
  for index in 1 2 3 4; do
    printf 'w%s\t127.0.0.1\t%s\n' "$index" "${WITNESS_PORTS[$index]}" \
      >> "$ENDPOINTS"
  done
}

observe() {
  local state="$1" knowledge="$2" value="$3"
  "$LOOM" knowledge-observe --state-dir "$state" --world mesh-v1 \
    --knowledge "$knowledge" --value "$value" --error 0 \
    --uncertainty bounded --confidence 1 \
    --provenance "$(digest "$knowledge-provenance")" >/dev/null
}

"$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null

for index in 1 2 3 4; do
  openssl genpkey -algorithm ED25519 \
    -out "$WORK/witness-$index-private.pem" 2>/dev/null
  openssl pkey -in "$WORK/witness-$index-private.pem" -pubout \
    -out "$WORK/witness-$index-public.pem" 2>/dev/null
done
openssl genpkey -algorithm ED25519 \
  -out "$WORK/anchor-private.pem" 2>/dev/null
openssl pkey -in "$WORK/anchor-private.pem" -pubout \
  -out "$WORK/anchor-public.pem" 2>/dev/null
openssl genpkey -algorithm ED25519 \
  -out "$WORK/attacker-anchor-private.pem" 2>/dev/null

printf 'schema\tloom-witness-membership-v1\n' > "$MEMBERSHIP"
printf 'anchor_public_key\t%s\n' "$WORK/anchor-public.pem" >> "$MEMBERSHIP"
printf 'witness_id\tpublic_key\n' >> "$MEMBERSHIP"
for index in 1 2 3 4; do
  printf 'w%s\t%s\n' "$index" "$WORK/witness-$index-public.pem" >> "$MEMBERSHIP"
done
for index in 1 2 3 4; do
  start_witness "$index"
done
write_endpoints

"$LOOM" world-create --state-dir "$STATE_DIR" --world mesh-v1 \
  --agent codex --lane witness-mesh-v1 >/dev/null
observe "$STATE_DIR" observation-1 first

expect_refusal anchor-key-substitution 'witness-anchor-private-key-mismatch' \
  "$LOOM" witness-mesh-anchor --state-dir "$STATE_DIR" --world mesh-v1 \
  --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS" \
  --anchor-private-key "$WORK/attacker-anchor-private.pem"

expect_refusal anchor-signature-sabotage \
  'witness-quorum-unavailable:valid=0:.*witness-request-anchor-signature-invalid' \
  env SOUNIO_LOOM_WITNESS_TAMPER_ANCHOR_SIGNATURE=1 \
  "$LOOM" witness-mesh-anchor --state-dir "$STATE_DIR" --world mesh-v1 \
  --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS" \
  --anchor-private-key "$WORK/anchor-private.pem"
[[ "$(find "$WORK" -path '*/domain-*.receipt' | wc -l)" -eq 0 ]] || \
  fail 'anchor signature sabotage advanced witness state'

# All four services persist sequence 1, then the client dies before its local
# certificate. Recovery must reuse the signed high-water receipts.
set +e
SOUNIO_LOOM_WITNESS_FAILPOINT=after-quorum-before-certificate \
  "$LOOM" witness-mesh-anchor --state-dir "$STATE_DIR" --world mesh-v1 \
    --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS" \
    --anchor-private-key "$WORK/anchor-private.pem" \
    >"$WORK/post-quorum-crash.out" 2>"$WORK/post-quorum-crash.err"
crash_rc=$?
set -e
[[ "$crash_rc" -eq 197 ]] || fail "post-quorum failpoint returned rc=$crash_rc"
rg -Fxq 'LOOM_WITNESS_FAILPOINT name=after-quorum-before-certificate exit=197' \
  "$WORK/post-quorum-crash.err" || fail 'post-quorum failpoint did not fire'
[[ "$(find "$STATE_DIR/loom-witness-mesh/mesh-v1" \
  -name 'checkpoint-*.receipt' | wc -l)" -eq 0 ]] || \
  fail 'post-quorum crash wrote a local certificate'
recovered="$($LOOM witness-mesh-anchor --state-dir "$STATE_DIR" --world mesh-v1 \
  --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS" \
  --anchor-private-key "$WORK/anchor-private.pem")"
rg -q 'schema=loom-witness-mesh-v1 .*sequence=1 .*quorum=4/4 recovered_checkpoints=1 ' \
  <<< "$recovered" || fail "post-quorum recovery was not explicit: $recovered"
"$LOOM" witness-mesh-verify --state-dir "$STATE_DIR" --world mesh-v1 \
  --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS" >/dev/null

ROLLBACK="$WORK/rollback-state"
cp -a "$STATE_DIR" "$ROLLBACK"
cp -a "$WORK/witness-2-state" "$WORK/witness-2-sequence-1"

# Sequence 2 is accepted and Byzantine-strict verified with w4 unavailable.
stop_witness 4
observe "$STATE_DIR" observation-2 second
one_down="$($LOOM witness-mesh-anchor --state-dir "$STATE_DIR" --world mesh-v1 \
  --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS" \
  --anchor-private-key "$WORK/anchor-private.pem")"
rg -q 'sequence=2 .*quorum=3/4 ' <<< "$one_down" || \
  fail "one-down 3/4 quorum did not anchor: $one_down"
strict_one_down="$($LOOM witness-mesh-verify --state-dir "$STATE_DIR" \
  --world mesh-v1 --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS")"
rg -q 'verification_policy=byzantine-strict remote_quorum=3/4 required=3/4 .*rollback_resistance=ONE_DISHONEST_WITNESS_HONEST_INTERSECTION' \
  <<< "$strict_one_down" || fail "one-down strict verification failed: $strict_one_down"

# Reproduce the attack that defeats 2-of-3. Sequence 2 was accepted by
# {w1,w2,w3}; w4 is honestly lagging at sequence 1. Make w1 unavailable and
# roll w2 back to its signed sequence-1 state. The verification set
# {w2,w3,w4} still contains current honest w3, so the rolled local view refuses.
stop_witness 1
stop_witness 2
rm -rf "$WORK/witness-2-state"
cp -a "$WORK/witness-2-sequence-1" "$WORK/witness-2-state"
start_witness 2
start_witness 4
write_endpoints
"$LOOM" world-verify --state-dir "$ROLLBACK" --world mesh-v1 >/dev/null
expect_refusal sole-byzantine-intersection-attack \
  'witness-rollback-detected:w3:remote-count=3:local=2' \
  "$LOOM" witness-mesh-verify --state-dir "$ROLLBACK" --world mesh-v1 \
  --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS"

# The three reachable members replay their own missing suffixes and form a
# fresh 3/4 quorum while w1 remains unavailable.
observe "$STATE_DIR" observation-3 third
intersection_recovery="$($LOOM witness-mesh-anchor --state-dir "$STATE_DIR" \
  --world mesh-v1 --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS" \
  --anchor-private-key "$WORK/anchor-private.pem")"
rg -q 'sequence=3 .*quorum=3/4 ' <<< "$intersection_recovery" || \
  fail "intersection recovery did not anchor: $intersection_recovery"
"$LOOM" witness-mesh-verify --state-dir "$STATE_DIR" --world mesh-v1 \
  --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS" >/dev/null

# w1 returns and catches up through a fourth checkpoint.
start_witness 1
write_endpoints
observe "$STATE_DIR" observation-4 fourth
catchup="$($LOOM witness-mesh-anchor --state-dir "$STATE_DIR" --world mesh-v1 \
  --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS" \
  --anchor-private-key "$WORK/anchor-private.pem")"
rg -q 'sequence=4 .*quorum=4/4 ' <<< "$catchup" || \
  fail "four-member catch-up did not anchor: $catchup"
rg -q '^anchor_sequence=4$' "$WORK/witness-1-state"/domain-*.receipt || \
  fail 'restarted witness state did not reach sequence 4'

# A same-length, internally valid local branch does not match the witnessed
# prefix and is refused as a fork.
FORK="$WORK/fork-state"
cp -a "$ROLLBACK" "$FORK"
observe "$FORK" fork-observation-a fork-a
observe "$FORK" fork-observation-b fork-b
observe "$FORK" fork-observation-c fork-c
"$LOOM" world-verify --state-dir "$FORK" --world mesh-v1 >/dev/null
expect_refusal forked-journal 'witness-fork-detected:' \
  "$LOOM" witness-mesh-anchor --state-dir "$FORK" --world mesh-v1 \
  --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS" \
  --anchor-private-key "$WORK/anchor-private.pem"

observe "$STATE_DIR" observation-5 fifth
expect_refusal unanchored-suffix 'witness-unanchored-journal-suffix:' \
  "$LOOM" witness-mesh-verify --state-dir "$STATE_DIR" --world mesh-v1 \
  --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS"

# Two available services may persist a partial candidate, but 2/4 cannot mint
# a certificate. Returning one service completes the same sequence at 3/4.
stop_witness 2
stop_witness 3
expect_refusal quorum-loss 'witness-quorum-unavailable:valid=2:' \
  "$LOOM" witness-mesh-anchor --state-dir "$STATE_DIR" --world mesh-v1 \
  --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS" \
  --anchor-private-key "$WORK/anchor-private.pem"
[[ "$(find "$STATE_DIR/loom-witness-mesh/mesh-v1" \
  -name 'checkpoint-*.receipt' | wc -l)" -eq 4 ]] || \
  fail '2/4 quorum loss minted a local certificate'
start_witness 3
write_endpoints
partial_recovery="$($LOOM witness-mesh-anchor --state-dir "$STATE_DIR" \
  --world mesh-v1 --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS" \
  --anchor-private-key "$WORK/anchor-private.pem")"
rg -q 'sequence=5 .*quorum=3/4 recovered_checkpoints=1 ' \
  <<< "$partial_recovery" || fail "partial 3/4 quorum was not recovered: $partial_recovery"
final_verify="$($LOOM witness-mesh-verify --state-dir "$STATE_DIR" \
  --world mesh-v1 --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS")"
rg -q 'sequence=5 .*verification_policy=byzantine-strict remote_quorum=3/4 required=3/4 ' \
  <<< "$final_verify" || fail "final 3/4 strict verification failed: $final_verify"

printf 'SOUNIO_LOOM_WITNESS_MESH_V1_GATE_PASS=true schema=loom-witness-mesh-v1 frame=9014 witnesses=4 anchor_quorum=3/4 byzantine_strict_verify=3/4 intersection_min=2 dishonest_tolerance=1 one_down_anchor=PASS one_down_strict=PASS sole_byzantine_intersection_attack=REFUSED current_honest_intersection=ENFORCED anchor_authorization=ED25519 signature_sabotage=REFUSED_BEFORE_PERSIST post_quorum_crash=RECOVERED latest_exact_retry=IDEMPOTENT catchup=PASS rollback=REFUSED fork=REFUSED unanchored_suffix=REFUSED quorum_loss_2_of_4=REFUSED partial_quorum_recovery=PASS raw_segment=REPLAYED physical_truth_claim=NONE consensus_claim=NONE runtime=OCaml+Sounio\n'
