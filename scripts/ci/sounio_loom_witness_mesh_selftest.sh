#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-witness-mesh.XXXXXX")"
STATE_DIR="$WORK/state"
MEMBERSHIP="$WORK/membership.tsv"
ENDPOINTS="$WORK/endpoints.tsv"
declare -a WITNESS_PIDS=("" "" "" "")
declare -a WITNESS_PORTS=("" "" "" "")

cleanup() {
  local index pid
  for index in 1 2 3; do
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
  printf 'loom-witness-mesh: FAIL: %s\n' "$*" >&2
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
  until rg -q 'LOOM_WITNESS_READY ' "$log"; do
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
  printf 'w1\t127.0.0.1\t%s\n' "${WITNESS_PORTS[1]}" >> "$ENDPOINTS"
  printf 'w2\t127.0.0.1\t%s\n' "${WITNESS_PORTS[2]}" >> "$ENDPOINTS"
  printf 'w3\t127.0.0.1\t%s\n' "${WITNESS_PORTS[3]}" >> "$ENDPOINTS"
}

observe() {
  local state="$1" knowledge="$2" value="$3"
  "$LOOM" knowledge-observe --state-dir "$state" --world mesh \
    --knowledge "$knowledge" --value "$value" --error 0 \
    --uncertainty bounded --confidence 1 \
    --provenance "$(digest "$knowledge-provenance")" >/dev/null
}

"$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null

for index in 1 2 3; do
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
printf 'anchor_public_key\t%s\n' "$WORK/anchor-public.pem" > "$MEMBERSHIP"
printf 'witness_id\tpublic_key\n' >> "$MEMBERSHIP"
for index in 1 2 3; do
  printf 'w%s\t%s\n' "$index" "$WORK/witness-$index-public.pem" >> "$MEMBERSHIP"
done
for index in 1 2 3; do
  start_witness "$index"
done
write_endpoints

"$LOOM" world-create --state-dir "$STATE_DIR" --world mesh \
  --agent codex --lane witness-mesh-v0 >/dev/null
observe "$STATE_DIR" observation-1 first

expect_refusal anchor-key-substitution 'witness-anchor-private-key-mismatch' \
  "$LOOM" witness-mesh-anchor --state-dir "$STATE_DIR" --world mesh \
  --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS" \
  --anchor-private-key "$WORK/attacker-anchor-private.pem"

# This request is signed with the configured key and then corrupted in flight.
# All three services must refuse the authorization before persisting state.
expect_refusal anchor-signature-sabotage \
  'witness-quorum-unavailable:valid=0:.*witness-request-anchor-signature-invalid' \
  env SOUNIO_LOOM_WITNESS_TAMPER_ANCHOR_SIGNATURE=1 \
  "$LOOM" witness-mesh-anchor --state-dir "$STATE_DIR" --world mesh \
  --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS" \
  --anchor-private-key "$WORK/anchor-private.pem"
[[ "$(find "$WORK" -path '*/domain-*.receipt' | wc -l)" -eq 0 ]] || \
  fail 'anchor signature sabotage advanced witness state'

# Keep exactly one member unavailable at the crash boundary: two external
# services persist the accepted checkpoint, while no local certificate exists.
stop_witness 3

# Both available witnesses persist sequence 1, then the client dies before its
# local certificate write. A fresh client must recover the signed checkpoint.
set +e
SOUNIO_LOOM_WITNESS_FAILPOINT=after-quorum-before-certificate \
  "$LOOM" witness-mesh-anchor --state-dir "$STATE_DIR" --world mesh \
    --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS" \
    --anchor-private-key "$WORK/anchor-private.pem" \
    >"$WORK/post-quorum-crash.out" 2>"$WORK/post-quorum-crash.err"
crash_rc=$?
set -e
[[ "$crash_rc" -eq 197 ]] || fail "post-quorum failpoint returned rc=$crash_rc"
rg -Fxq 'LOOM_WITNESS_FAILPOINT name=after-quorum-before-certificate exit=197' \
  "$WORK/post-quorum-crash.err" || fail 'post-quorum failpoint did not fire'
[[ "$(find "$STATE_DIR/loom-witness-mesh/mesh" \
  -name 'checkpoint-*.receipt' | wc -l)" -eq 0 ]] || \
  fail 'post-quorum crash wrote a local certificate'
recovered="$($LOOM witness-mesh-anchor --state-dir "$STATE_DIR" --world mesh \
  --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS" \
  --anchor-private-key "$WORK/anchor-private.pem")"
rg -q 'sequence=1 .*quorum=2/3 recovered_checkpoints=1 ' <<< "$recovered" || \
  fail "post-quorum recovery was not explicit: $recovered"
expect_refusal byzantine-strict-one-down \
  'witness-current-quorum-unavailable:policy=byzantine-strict:valid=2:required=3' \
  "$LOOM" witness-mesh-verify --state-dir "$STATE_DIR" --world mesh \
  --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS"
"$LOOM" witness-mesh-verify --state-dir "$STATE_DIR" --world mesh \
  --policy crash-quorum \
  --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS" >/dev/null

ROLLBACK="$WORK/rollback-state"
cp -a "$STATE_DIR" "$ROLLBACK"

# Sequence 2 remains available with one witness down.
observe "$STATE_DIR" observation-2 second
stop_witness 3
one_down="$($LOOM witness-mesh-anchor --state-dir "$STATE_DIR" --world mesh \
  --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS" \
  --anchor-private-key "$WORK/anchor-private.pem")"
rg -q 'sequence=2 .*quorum=2/3 ' <<< "$one_down" || \
  fail "one-down quorum did not anchor: $one_down"
"$LOOM" witness-mesh-verify --state-dir "$STATE_DIR" --world mesh \
  --policy crash-quorum \
  --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS" >/dev/null

# The restarted witness retains sequence 1, receives the missed raw suffix,
# and jumps monotonically to sequence 3 while the other two advance normally.
start_witness 3
write_endpoints
observe "$STATE_DIR" observation-3 third
catchup="$($LOOM witness-mesh-anchor --state-dir "$STATE_DIR" --world mesh \
  --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS" \
  --anchor-private-key "$WORK/anchor-private.pem")"
rg -q 'sequence=3 .*quorum=3/3 ' <<< "$catchup" || \
  fail "restarted witness did not catch up: $catchup"
rg -q '^anchor_sequence=3$' "$WORK/witness-3-state"/domain-*.receipt || \
  fail 'restarted witness state did not reach sequence 3'
"$LOOM" witness-mesh-verify --state-dir "$STATE_DIR" --world mesh \
  --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS" >/dev/null

# The ordinary local verifier accepts the old snapshot. External witnesses do
# not: their monotonic state proves that the local snapshot is behind.
"$LOOM" world-verify --state-dir "$ROLLBACK" --world mesh >/dev/null
expect_refusal rollback-snapshot 'witness-rollback-detected:' \
  "$LOOM" witness-mesh-verify --state-dir "$ROLLBACK" --world mesh \
  --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS"

# Grow a valid but different local branch to the remote event count. The
# branch is internally hash-consistent, yet the witnessed prefix differs.
FORK="$WORK/fork-state"
cp -a "$ROLLBACK" "$FORK"
observe "$FORK" fork-observation-a fork-a
observe "$FORK" fork-observation-b fork-b
"$LOOM" world-verify --state-dir "$FORK" --world mesh >/dev/null
expect_refusal forked-journal 'witness-fork-detected:' \
  "$LOOM" witness-mesh-anchor --state-dir "$FORK" --world mesh \
  --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS" \
  --anchor-private-key "$WORK/anchor-private.pem"

# A locally appended suffix is visible but not covered by the last certificate.
observe "$STATE_DIR" observation-4 fourth
expect_refusal unanchored-suffix 'witness-unanchored-journal-suffix:' \
  "$LOOM" witness-mesh-verify --state-dir "$STATE_DIR" --world mesh \
  --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS"

# With two services absent, one witness may persist the candidate checkpoint,
# but the client refuses to mint a local quorum certificate. When one service
# returns, the signed partial state is completed instead of rolled back.
stop_witness 2
stop_witness 3
expect_refusal quorum-loss 'witness-quorum-unavailable:valid=1:' \
  "$LOOM" witness-mesh-anchor --state-dir "$STATE_DIR" --world mesh \
  --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS" \
  --anchor-private-key "$WORK/anchor-private.pem"
[[ "$(find "$STATE_DIR/loom-witness-mesh/mesh" \
  -name 'checkpoint-*.receipt' | wc -l)" -eq 3 ]] || \
  fail 'quorum loss minted a local certificate'
start_witness 2
write_endpoints
partial_recovery="$($LOOM witness-mesh-anchor --state-dir "$STATE_DIR" --world mesh \
  --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS" \
  --anchor-private-key "$WORK/anchor-private.pem")"
rg -q 'sequence=4 .*quorum=2/3 recovered_checkpoints=1 ' \
  <<< "$partial_recovery" || \
  fail "partial quorum was not recovered: $partial_recovery"
expect_refusal final-byzantine-strict-one-down \
  'witness-current-quorum-unavailable:policy=byzantine-strict:valid=2:required=3' \
  "$LOOM" witness-mesh-verify --state-dir "$STATE_DIR" --world mesh \
  --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS"
final_verify="$($LOOM witness-mesh-verify --state-dir "$STATE_DIR" --world mesh \
  --membership "$MEMBERSHIP" --endpoints "$ENDPOINTS" --policy crash-quorum)"
rg -q 'sequence=4 .*verification_policy=crash-quorum remote_quorum=2/3 required=2/3 ' \
  <<< "$final_verify" || \
  fail "final external quorum did not verify: $final_verify"

printf 'SOUNIO_LOOM_WITNESS_MESH_GATE_PASS=true schema=loom-witness-mesh-v0 frame=9013 witnesses=3 anchor_quorum=2/3 crash_verify_quorum=2/3 byzantine_strict_verify=3/3 raw_segment=REPLAYED witness_signatures=ED25519 anchor_authorization=ED25519 key_substitution=REFUSED signature_sabotage=REFUSED_BEFORE_PERSIST post_quorum_crash=RECOVERED latest_exact_retry=IDEMPOTENT catchup=PASS one_down_anchor=PASS one_down_byzantine_strict=REFUSED rollback=REFUSED fork=REFUSED unanchored_suffix=REFUSED quorum_loss=REFUSED partial_quorum_recovery=PASS restart_state=PASS local_only_verifier=INSUFFICIENT rollback_scope=THROUGH_LATEST_CHECKPOINT physical_truth_claim=NONE runtime=OCaml+Sounio\n'
