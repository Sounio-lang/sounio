#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-witness-epoch-transparency.XXXXXX")"
WORLD="epoch-transparency"
BASE_ROOT="$WORK/base-root"
E1_ROOT="$WORK/epoch1-root"
E2_ROOT="$WORK/epoch2-root"
E3_ROOT="$WORK/epoch3-root"
EPOCH_STATE="$WORK/epoch-state"
ROLLED_STATE="$WORK/rolled-epoch-state"
TRANSPARENCY_STATE="$WORK/transparency-state"
LOG_STATE="$WORK/log-state"
declare -A PIDS=()
declare -A PORTS=()
LOG_PID=""
LOG_PORT=""

cleanup() {
  local key pid
  if [[ -n "$LOG_PID" ]]; then
    kill "$LOG_PID" 2>/dev/null || true
    wait "$LOG_PID" 2>/dev/null || true
  fi
  for key in "${!PIDS[@]}"; do
    pid="${PIDS[$key]:-}"
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  rm -rf "$WORK"
}
trap cleanup EXIT

fail() {
  printf 'loom-witness-epoch-transparency: FAIL: %s\n' "$*" >&2
  exit 1
}

digest() {
  printf '%s' "$1" | sha256sum | awk '{print $1}'
}

membership() { printf '%s/%s-membership.tsv\n' "$WORK" "$1"; }
endpoints() { printf '%s/%s-endpoints.tsv\n' "$WORK" "$1"; }

start_witness() {
  local group="$1" index="$2" key="${1}${2}" log="$WORK/${1}${2}.log"
  local pid attempt=0
  : > "$log"
  "$LOOM" witness-serve --witness-state-dir "$WORK/$key-state" \
    --membership "$(membership "$group")" --witness "$key" \
    --private-key "$WORK/$key-private.pem" --bind 127.0.0.1 --port 0 \
    >"$log" 2>&1 &
  pid=$!
  PIDS[$key]="$pid"
  until rg -q 'LOOM_WITNESS_READY schema=loom-witness-service-v1 ' "$log"; do
    kill -0 "$pid" 2>/dev/null || {
      sed -n '1,160p' "$log" >&2
      fail "$key exited before readiness"
    }
    attempt=$((attempt + 1))
    [[ "$attempt" -lt 240 ]] || fail "$key readiness timed out"
    sleep 0.025
  done
  PORTS[$key]="$(sed -n 's/.* port=\([0-9][0-9]*\) .*/\1/p' "$log")"
  [[ -n "${PORTS[$key]}" ]] || fail "$key omitted its port"
}

stop_witness() {
  local key="${1}${2}" pid="${PIDS[${1}${2}]:-}"
  if [[ -n "$pid" ]]; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    PIDS[$key]=""
  fi
}

write_endpoints() {
  local group="$1" index key file
  file="$(endpoints "$group")"
  printf 'witness_id\thost\tport\n' > "$file"
  for index in 1 2 3 4; do
    key="${group}${index}"
    printf '%s\t127.0.0.1\t%s\n' "$key" "${PORTS[$key]}" >> "$file"
  done
}

start_log() {
  local log="$WORK/operator.log" attempt=0
  : > "$log"
  "$LOOM" witness-epoch-log-serve --log-state-dir "$LOG_STATE" \
    --operator log-operator \
    --operator-public-key "$WORK/operator-public.pem" \
    --operator-private-key "$WORK/operator-private.pem" \
    --publisher-public-key "$WORK/publisher-public.pem" \
    --bind 127.0.0.1 --log-port 0 >"$log" 2>&1 &
  LOG_PID=$!
  until rg -q 'LOOM_EPOCH_TRANSPARENCY_LOG_READY ' "$log"; do
    kill -0 "$LOG_PID" 2>/dev/null || {
      sed -n '1,180p' "$log" >&2
      fail 'log operator exited before readiness'
    }
    attempt=$((attempt + 1))
    [[ "$attempt" -lt 240 ]] || fail 'log operator readiness timed out'
    sleep 0.025
  done
  LOG_PORT="$(sed -n 's/.* port=\([0-9][0-9]*\) .*/\1/p' "$log")"
  [[ -n "$LOG_PORT" ]] || fail 'log operator omitted its port'
}

stop_log() {
  if [[ -n "$LOG_PID" ]]; then
    kill "$LOG_PID" 2>/dev/null || true
    wait "$LOG_PID" 2>/dev/null || true
    LOG_PID=""
  fi
}

expect_refusal() {
  local label="$1" expected="$2" rc=0
  shift 2
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

observe() {
  "$LOOM" knowledge-observe --state-dir "$1" --world "$WORLD" \
    --knowledge checkpoint --value shared --error 0 --uncertainty bounded \
    --confidence 1 --provenance "$(digest checkpoint-provenance)" >/dev/null
}

anchor_epoch() {
  local group="$1" root="$2"
  "$LOOM" witness-mesh-anchor --state-dir "$root" --world "$WORLD" \
    --membership "$(membership "$group")" --endpoints "$(endpoints "$group")" \
    --anchor-private-key "$WORK/$group-anchor-private.pem"
}

handoff() {
  "$LOOM" witness-epoch-handoff --epoch-state-dir "$EPOCH_STATE" \
    --world "$WORLD" --from-epoch "$1" --to-epoch "$2" \
    --old-state-dir "$3" --old-membership "$(membership "$4")" \
    --old-endpoints "$(endpoints "$4")" --new-state-dir "$5" \
    --new-membership "$(membership "$6")" --new-endpoints "$(endpoints "$6")"
}

publish() {
  "$LOOM" witness-epoch-transparency-publish --epoch-state-dir "$EPOCH_STATE" \
    --transparency-state-dir "$TRANSPARENCY_STATE" --world "$WORLD" \
    --log-host 127.0.0.1 --log-port "$LOG_PORT" --operator log-operator \
    --operator-public-key "$WORK/operator-public.pem" \
    --publisher-public-key "$WORK/publisher-public.pem" \
    --publisher-private-key "$WORK/publisher-private.pem" \
    --transparency-membership "$(membership tr)" \
    --transparency-endpoints "$(endpoints tr)" \
    --transparency-anchor-private-key "$WORK/tr-anchor-private.pem"
}

verify_transparency_at() {
  "$LOOM" witness-epoch-transparency-verify --epoch-state-dir "$1" \
    --transparency-state-dir "$TRANSPARENCY_STATE" --world "$WORLD" \
    --log-host 127.0.0.1 --log-port "$LOG_PORT" --operator log-operator \
    --operator-public-key "$WORK/operator-public.pem" \
    --transparency-membership "$(membership tr)" \
    --transparency-endpoints "$(endpoints tr)"
}

"$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null

for group in e1 e2 e3 tr; do
  openssl genpkey -algorithm ED25519 -out "$WORK/$group-anchor-private.pem" 2>/dev/null
  openssl pkey -in "$WORK/$group-anchor-private.pem" -pubout \
    -out "$WORK/$group-anchor-public.pem" 2>/dev/null
  for index in 1 2 3 4; do
    key="${group}${index}"
    openssl genpkey -algorithm ED25519 -out "$WORK/$key-private.pem" 2>/dev/null
    openssl pkey -in "$WORK/$key-private.pem" -pubout \
      -out "$WORK/$key-public.pem" 2>/dev/null
  done
  file="$(membership "$group")"
  printf 'schema\tloom-witness-membership-v1\n' > "$file"
  printf 'anchor_public_key\t%s\n' "$WORK/$group-anchor-public.pem" >> "$file"
  printf 'witness_id\tpublic_key\n' >> "$file"
  for index in 1 2 3 4; do
    key="${group}${index}"
    printf '%s\t%s\n' "$key" "$WORK/$key-public.pem" >> "$file"
  done
done

openssl genpkey -algorithm ED25519 -out "$WORK/operator-private.pem" 2>/dev/null
openssl pkey -in "$WORK/operator-private.pem" -pubout \
  -out "$WORK/operator-public.pem" 2>/dev/null
openssl genpkey -algorithm ED25519 -out "$WORK/publisher-private.pem" 2>/dev/null
openssl pkey -in "$WORK/publisher-private.pem" -pubout \
  -out "$WORK/publisher-public.pem" 2>/dev/null

for group in e1 e2 e3 tr; do
  for index in 1 2 3 4; do start_witness "$group" "$index"; done
  write_endpoints "$group"
done
start_log

"$LOOM" world-create --state-dir "$BASE_ROOT" --world "$WORLD" \
  --agent codex --lane epoch-transparency >/dev/null
observe "$BASE_ROOT"
cp -a "$BASE_ROOT" "$E1_ROOT"
cp -a "$BASE_ROOT" "$E2_ROOT"
cp -a "$BASE_ROOT" "$E3_ROOT"
anchor_epoch e1 "$E1_ROOT" >/dev/null
anchor_epoch e2 "$E2_ROOT" >/dev/null
anchor_epoch e3 "$E3_ROOT" >/dev/null

handoff 1 2 "$E1_ROOT" e1 "$E2_ROOT" e2 >/dev/null

# A physically co-located operator is rejected unless the explicit test-only
# switch is present. The local gate never promotes that switch as custody proof.
expect_refusal independence-collapse 'epoch-transparency-operator-host-collapse' publish

set +e
SOUNIO_LOOM_WITNESS_EPOCH_TRANSPARENCY_TEST_ALLOW_SAME_HOST=1 \
SOUNIO_LOOM_WITNESS_EPOCH_TRANSPARENCY_FAILPOINT=after-log-append-before-anchor \
  publish >"$WORK/append-crash.out" 2>"$WORK/append-crash.err"
append_rc=$?
set -e
[[ "$append_rc" -eq 196 ]] || fail "append crash returned rc=$append_rc"
rg -q 'after-log-append-before-anchor' "$WORK/append-crash.err" || \
  fail 'append crash failpoint did not fire'
TRANSPARENT_ACTIVE="$EPOCH_STATE/loom-witness-epochs/$WORLD/transparency/transparent-active.receipt"
[[ ! -e "$TRANSPARENT_ACTIVE" ]] || fail 'append crash published strong active state'

first_publish="$(SOUNIO_LOOM_WITNESS_EPOCH_TRANSPARENCY_TEST_ALLOW_SAME_HOST=1 publish)"
rg -q 'epoch=2 tree_size=1 .*quorum=4/4 .*custody=SIMULATED_NOT_CLAIMED native_frame=9016' \
  <<< "$first_publish" || fail "first publish did not recover: $first_publish"
first_verify="$(SOUNIO_LOOM_WITNESS_EPOCH_TRANSPARENCY_TEST_ALLOW_SAME_HOST=1 \
  verify_transparency_at "$EPOCH_STATE")"
rg -q 'epoch=2 tree_size=1 .*quorum=4/4 .*rollback=NOT_BELOW_LATEST_QUORUM_WITNESSED .*freeze_claim=NONE .*custody=SIMULATED_NOT_CLAIMED native_frame=9016' \
  <<< "$first_verify" || fail "first transparency verify failed: $first_verify"

cp -a "$EPOCH_STATE" "$ROLLED_STATE"

handoff 2 3 "$E2_ROOT" e2 "$E3_ROOT" e3 >/dev/null
second_publish="$(SOUNIO_LOOM_WITNESS_EPOCH_TRANSPARENCY_TEST_ALLOW_SAME_HOST=1 publish)"
rg -q 'epoch=3 tree_size=2 .*quorum=4/4 ' <<< "$second_publish" || \
  fail "second publish failed: $second_publish"
second_verify="$(SOUNIO_LOOM_WITNESS_EPOCH_TRANSPARENCY_TEST_ALLOW_SAME_HOST=1 \
  verify_transparency_at "$EPOCH_STATE")"
rg -q 'epoch=3 tree_size=2 .*active=LATEST' <<< "$second_verify" || \
  fail "second transparency verify failed: $second_verify"

expect_refusal rolled-local-state 'epoch-transparency-rollback-or-split-view-detected' \
  env SOUNIO_LOOM_WITNESS_EPOCH_TRANSPARENCY_TEST_ALLOW_SAME_HOST=1 \
  "$LOOM" witness-epoch-transparency-verify --epoch-state-dir "$ROLLED_STATE" \
    --transparency-state-dir "$TRANSPARENCY_STATE" --world "$WORLD" \
    --log-host 127.0.0.1 --log-port "$LOG_PORT" --operator log-operator \
    --operator-public-key "$WORK/operator-public.pem" \
    --transparency-membership "$(membership tr)" \
    --transparency-endpoints "$(endpoints tr)"

stop_log
expect_refusal unreachable-log 'Connection refused' \
  env SOUNIO_LOOM_WITNESS_EPOCH_TRANSPARENCY_TEST_ALLOW_SAME_HOST=1 \
  "$LOOM" witness-epoch-transparency-verify --epoch-state-dir "$EPOCH_STATE" \
    --transparency-state-dir "$TRANSPARENCY_STATE" --world "$WORLD" \
    --log-host 127.0.0.1 --log-port "$LOG_PORT" --operator log-operator \
    --operator-public-key "$WORK/operator-public.pem" \
    --transparency-membership "$(membership tr)" \
    --transparency-endpoints "$(endpoints tr)"
start_log

stop_witness tr 3
stop_witness tr 4
expect_refusal quorum-loss 'epoch-transparency-current-quorum-unavailable:valid=2:required=3' \
  env SOUNIO_LOOM_WITNESS_EPOCH_TRANSPARENCY_TEST_ALLOW_SAME_HOST=1 \
  "$LOOM" witness-epoch-transparency-verify --epoch-state-dir "$EPOCH_STATE" \
    --transparency-state-dir "$TRANSPARENCY_STATE" --world "$WORLD" \
    --log-host 127.0.0.1 --log-port "$LOG_PORT" --operator log-operator \
    --operator-public-key "$WORK/operator-public.pem" \
    --transparency-membership "$(membership tr)" \
    --transparency-endpoints "$(endpoints tr)"
start_witness tr 3
start_witness tr 4
write_endpoints tr

JOURNAL="$(find "$LOG_STATE" -name journal.tsv -type f -print -quit)"
[[ -n "$JOURNAL" ]] || fail 'operator journal was not found'
cp "$JOURNAL" "$WORK/operator-journal.good"
stop_log
head -n 1 "$WORK/operator-journal.good" > "$JOURNAL"
start_log
expect_refusal signed-split-view 'epoch-transparency-rollback-or-split-view-detected' \
  env SOUNIO_LOOM_WITNESS_EPOCH_TRANSPARENCY_TEST_ALLOW_SAME_HOST=1 \
  "$LOOM" witness-epoch-transparency-verify --epoch-state-dir "$EPOCH_STATE" \
    --transparency-state-dir "$TRANSPARENCY_STATE" --world "$WORLD" \
    --log-host 127.0.0.1 --log-port "$LOG_PORT" --operator log-operator \
    --operator-public-key "$WORK/operator-public.pem" \
    --transparency-membership "$(membership tr)" \
    --transparency-endpoints "$(endpoints tr)"
stop_log
cp "$WORK/operator-journal.good" "$JOURNAL"
start_log
SOUNIO_LOOM_WITNESS_EPOCH_TRANSPARENCY_TEST_ALLOW_SAME_HOST=1 \
  verify_transparency_at "$EPOCH_STATE" >/dev/null

printf 'SOUNIO_LOOM_WITNESS_EPOCH_TRANSPARENCY_GATE_PASS=true schema=loom-witness-epoch-transparency-v0 frame=9016 transitions=2 merkle=RFC6962_STYLE proof=MATERIALIZED_PREFIX_RECOMPUTED witnesses=4 quorum=3/4 log_signature=VERIFIED append_crash=RECOVERED exact_retry=IDEMPOTENT rolled_local_state=REFUSED signed_split_view=REFUSED unreachable_log=FAIL_CLOSED quorum_loss=REFUSED independence_collapse=REFUSED latest_leaf=REQUIRED freeze_claim=NONE availability_claim=NONE recovery_claim=NONE local_custody=SIMULATED_NOT_CLAIMED runtime=OCaml+Sounio\n'
