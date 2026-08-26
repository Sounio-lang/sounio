#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-coord-runtime-selftest.XXXXXX")"
REPO="$TEST_ROOT/repo"
SECOND="$TEST_ROOT/second-worktree"
STATE="$TEST_ROOT/state"
ALT="$TEST_ROOT/upgrade-source"
BAD="$TEST_ROOT/bad-source"

"$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null
export SOUNIO_LOOM_CONTINUITY_PREBUILT="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-continuity-runtime"
export SOUNIO_LOOM_OBLIGATION_PREBUILT="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-obligation-runtime"

cleanup() {
  [[ -z "${supervisor_pid:-}" ]] || kill "$supervisor_pid" 2>/dev/null || true
  [[ -z "${failing_supervisor_pid:-}" ]] || \
    kill "$failing_supervisor_pid" 2>/dev/null || true
  git -C "$REPO" worktree remove --force "$SECOND" >/dev/null 2>&1 || true
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  echo "sounio-coord-runtime-selftest: FAIL: $*" >&2
  exit 1
}

mkdir -p "$REPO/bin" "$REPO/scripts/dev" "$REPO/formal/tla" "$REPO/tools"
cp "$ROOT_DIR/bin/sounio-coord" "$ROOT_DIR/bin/sounio-agentd" \
  "$ROOT_DIR/bin/sounio-fleet" "$ROOT_DIR/bin/sounio-loom" "$REPO/bin/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_runtime.sh" "$REPO/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_agentd.py" "$REPO/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_fleet.py" "$REPO/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_fleetd.py" "$REPO/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_fleet_tla_sabotage.py" "$REPO/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_fleet_trace_verify.py" "$REPO/scripts/dev/"
cp "$ROOT_DIR/formal/tla/SounioFleet.tla" "$ROOT_DIR/formal/tla/SounioFleet.cfg" \
  "$REPO/formal/tla/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_agent_hook.py" "$REPO/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_agent_hook_runtime.py" "$REPO/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_causal_runtime.py" "$REPO/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/install_sounio_coord_runtime.sh" "$REPO/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/build_sounio_loom.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_continuity_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_obligation_adapter.sh" \
  "$REPO/scripts/dev/"
mkdir -p "$REPO/tools/loom/src"
cp "$ROOT_DIR/tools/loom/dune-project" "$REPO/tools/loom/"
cp "$ROOT_DIR/tools/loom/continuity_adapter_main.sio" "$REPO/tools/loom/"
cp "$ROOT_DIR/tools/loom/obligation_adapter_main.sio" "$REPO/tools/loom/"
cp "$ROOT_DIR/tools/loom/src/dune" "$ROOT_DIR/tools/loom/src/loom.ml" \
  "$ROOT_DIR/tools/loom/src/loom_arrow.ml" \
  "$ROOT_DIR/tools/loom/src/loom_ui.ml" \
  "$ROOT_DIR/tools/loom/src/loom_pty_stubs.c" \
  "$ROOT_DIR/tools/loom/src/loom_arrow_stubs.c" \
  "$ROOT_DIR/tools/loom/src/loom_nanoarrow.c" \
  "$ROOT_DIR/tools/loom/src/loom_nanoarrow_ipc.c" \
  "$ROOT_DIR/tools/loom/src/loom_flatcc.c" "$REPO/tools/loom/src/"
cp -R "$ROOT_DIR/tools/loom/src/vendor" "$REPO/tools/loom/src/"
mkdir -p "$REPO/stdlib/coordination"
cp "$ROOT_DIR/stdlib/coordination/loom_continuity.sio" \
  "$REPO/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_obligation.sio" \
  "$REPO/stdlib/coordination/"
chmod +x "$REPO/bin/"* "$REPO/scripts/dev/"*.sh "$REPO/scripts/dev/"*.py
git -C "$REPO" init -q
git -C "$REPO" config user.name 'Sounio Runtime Selftest'
git -C "$REPO" config user.email 'coord-runtime-selftest@sounio.local'
git -C "$REPO" add .
git -C "$REPO" commit -qm seed
git -C "$REPO" worktree add -q -b second-lane "$SECOND"
RUNTIME_ROOT="$REPO/.git/sounio-coord-runtime"

output="$(cd "$REPO" && SOUNIO_COORD_RUNTIME_MODE=local bin/sounio-coord runtime-info)"
grep -q '^selection=local$' <<< "$output" || fail 'launcher did not report its local fallback'
grep -q '^protocol_version=3$' <<< "$output" || fail 'local runtime protocol is wrong'

output="$(cd "$REPO" && bin/sounio-coord install-runtime)"
first_id="$(sed -n 's/^INSTALLED runtime_id=\([^ ]*\).*/\1/p' <<< "$output")"
[[ -n "$first_id" ]] || fail 'installer did not return the first runtime id'
grep -q "^ACTIVATED runtime_id=$first_id " <<< "$output" || fail 'first runtime was not activated'
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-coord-causal-runtime" ]] || \
  fail 'installed runtime omitted the causal receipt verifier'
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-agentd-runtime" ]] || \
  fail 'installed runtime omitted the detached agent supervisor'
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-loom-runtime" ]] || \
  fail 'installed runtime omitted the OCaml Loom kernel'
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-loom-continuity-runtime" ]] || \
  fail 'installed runtime omitted the native Sounio continuity adapter'
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-loom-obligation-runtime" ]] || \
  fail 'installed runtime omitted the native Sounio obligation adapter'
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-fleet-agent-runtime" ]] || \
  fail 'installed runtime omitted the fleet launcher'
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-fleet-runtime" ]] || \
  fail 'installed runtime omitted the fleet reconciler'
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-fleet-tla-sabotage" ]] || \
  fail 'installed runtime omitted the model-derived sabotage generator'
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-fleet-trace-verify" ]] || \
  fail 'installed runtime omitted the independent trace verifier'
activation_file="$REPO/.git/sounio-coord-state/loom-obligation-activation.v1"
[[ -f "$activation_file" ]] || fail 'installer omitted the durable obligation activation watermark'
grep -q '^schema=loom-obligation-activation-v1$' "$activation_file" || \
  fail 'installer wrote the wrong durable obligation activation schema'
grep -Eq '^activated_epoch=[1-9][0-9]*$' "$activation_file" || \
  fail 'installer wrote an invalid durable obligation activation epoch'
activation_sha="$(sha256sum "$activation_file" | awk '{print $1}')"
[[ -f "$RUNTIME_ROOT/versions/$first_id/formal/SounioFleet.tla" ]] || \
  fail 'installed runtime omitted the TLA+ fleet model'
grep -q '^capability=crash-recovery-v1$' "$RUNTIME_ROOT/versions/$first_id/manifest" || \
  fail 'installed runtime omitted the crash-recovery capability'
grep -q '^capability=agentd-transport-v1$' "$RUNTIME_ROOT/versions/$first_id/manifest" || \
  fail 'installed runtime omitted the agentd transport capability'
grep -q '^capability=fleet-launcher-v1$' "$RUNTIME_ROOT/versions/$first_id/manifest" || \
  fail 'installed runtime omitted the fleet-launcher capability'
grep -q '^capability=fleet-event-log-v1$' "$RUNTIME_ROOT/versions/$first_id/manifest" || \
  fail 'installed runtime omitted the fleet-event-log capability'
grep -q '^capability=fleet-reconciler-v1$' "$RUNTIME_ROOT/versions/$first_id/manifest" || \
  fail 'installed runtime omitted the fleet-reconciler capability'
for capability in agentd-argv-attestation-v1 agentd-tui-submit-v1 \
  agentd-logical-command-v1 coord-reply-correlation-v1 \
  agentd-runtime-registration-v1 loom-kernel-v1 loom-cursor-replay-v1 \
  loom-native-sounio-continuity-v1 \
  loom-durable-obligation-v1 \
  loom-post-activation-request-bridge-v1 \
  loom-recoverable-control-service-v1 \
  loom-beagle-coordination-endpoint-v1 loom-separate-pod-inbox-replay-v1 \
  loom-signed-continuity-receipt-v2 loom-principal-independence-v1 \
  loom-independent-measurement-v1 \
  loom-observation-authority-v1 \
  loom-journal-authority-quorum-v1 \
  loom-cross-node-replay-v1 \
  loom-exclusive-input-lease-v1 loom-read-only-gui-v1 \
  loom-fusion-cockpit-v1 loom-authority-overlay-v1 \
  coord-cockpit-snapshot-v1 loom-persistent-provider-custody-v1 \
  coord-reply-command-v1 loom-coord-transport-v1 \
  coord-generation-scoped-wake-v1 \
  loom-recoverable-guardian-v1 loom-kernel-recovery-v1 loom-dual-journal-v1 \
  loom-persistent-fleet-catalog-v1 loom-post-pod-reconcile-v1 \
  loom-fleet-custody-catalog-v2 loom-conflict-free-active-adoption-v1 \
  loom-coordination-authority-binding-v1 \
  fleet-linear-capability-v1 \
  fleet-home-isolation-v1 \
  fleet-presentation-follow-v1 \
  fleet-proven-exit-v1 fleet-ed25519-anchor-v1 \
  fleet-checkpoint-handoff-v1 fleet-tla-model-v1 \
  fleet-trace-refinement-v1 fleet-temporal-authority-v1 \
  fleet-recovery-start-only-v1 fleet-recovery-directory-v1 \
  fleet-recovery-latch-trace-v1; do
  grep -q "^capability=$capability$" "$RUNTIME_ROOT/versions/$first_id/manifest" || \
    fail "installed runtime omitted capability=$capability"
done

output="$(cd "$SECOND" && bin/sounio-coord runtime-info)"
grep -q '^selection=shared$' <<< "$output" || fail 'second worktree did not select shared runtime'
grep -q "^runtime_id=$first_id$" <<< "$output" || fail 'worktrees selected different runtime ids'
grep -q "$RUNTIME_ROOT/versions/$first_id/bin/sounio-coord-runtime" <<< "$output" || \
  fail 'runtime path is not anchored in the Git common directory'
output="$(cd "$SECOND" && bin/sounio-agentd runtime-info)"
grep -q '^selection=shared$' <<< "$output" || fail 'agentd launcher did not select the shared runtime'
grep -q "^runtime_id=$first_id$" <<< "$output" || fail 'agentd selected a different runtime id'
output="$(cd "$SECOND" && bin/sounio-loom runtime-info)"
grep -q '^selection=shared$' <<< "$output" || fail 'Loom launcher did not select the shared runtime'
grep -q "^runtime_id=$first_id$" <<< "$output" || fail 'Loom selected a different runtime id'
grep -q '^language=OCaml$' <<< "$output" || fail 'shared Loom runtime is not the OCaml kernel'
output="$(cd "$SECOND" && bin/sounio-fleet runtime-info)"
grep -q '^selection=shared$' <<< "$output" || fail 'fleet launcher did not select the shared runtime'
grep -q "^runtime_id=$first_id$" <<< "$output" || fail 'fleet selected a different runtime id'

output="$(
  cd "$SECOND"
  SOUNIO_COORD_DIR="$STATE" bin/sounio-coord send --agent runtime-sender \
    --lane source --to-agent runtime-worker --to-lane target --kind request \
    --message 'runtime-distributed durable obligation'
)"
runtime_message="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$output")"
[[ -n "$runtime_message" ]] || fail 'shared runtime did not send its obligation request'
grep -q '^LOOM_OBLIGATION_OPEN idempotent=no ' <<< "$output" || \
  fail 'shared runtime did not atomically project the request into Loom'
output="$(cd "$SECOND" && SOUNIO_COORD_DIR="$STATE" bin/sounio-coord obligation-list --json)"
grep -q '"count":1,"unclosed":1' <<< "$output" || \
  fail 'shared runtime did not expose its durable obligation projection'

output="$(
  cd "$SECOND"
  SOUNIO_COORD_DIR="$STATE" SOUNIO_COORD_DURABLE_OBLIGATIONS=0 \
    bin/sounio-coord send --agent runtime-sender --lane source \
    --to-agent runtime-worker --to-lane target --kind request \
    --message 'explicit durable obligation opt-out control'
)"
opt_out_message="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$output")"
grep -q '^obligation_opt_out=1$' "$STATE/messages/$opt_out_message.message" || \
  fail 'new client did not distinguish explicit obligation opt-out'

output="$(
  cd "$SECOND"
  SOUNIO_COORD_DIR="$STATE" SOUNIO_COORD_DURABLE_OBLIGATIONS=0 \
    bin/sounio-coord send --agent stale-runtime-sender --lane source \
    --to-agent runtime-worker --to-lane target --kind request \
    --message 'post-activation stale-client request'
)"
stale_message="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$output")"
sed -i '/^obligation_opt_out=1$/d' "$STATE/messages/$stale_message.message"

output="$(
  cd "$SECOND"
  SOUNIO_COORD_DIR="$STATE" SOUNIO_COORD_DURABLE_OBLIGATIONS=0 \
    bin/sounio-coord send --agent historical-runtime-sender --lane source \
    --to-agent runtime-worker --to-lane target --kind request \
    --message 'pre-activation historical request control'
)"
historical_message="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$output")"
activation_epoch="$(sed -n 's/^activated_epoch=//p' "$activation_file")"
historical_epoch=$((activation_epoch - 1))
sed -i '/^obligation_opt_out=1$/d' "$STATE/messages/$historical_message.message"
sed -i "s/^created_epoch=.*/created_epoch=$historical_epoch/" \
  "$STATE/messages/$historical_message.message"

output="$(cd "$SECOND" && SOUNIO_COORD_DIR="$STATE" bin/sounio-coord obligation-reconcile)"
grep -q '^LOOM_OBLIGATION_RECONCILE requests=2 marked=1 legacy=1 ignored=2 state=PASS$' \
  <<< "$output" || \
  fail 'shared runtime obligation reconciliation failed'
output="$(cd "$SECOND" && SOUNIO_COORD_DIR="$STATE" bin/sounio-coord obligation-list --json)"
grep -q '"count":2,"unclosed":2' <<< "$output" || \
  fail 'post-activation stale request was not imported or historical control leaked in'

output="$(cd "$SECOND" && SOUNIO_COORD_DIR="$STATE" \
  bin/sounio-coord obligation-supervisor-ensure --interval-seconds 1)"
grep -q '^LOOM_OBLIGATION_SUPERVISOR_ENSURED state=started ' <<< "$output" || \
  fail 'control-service ensure did not start the obligation supervisor'
supervisor_pid="$(sed -n 's/.* pid=\([0-9][0-9]*\) .*/\1/p' <<< "$output")"
[[ -n "$supervisor_pid" ]] || fail 'control-service ensure omitted its supervisor PID'
supervisor_pid_start="$(sed -n 's/.* pid_start=\([0-9][0-9]*\) .*/\1/p' <<< "$output")"
[[ -n "$supervisor_pid_start" ]] || fail 'control-service ensure omitted its process-start tick'
bootstrap_lock="$STATE/.obligation-supervisor-bootstrap.lock"
flock -n "$bootstrap_lock" -c true || \
  fail 'detached obligation supervisor retained the bootstrap lock'
supervisor_wrapper_pid="$(sed -n 's/^PPid:[[:space:]]*//p' "/proc/$supervisor_pid/status")"
for service_pid in "$supervisor_wrapper_pid" "$supervisor_pid"; do
  for fd in "/proc/$service_pid/fd/"*; do
    [[ "$(readlink -f "$fd" 2>/dev/null || true)" != "$(readlink -f "$bootstrap_lock")" ]] || \
      fail 'detached control service inherited the bootstrap-lock descriptor'
  done
done
output="$(cd "$SECOND" && SOUNIO_COORD_DIR="$STATE" \
  bin/sounio-coord obligation-supervisor-ensure --interval-seconds 1)"
grep -q "state=already-running pid=$supervisor_pid " <<< "$output" || \
  fail 'control-service ensure was not idempotent'
output="$(cd "$SECOND" && SOUNIO_COORD_DIR="$STATE" \
  bin/sounio-coord obligation-supervisor-status)"
grep -q "state=live pid=$supervisor_pid " <<< "$output" || \
  fail 'ensured obligation supervisor did not become live'
output="$(
  cd "$SECOND"
  SOUNIO_COORD_DIR="$STATE" SOUNIO_COORD_DURABLE_OBLIGATIONS=0 \
    bin/sounio-coord send --agent supervised-stale-sender --lane source \
    --to-agent runtime-worker --to-lane target --kind request \
    --message 'stale request created after supervisor start'
)"
supervised_message="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$output")"
sed -i '/^obligation_opt_out=1$/d' "$STATE/messages/$supervised_message.message"
supervised_imported=0
for _ in 1 2 3 4 5; do
  output="$(cd "$SECOND" && SOUNIO_COORD_DIR="$STATE" bin/sounio-coord obligation-list --json)"
  if grep -q '"count":3,"unclosed":3' <<< "$output"; then
    supervised_imported=1
    break
  fi
  sleep 1
done
((supervised_imported == 1)) || fail 'running supervisor did not import a new stale request'
output="$(cd "$SECOND" && SOUNIO_COORD_DIR="$STATE" \
  bin/sounio-coord obligation-supervisor-stop --timeout-seconds 5)"
grep -q "state=stopped pid=$supervisor_pid " <<< "$output" || \
  fail 'control-service stop did not identify the stopped supervisor'
output="$(cd "$SECOND" && SOUNIO_COORD_DIR="$STATE" \
  bin/sounio-coord obligation-supervisor-status 2>/dev/null || true)"
grep -q 'state=stopped' <<< "$output" || \
  fail "terminated obligation supervisor left its OCaml child live: $output"
cp "$STATE/obligation-supervisor.state" "$TEST_ROOT/stopped-supervisor-state.saved"
sed -i '/^schema=/d' "$STATE/obligation-supervisor.state"
set +e
output="$(cd "$SECOND" && SOUNIO_COORD_DIR="$STATE" \
  bin/sounio-coord obligation-supervisor-ensure --interval-seconds 1 2>&1)"
missing_schema_status=$?
set -e
((missing_schema_status != 0)) && grep -q 'invalid obligation supervisor process identity' \
  <<< "$output" || fail 'missing supervisor state schema was not refused'
cp "$TEST_ROOT/stopped-supervisor-state.saved" "$STATE/obligation-supervisor.state"

unowned_pid="$BASHPID"
unowned_start="$(sed 's/^[^)]*) //' "/proc/$unowned_pid/stat" | awk '{print $20}')"
{
  printf 'schema=loom-obligation-supervisor-v1\n'
  printf 'pid=%s\n' "$unowned_pid"
  printf 'pid_start=%s\n' "$unowned_start"
  printf 'replayed_utc=2026-08-25T00:00:00Z\n'
  printf 'count=0\n'
  printf 'unclosed=0\n'
} > "$STATE/obligation-supervisor.state"
set +e
output="$(cd "$SECOND" && SOUNIO_COORD_DIR="$STATE" \
  bin/sounio-coord obligation-supervisor-ensure --interval-seconds 1 2>&1)"
unowned_pid_status=$?
set -e
((unowned_pid_status != 0)) && grep -q 'refusing to signal unowned obligation supervisor' \
  <<< "$output" || fail 'unowned live PID in supervisor state was not fenced'
kill -0 "$unowned_pid" 2>/dev/null || fail 'unowned PID sabotage killed the selftest shell'
cp "$TEST_ROOT/stopped-supervisor-state.saved" "$STATE/obligation-supervisor.state"
(
  cd "$SECOND"
  SOUNIO_COORD_DIR="$STATE" bin/sounio-coord obligation-supervisor-ensure \
    --interval-seconds 1
) > "$TEST_ROOT/concurrent-ensure-a.out" 2>&1 &
ensure_a_pid=$!
(
  cd "$SECOND"
  SOUNIO_COORD_DIR="$STATE" bin/sounio-coord obligation-supervisor-ensure \
    --interval-seconds 1
) > "$TEST_ROOT/concurrent-ensure-b.out" 2>&1 &
ensure_b_pid=$!
wait "$ensure_a_pid"
wait "$ensure_b_pid"
cat "$TEST_ROOT/concurrent-ensure-a.out" "$TEST_ROOT/concurrent-ensure-b.out" > \
  "$TEST_ROOT/concurrent-ensure.out"
[[ "$(grep -c 'state=started ' "$TEST_ROOT/concurrent-ensure.out")" == 1 &&
  "$(grep -c 'state=already-running ' "$TEST_ROOT/concurrent-ensure.out")" == 1 ]] || \
  fail 'concurrent ensure did not elect exactly one control-service starter'
ensure_a_supervisor_pid="$(sed -n 's/.* pid=\([0-9][0-9]*\) .*/\1/p' \
  "$TEST_ROOT/concurrent-ensure-a.out")"
ensure_b_supervisor_pid="$(sed -n 's/.* pid=\([0-9][0-9]*\) .*/\1/p' \
  "$TEST_ROOT/concurrent-ensure-b.out")"
recovered_supervisor_start="$(sed -n 's/.* pid_start=\([0-9][0-9]*\) .*/\1/p' \
  "$TEST_ROOT/concurrent-ensure-a.out")"
[[ -n "$ensure_a_supervisor_pid" && "$ensure_a_supervisor_pid" == "$ensure_b_supervisor_pid" ]] || \
  fail 'concurrent ensure returned different supervisor identities'
recovered_supervisor_pid="$ensure_a_supervisor_pid"
[[ -n "$recovered_supervisor_pid" && "$recovered_supervisor_pid" != "$supervisor_pid" &&
  -n "$recovered_supervisor_start" && "$recovered_supervisor_start" != "$supervisor_pid_start" ]] || \
  fail 'control-service resurrection did not produce a new supervisor generation'
supervisor_pid="$recovered_supervisor_pid"
output="$(cd "$SECOND" && SOUNIO_COORD_DIR="$STATE" \
  bin/sounio-coord obligation-supervisor-stop --timeout-seconds 5)"
grep -q "state=stopped pid=$supervisor_pid " <<< "$output" || \
  fail 'control-service stop did not terminate the resurrected supervisor'
supervisor_pid=''

FAIL_STATE="$TEST_ROOT/failing-supervisor-state"
(
  cd "$SECOND"
  exec env SOUNIO_COORD_DIR="$FAIL_STATE" bin/sounio-coord obligation-supervise \
    --interval-seconds 1
) > "$TEST_ROOT/failing-obligation-supervisor.log" 2>&1 &
failing_supervisor_pid=$!
failing_supervisor_live=0
for _ in 1 2 3 4 5; do
  output="$(cd "$SECOND" && SOUNIO_COORD_DIR="$FAIL_STATE" \
    bin/sounio-coord obligation-supervisor-status 2>/dev/null || true)"
  if grep -q 'state=live' <<< "$output"; then
    failing_supervisor_live=1
    break
  fi
  sleep 1
done
((failing_supervisor_live == 1)) || fail 'negative-control supervisor did not become live'
output="$(
  cd "$SECOND"
  SOUNIO_COORD_DIR="$FAIL_STATE" SOUNIO_COORD_DURABLE_OBLIGATIONS=0 \
    bin/sounio-coord send --agent malformed-sender --lane source \
    --to-agent runtime-worker --to-lane target --kind request \
    --message 'malformed contract must stop supervisor'
)"
malformed_message="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$output")"
printf 'obligation_schema=loom-durable-obligation-v1\n' >> \
  "$FAIL_STATE/messages/$malformed_message.message"
failing_supervisor_stopped=0
for _ in 1 2 3 4 5; do
  if ! kill -0 "$failing_supervisor_pid" 2>/dev/null; then
    failing_supervisor_stopped=1
    break
  fi
  sleep 1
done
if ((failing_supervisor_stopped == 0)); then
  kill "$failing_supervisor_pid" 2>/dev/null || true
  wait "$failing_supervisor_pid" 2>/dev/null || true
  fail 'periodic reconciliation failure did not stop the supervisor'
fi
set +e
wait "$failing_supervisor_pid" 2>/dev/null
failing_supervisor_status=$?
set -e
((failing_supervisor_status != 0)) || \
  fail 'periodic reconciliation failure produced a successful supervisor exit'
output="$(cd "$SECOND" && SOUNIO_COORD_DIR="$FAIL_STATE" \
  bin/sounio-coord obligation-supervisor-status 2>/dev/null || true)"
grep -q 'state=stopped' <<< "$output" || \
  fail 'failed supervisor left its OCaml child live'

# Sabotage only the activation boundary. The formerly historical request must
# become eligible, proving that this boundary caused the earlier refusal.
cp "$activation_file" "$TEST_ROOT/activation-watermark.saved"
sed -i "s/^activated_epoch=.*/activated_epoch=$historical_epoch/" "$activation_file"
output="$(cd "$SECOND" && SOUNIO_COORD_DIR="$STATE" bin/sounio-coord obligation-reconcile)"
grep -q '^LOOM_OBLIGATION_RECONCILE requests=4 marked=1 legacy=3 ignored=1 state=PASS$' \
  <<< "$output" || fail 'activation-boundary sabotage did not admit the historical request'
output="$(cd "$SECOND" && SOUNIO_COORD_DIR="$STATE" bin/sounio-coord obligation-list --json)"
grep -q '"count":4,"unclosed":4' <<< "$output" || \
  fail 'activation-boundary sabotage did not isolate the governing rule'
printf 'runtime outcome\n' > "$TEST_ROOT/runtime-outcome.txt"
printf 'runtime evidence\n' > "$TEST_ROOT/runtime-evidence.txt"
output="$(
  cd "$SECOND"
  runtime_pid="$BASHPID"
  runtime_start="$(sed 's/^[^)]*) //' "/proc/$runtime_pid/stat" | awk '{print $20}')"
  boot_id="$(cat /proc/sys/kernel/random/boot_id)"
  pid_namespace="$(readlink /proc/self/ns/pid)"
  SOUNIO_COORD_DIR="$STATE" bin/sounio-coord claim \
    --agent runtime-worker --lane target \
    --intent 'complete installed-runtime durable obligation' \
    --resources api:runtime-obligation-selftest
  SOUNIO_COORD_DIR="$STATE" bin/sounio-coord presence-register \
    --agent runtime-worker --lane target --harness codex \
    --session-id runtime-obligation-session --pid "$runtime_pid" \
    --pid-start "$runtime_start" --boot-id "$boot_id" \
    --pid-namespace "$pid_namespace" --host runtime-selftest --ttl-seconds 120
  SOUNIO_COORD_DIR="$STATE" bin/sounio-coord cockpit-snapshot
  SOUNIO_COORD_DIR="$STATE" bin/sounio-coord obligation-consume \
    --agent runtime-worker --lane target --message "$runtime_message"
  SOUNIO_COORD_DIR="$STATE" bin/sounio-coord obligation-claim \
    --agent runtime-worker --lane target --message "$runtime_message" \
    --claim runtime-claim --ttl-seconds 120
  SOUNIO_COORD_DIR="$STATE" bin/sounio-coord obligation-complete \
    --agent runtime-worker --lane target --message "$runtime_message" \
    --claim runtime-claim --outcome "$TEST_ROOT/runtime-outcome.txt" \
    --evidence "$TEST_ROOT/runtime-evidence.txt"
  for message in "$stale_message" "$supervised_message" "$historical_message"; do
    SOUNIO_COORD_DIR="$STATE" bin/sounio-coord obligation-consume \
      --agent runtime-worker --lane target --message "$message"
    SOUNIO_COORD_DIR="$STATE" bin/sounio-coord obligation-claim \
      --agent runtime-worker --lane target --message "$message" \
      --claim "runtime-claim-$message" --ttl-seconds 120
    SOUNIO_COORD_DIR="$STATE" bin/sounio-coord obligation-complete \
      --agent runtime-worker --lane target --message "$message" \
      --claim "runtime-claim-$message" --outcome "$TEST_ROOT/runtime-outcome.txt" \
      --evidence "$TEST_ROOT/runtime-evidence.txt"
  done
)"
cp "$TEST_ROOT/activation-watermark.saved" "$activation_file"
[[ "$(sha256sum "$activation_file" | awk '{print $1}')" == "$activation_sha" ]] || \
  fail 'sabotage control did not restore the activation watermark exactly'
grep -q '^LOOM_OBLIGATION_COMPLETED .*state=completed .*unclosed=no ' <<< "$output" || \
  fail 'presence-derived shared-runtime obligation did not complete'
grep -Fq $'COCKPIT\tprotocol=1\t' <<< "$output" || \
  fail 'installed runtime omitted the cockpit machine protocol'
grep -Fq $'CLAIM\tstate=active\tagent=runtime-worker\tlane=target\t' <<< "$output" || \
  fail 'cockpit snapshot omitted its active claim'
grep -Fq $'PRESENCE\tstate=live\treason=process-verified\tagent=runtime-worker\tlane=target\t' \
  <<< "$output" || fail 'cockpit snapshot omitted verified process presence'
if grep -Eq 'token_file=|socket=|address=' <<< "$output"; then
  fail 'cockpit snapshot disclosed a delivery capability'
fi
output="$(cd "$SECOND" && SOUNIO_COORD_DIR="$STATE" bin/sounio-coord obligation-list --json)"
grep -q '"count":4,"unclosed":0' <<< "$output" || \
  fail 'completed shared-runtime obligation remained unclosed'

printf '#!/usr/bin/env bash\nexit 97\n' > "$SECOND/scripts/dev/sounio_coord_runtime.sh"
printf '#!/usr/bin/env python3\nraise SystemExit(98)\n' > \
  "$SECOND/scripts/dev/sounio_coord_agent_hook_runtime.py"
printf '#!/usr/bin/env python3\nraise SystemExit(99)\n' > \
  "$SECOND/scripts/dev/sounio_coord_causal_runtime.py"
printf '#!/usr/bin/env python3\nraise SystemExit(100)\n' > \
  "$SECOND/scripts/dev/sounio_coord_agentd.py"
chmod +x "$SECOND/scripts/dev/sounio_coord_runtime.sh" \
  "$SECOND/scripts/dev/sounio_coord_agent_hook_runtime.py" \
  "$SECOND/scripts/dev/sounio_coord_causal_runtime.py" \
  "$SECOND/scripts/dev/sounio_coord_agentd.py"
output="$(cd "$SECOND" && bin/sounio-coord runtime-info)"
grep -q "^runtime_id=$first_id$" <<< "$output" || \
  fail 'sabotaged worktree fallback displaced the shared CLI runtime'
output="$(
  cd "$SECOND"
  printf '%s\n' \
    "{\"session_id\":\"runtime-test\",\"cwd\":\"$SECOND\",\"hook_event_name\":\"SessionStart\"}" | \
    SOUNIO_COORD_DIR="$STATE" python3 scripts/dev/sounio_coord_agent_hook.py --agent claude
)"
grep -q 'agent=claude lane=session-runtime-test' <<< "$output" || \
  fail 'sabotaged worktree fallback displaced the shared hook runtime'
output="$(cd "$SECOND" && bin/sounio-agentd runtime-info)"
grep -q "^runtime_id=$first_id$" <<< "$output" || \
  fail 'sabotaged worktree fallback displaced the shared agentd runtime'
output="$(cd "$SECOND" && SOUNIO_COORD_DIR="$STATE" \
  bin/sounio-coord obligation-supervisor-ensure --interval-seconds 1)"
grep -q '^LOOM_OBLIGATION_SUPERVISOR_ENSURED state=started ' <<< "$output" || \
  fail 'pre-upgrade control service did not start'
supervisor_pid="$(sed -n 's/.* pid=\([0-9][0-9]*\) .*/\1/p' <<< "$output")"

mkdir -p "$ALT/scripts/dev" "$ALT/formal/tla" "$ALT/tools"
cp "$ROOT_DIR/scripts/dev/sounio_coord_runtime.sh" "$ALT/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_agent_hook_runtime.py" "$ALT/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_causal_runtime.py" "$ALT/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_agentd.py" "$ALT/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_fleet.py" "$ALT/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_fleetd.py" "$ALT/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_fleet_tla_sabotage.py" "$ALT/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_fleet_trace_verify.py" "$ALT/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/build_sounio_loom.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_continuity_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_obligation_adapter.sh" \
  "$ALT/scripts/dev/"
mkdir -p "$ALT/tools/loom/src"
cp "$ROOT_DIR/tools/loom/dune-project" "$ALT/tools/loom/"
cp "$ROOT_DIR/tools/loom/continuity_adapter_main.sio" "$ALT/tools/loom/"
cp "$ROOT_DIR/tools/loom/obligation_adapter_main.sio" "$ALT/tools/loom/"
cp "$ROOT_DIR/tools/loom/src/dune" "$ROOT_DIR/tools/loom/src/loom.ml" \
  "$ROOT_DIR/tools/loom/src/loom_arrow.ml" \
  "$ROOT_DIR/tools/loom/src/loom_ui.ml" \
  "$ROOT_DIR/tools/loom/src/loom_pty_stubs.c" \
  "$ROOT_DIR/tools/loom/src/loom_arrow_stubs.c" \
  "$ROOT_DIR/tools/loom/src/loom_nanoarrow.c" \
  "$ROOT_DIR/tools/loom/src/loom_nanoarrow_ipc.c" \
  "$ROOT_DIR/tools/loom/src/loom_flatcc.c" "$ALT/tools/loom/src/"
cp -R "$ROOT_DIR/tools/loom/src/vendor" "$ALT/tools/loom/src/"
mkdir -p "$ALT/stdlib/coordination"
cp "$ROOT_DIR/stdlib/coordination/loom_continuity.sio" \
  "$ALT/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_obligation.sio" \
  "$ALT/stdlib/coordination/"
cp "$ROOT_DIR/formal/tla/SounioFleet.tla" "$ROOT_DIR/formal/tla/SounioFleet.cfg" \
  "$ALT/formal/tla/"
sed -i 's/^SOUNIO_COORD_RUNTIME_VERSION=.*/SOUNIO_COORD_RUNTIME_VERSION=2026.08.23.8-test/' \
  "$ALT/scripts/dev/sounio_coord_runtime.sh"
chmod +x "$ALT/scripts/dev/"*
output="$(cd "$REPO" && bin/sounio-coord install-runtime --source-root "$ALT")"
second_id="$(sed -n 's/^INSTALLED runtime_id=\([^ ]*\).*/\1/p' <<< "$output")"
[[ -n "$second_id" && "$second_id" != "$first_id" ]] || fail 'upgrade did not create a new runtime id'
output="$(cd "$SECOND" && bin/sounio-coord runtime-info)"
grep -q "^runtime_id=$second_id$" <<< "$output" || fail 'worktree did not observe atomic runtime upgrade'
grep -q '^runtime_version=2026.08.23.8-test$' <<< "$output" || fail 'upgraded runtime version is wrong'
[[ "$(sha256sum "$activation_file" | awk '{print $1}')" == "$activation_sha" ]] || \
  fail 'runtime upgrade rewrote the activation watermark'
output="$(cd "$SECOND" && SOUNIO_COORD_DIR="$STATE" \
  bin/sounio-coord obligation-supervisor-ensure --interval-seconds 1)"
grep -q '^LOOM_OBLIGATION_SUPERVISOR_ENSURED state=restarted ' <<< "$output" || \
  fail 'runtime upgrade did not restart the control service onto the selected bundle'
upgraded_supervisor_pid="$(sed -n 's/.* pid=\([0-9][0-9]*\) .*/\1/p' <<< "$output")"
[[ -n "$upgraded_supervisor_pid" && "$upgraded_supervisor_pid" != "$supervisor_pid" ]] || \
  fail 'runtime upgrade retained the old control-service generation'
supervisor_pid="$upgraded_supervisor_pid"

output="$(cd "$REPO" && bin/sounio-coord install-runtime --activate "$first_id")"
grep -q "^ACTIVATED runtime_id=$first_id " <<< "$output" || fail 'runtime rollback failed'
output="$(cd "$SECOND" && bin/sounio-coord runtime-info)"
grep -q "^runtime_id=$first_id$" <<< "$output" || fail 'worktree did not observe runtime rollback'
[[ "$(sha256sum "$activation_file" | awk '{print $1}')" == "$activation_sha" ]] || \
  fail 'runtime rollback rewrote the activation watermark'
output="$(cd "$SECOND" && SOUNIO_COORD_DIR="$STATE" \
  bin/sounio-coord obligation-supervisor-ensure --interval-seconds 1)"
grep -q '^LOOM_OBLIGATION_SUPERVISOR_ENSURED state=restarted ' <<< "$output" || \
  fail 'runtime rollback did not restart the control service onto the selected bundle'
rolled_back_supervisor_pid="$(sed -n 's/.* pid=\([0-9][0-9]*\) .*/\1/p' <<< "$output")"
[[ -n "$rolled_back_supervisor_pid" && "$rolled_back_supervisor_pid" != "$supervisor_pid" ]] || \
  fail 'runtime rollback retained the upgraded control-service generation'
supervisor_pid="$rolled_back_supervisor_pid"
output="$(cd "$SECOND" && SOUNIO_COORD_DIR="$STATE" \
  bin/sounio-coord obligation-supervisor-stop --timeout-seconds 5)"
grep -q "state=stopped pid=$supervisor_pid " <<< "$output" || \
  fail 'control service did not stop after rollback validation'
supervisor_pid=''

mkdir -p "$BAD/scripts/dev" "$BAD/formal/tla" "$BAD/tools"
cp "$ROOT_DIR/scripts/dev/sounio_coord_runtime.sh" "$BAD/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_agent_hook_runtime.py" "$BAD/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_causal_runtime.py" "$BAD/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_agentd.py" "$BAD/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_fleet.py" "$BAD/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_fleetd.py" "$BAD/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_fleet_tla_sabotage.py" "$BAD/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_fleet_trace_verify.py" "$BAD/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/build_sounio_loom.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_continuity_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_obligation_adapter.sh" \
  "$BAD/scripts/dev/"
mkdir -p "$BAD/tools/loom/src"
cp "$ROOT_DIR/tools/loom/dune-project" "$BAD/tools/loom/"
cp "$ROOT_DIR/tools/loom/continuity_adapter_main.sio" "$BAD/tools/loom/"
cp "$ROOT_DIR/tools/loom/obligation_adapter_main.sio" "$BAD/tools/loom/"
cp "$ROOT_DIR/tools/loom/src/dune" "$ROOT_DIR/tools/loom/src/loom.ml" \
  "$ROOT_DIR/tools/loom/src/loom_arrow.ml" \
  "$ROOT_DIR/tools/loom/src/loom_ui.ml" \
  "$ROOT_DIR/tools/loom/src/loom_pty_stubs.c" \
  "$ROOT_DIR/tools/loom/src/loom_arrow_stubs.c" \
  "$ROOT_DIR/tools/loom/src/loom_nanoarrow.c" \
  "$ROOT_DIR/tools/loom/src/loom_nanoarrow_ipc.c" \
  "$ROOT_DIR/tools/loom/src/loom_flatcc.c" "$BAD/tools/loom/src/"
cp -R "$ROOT_DIR/tools/loom/src/vendor" "$BAD/tools/loom/src/"
mkdir -p "$BAD/stdlib/coordination"
cp "$ROOT_DIR/stdlib/coordination/loom_continuity.sio" \
  "$BAD/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_obligation.sio" \
  "$BAD/stdlib/coordination/"
cp "$ROOT_DIR/formal/tla/SounioFleet.tla" "$ROOT_DIR/formal/tla/SounioFleet.cfg" \
  "$BAD/formal/tla/"
sed -i 's/SOUNIO_COORD_PROTOCOL_VERSION=3/SOUNIO_COORD_PROTOCOL_VERSION=4/' \
  "$BAD/scripts/dev/sounio_coord_runtime.sh"
chmod +x "$BAD/scripts/dev/"*
if (cd "$REPO" && bin/sounio-coord install-runtime --source-root "$BAD") >/dev/null 2>&1; then
  fail 'installer accepted an incompatible protocol'
fi
mkdir -p "$RUNTIME_ROOT/versions/incomplete"
if (cd "$REPO" && bin/sounio-coord install-runtime --activate incomplete) >/dev/null 2>&1; then
  fail 'installer activated an incomplete runtime'
fi
output="$(cd "$REPO" && bin/sounio-coord runtime-info)"
grep -q "^runtime_id=$first_id$" <<< "$output" || fail 'failed activation changed the current runtime'

output="$(cd "$REPO" && bin/sounio-coord install-runtime --list)"
grep -q "runtime_id=$first_id current=yes" <<< "$output" || fail 'runtime list lost the current marker'
grep -q "runtime_id=$second_id current=no" <<< "$output" || fail 'runtime list lost installed upgrade'

unlink "$RUNTIME_ROOT/current"
ln -s versions/missing "$RUNTIME_ROOT/current"
if (cd "$REPO" && bin/sounio-coord runtime-info) >/dev/null 2>&1; then
  fail 'CLI launcher silently fell back across a broken shared-runtime link'
fi
if (cd "$REPO" && bin/sounio-agentd runtime-info) >/dev/null 2>&1; then
  fail 'agentd launcher silently fell back across a broken shared-runtime link'
fi
if (cd "$REPO" && bin/sounio-fleet runtime-info) >/dev/null 2>&1; then
  fail 'fleet launcher silently fell back across a broken shared-runtime link'
fi
if (
  cd "$REPO"
  printf '%s\n' \
    "{\"session_id\":\"broken-link\",\"cwd\":\"$REPO\",\"hook_event_name\":\"SessionStart\"}" | \
    SOUNIO_COORD_DIR="$STATE" python3 scripts/dev/sounio_coord_agent_hook.py --agent claude
) >/dev/null 2>&1; then
  fail 'hook launcher silently fell back across a broken shared-runtime link'
fi
output="$(cd "$REPO" && scripts/dev/install_sounio_coord_runtime.sh --activate "$first_id")"
grep -q "^ACTIVATED runtime_id=$first_id " <<< "$output" || \
  fail 'installer did not recover a broken current link atomically'

echo 'sounio-coord-runtime-selftest: PASS'
