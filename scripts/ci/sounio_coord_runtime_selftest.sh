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

cleanup() {
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
  "$REPO/scripts/dev/"
mkdir -p "$REPO/tools/loom/src"
cp "$ROOT_DIR/tools/loom/dune-project" "$REPO/tools/loom/"
cp "$ROOT_DIR/tools/loom/continuity_adapter_main.sio" "$REPO/tools/loom/"
cp "$ROOT_DIR/tools/loom/src/dune" "$ROOT_DIR/tools/loom/src/loom.ml" \
  "$ROOT_DIR/tools/loom/src/loom_pty_stubs.c" "$REPO/tools/loom/src/"
mkdir -p "$REPO/stdlib/coordination"
cp "$ROOT_DIR/stdlib/coordination/loom_continuity.sio" \
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
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-fleet-agent-runtime" ]] || \
  fail 'installed runtime omitted the fleet launcher'
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-fleet-runtime" ]] || \
  fail 'installed runtime omitted the fleet reconciler'
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-fleet-tla-sabotage" ]] || \
  fail 'installed runtime omitted the model-derived sabotage generator'
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-fleet-trace-verify" ]] || \
  fail 'installed runtime omitted the independent trace verifier'
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
  loom-exclusive-input-lease-v1 loom-read-only-gui-v1 loom-coord-transport-v1 \
  coord-generation-scoped-wake-v1 \
  loom-recoverable-guardian-v1 loom-kernel-recovery-v1 loom-dual-journal-v1 \
  loom-persistent-fleet-catalog-v1 loom-post-pod-reconcile-v1 \
  fleet-linear-capability-v1 \
  fleet-home-isolation-v1 \
  fleet-proven-exit-v1 fleet-ed25519-anchor-v1 \
  fleet-checkpoint-handoff-v1 fleet-tla-model-v1 \
  fleet-trace-refinement-v1 fleet-temporal-authority-v1; do
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
  "$ALT/scripts/dev/"
mkdir -p "$ALT/tools/loom/src"
cp "$ROOT_DIR/tools/loom/dune-project" "$ALT/tools/loom/"
cp "$ROOT_DIR/tools/loom/continuity_adapter_main.sio" "$ALT/tools/loom/"
cp "$ROOT_DIR/tools/loom/src/dune" "$ROOT_DIR/tools/loom/src/loom.ml" \
  "$ROOT_DIR/tools/loom/src/loom_pty_stubs.c" "$ALT/tools/loom/src/"
mkdir -p "$ALT/stdlib/coordination"
cp "$ROOT_DIR/stdlib/coordination/loom_continuity.sio" \
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

output="$(cd "$REPO" && bin/sounio-coord install-runtime --activate "$first_id")"
grep -q "^ACTIVATED runtime_id=$first_id " <<< "$output" || fail 'runtime rollback failed'
output="$(cd "$SECOND" && bin/sounio-coord runtime-info)"
grep -q "^runtime_id=$first_id$" <<< "$output" || fail 'worktree did not observe runtime rollback'

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
  "$BAD/scripts/dev/"
mkdir -p "$BAD/tools/loom/src"
cp "$ROOT_DIR/tools/loom/dune-project" "$BAD/tools/loom/"
cp "$ROOT_DIR/tools/loom/continuity_adapter_main.sio" "$BAD/tools/loom/"
cp "$ROOT_DIR/tools/loom/src/dune" "$ROOT_DIR/tools/loom/src/loom.ml" \
  "$ROOT_DIR/tools/loom/src/loom_pty_stubs.c" "$BAD/tools/loom/src/"
mkdir -p "$BAD/stdlib/coordination"
cp "$ROOT_DIR/stdlib/coordination/loom_continuity.sio" \
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
