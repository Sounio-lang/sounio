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
export SOUNIO_LOOM_LANGUAGE_AUTHORITY_PREBUILT="$ROOT_DIR/tools/loom/.runtime/sounio-loom-language-authority-runtime"
export SOUNIO_LOOM_EXECUTION_AUTHORITY_PREBUILT="$ROOT_DIR/tools/loom/.runtime/sounio-loom-execution-authority-runtime"
export SOUNIO_LOOM_CUSTODY_TRANSFER_PREBUILT="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-custody-transfer-runtime"
export SOUNIO_LOOM_LANE_HEALTH_PREBUILT="$ROOT_DIR/tools/loom/.runtime/sounio-loom-lane-health-runtime"
export SOUNIO_LOOM_LANE_HEALTH_PARITY_PREBUILT="$ROOT_DIR/tools/loom/.runtime/sounio-loom-lane-health-parity-runtime"
export SOUNIO_LOOM_CONTINUITY_PREBUILT="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-continuity-runtime"
export SOUNIO_LOOM_OBLIGATION_PREBUILT="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-obligation-runtime"
export SOUNIO_LOOM_EPISTEMIC_PREBUILT="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-epistemic-runtime"
export SOUNIO_LOOM_ATTENTION_PREBUILT="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-attention-runtime"
export SOUNIO_LOOM_PORTFOLIO_PREBUILT="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-portfolio-runtime"
export SOUNIO_LOOM_CONTINGENT_PREBUILT="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-contingent-runtime"
export SOUNIO_LOOM_OUTCOME_AUTHORITY_PREBUILT="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-outcome-authority-runtime"
export SOUNIO_LOOM_WITNESS_MESH_PREBUILT="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-witness-mesh-runtime"
export SOUNIO_LOOM_WITNESS_MESH_V1_PREBUILT="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-witness-mesh-v1-runtime"
export SOUNIO_LOOM_WITNESS_EPOCH_HANDOFF_PREBUILT="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-witness-epoch-handoff-runtime"
export SOUNIO_LOOM_WITNESS_EPOCH_TRANSPARENCY_PREBUILT="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-witness-epoch-transparency-runtime"

cleanup() {
  [[ -z "${lock_holder_pid:-}" ]] || kill "$lock_holder_pid" 2>/dev/null || true
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

snapshot_coord_state() {
  (
    cd "$STATE"
    find . -mindepth 1 ! -name '.claims.lock' -printf '%P|%y|%s\n' | sort
    find . -type f ! -name '.claims.lock' -print0 | sort -z | \
      xargs -0 -r sha256sum
  )
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
  "$ROOT_DIR/scripts/dev/build_sounio_loom_language_authority.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_custody_transfer.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_lane_health.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_lane_health_parity.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_continuity_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_obligation_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_epistemic_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_attention_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_portfolio_attention_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_contingent_policy_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_outcome_authority_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_witness_mesh_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_witness_mesh_v1_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_witness_epoch_handoff_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_witness_epoch_transparency_adapter.sh" \
  "$REPO/scripts/dev/"
mkdir -p "$REPO/tools/loom/src"
cp "$ROOT_DIR/tools/loom/dune-project" "$REPO/tools/loom/"
cp "$ROOT_DIR/tools/loom/language_authority_main.sio" \
  "$ROOT_DIR/tools/loom/language_authority.freeze.v1" "$REPO/tools/loom/"
cp "$ROOT_DIR/tools/loom/custody_transfer_main.sio" \
  "$ROOT_DIR/tools/loom/custody_transfer.freeze.v1" "$REPO/tools/loom/"
cp "$ROOT_DIR/tools/loom/lane_health_main.sio" \
  "$ROOT_DIR/tools/loom/lane_health_parity_main.sio" \
  "$ROOT_DIR/tools/loom/lane_health.freeze.v1" \
  "$ROOT_DIR/tools/loom/lane_health.ocaml.v1" "$REPO/tools/loom/"
cp "$ROOT_DIR/tools/loom/continuity_adapter_main.sio" "$REPO/tools/loom/"
cp "$ROOT_DIR/tools/loom/obligation_adapter_main.sio" "$REPO/tools/loom/"
cp "$ROOT_DIR/tools/loom/epistemic_adapter_main.sio" "$REPO/tools/loom/"
cp "$ROOT_DIR/tools/loom/attention_adapter_main.sio" "$REPO/tools/loom/"
cp "$ROOT_DIR/tools/loom/portfolio_attention_adapter_main.sio" "$REPO/tools/loom/"
cp "$ROOT_DIR/tools/loom/contingent_policy_adapter_main.sio" "$REPO/tools/loom/"
cp "$ROOT_DIR/tools/loom/outcome_authority_adapter_main.sio" "$REPO/tools/loom/"
cp "$ROOT_DIR/tools/loom/witness_mesh_adapter_main.sio" "$REPO/tools/loom/"
cp "$ROOT_DIR/tools/loom/witness_mesh_v1_adapter_main.sio" "$REPO/tools/loom/"
cp "$ROOT_DIR/tools/loom/witness_epoch_handoff_adapter_main.sio" \
  "$REPO/tools/loom/"
cp "$ROOT_DIR/tools/loom/epoch_transparency_adapter_main.sio" \
  "$REPO/tools/loom/"
cp "$ROOT_DIR/tools/loom/src/dune" "$ROOT_DIR/tools/loom/src/loom.ml" \
  "$ROOT_DIR/tools/loom/src/loom_arrow.ml" \
  "$ROOT_DIR/tools/loom/src/loom_epistemic.ml" \
  "$ROOT_DIR/tools/loom/src/loom_exec.ml" \
  "$ROOT_DIR/tools/loom/src/loom_hook.ml" \
  "$ROOT_DIR/tools/loom/src/loom_lane_health.ml" \
  "$ROOT_DIR/tools/loom/src/loom_witness.ml" \
  "$ROOT_DIR/tools/loom/src/loom_witness_epoch.ml" \
  "$ROOT_DIR/tools/loom/src/loom_witness_transparency.ml" \
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
cp "$ROOT_DIR/stdlib/coordination/loom_language_authority.sio" \
  "$REPO/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_custody_transfer.sio" \
  "$REPO/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_lane_health.sio" \
  "$REPO/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_obligation.sio" \
  "$REPO/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_epistemic_machine.sio" \
  "$REPO/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_attention_compiler.sio" \
  "$REPO/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_portfolio_attention.sio" \
  "$REPO/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_contingent_policy.sio" \
  "$REPO/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_outcome_authority.sio" \
  "$REPO/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_witness_mesh.sio" \
  "$REPO/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_witness_mesh_v1.sio" \
  "$REPO/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_witness_epoch_handoff.sio" \
  "$REPO/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_witness_epoch_transparency.sio" \
  "$REPO/stdlib/coordination/"
mkdir -p "$REPO/stdlib/crypto"
cp "$ROOT_DIR/stdlib/crypto/sha256.sio" "$REPO/stdlib/crypto/"
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

# Lock contention waits for a bounded interval, then fails closed without
# changing coordination state.
SOUNIO_COORD_DIR="$STATE" SOUNIO_COORD_RUNTIME_MODE=local \
  "$REPO/bin/sounio-coord" brief >/dev/null
: > "$STATE/.claims.lock"
lock_snapshot_before="$(snapshot_coord_state)"
lock_ready="$TEST_ROOT/lock-ready"
(
  exec 9>"$STATE/.claims.lock"
  flock 9
  : > "$lock_ready"
  sleep 5
) &
lock_holder_pid=$!
for _ in $(seq 1 50); do
  [[ -f "$lock_ready" ]] && break
  sleep 0.02
done
[[ -f "$lock_ready" ]] || fail 'lock sabotage holder did not become ready'
lock_started_ns="$(date +%s%N)"
set +e
lock_output="$(cd "$REPO" && SOUNIO_COORD_DIR="$STATE" \
  SOUNIO_COORD_RUNTIME_MODE=local SOUNIO_COORD_LOCK_WAIT_SECONDS=0.2 \
  bin/sounio-coord claim --agent lock-test --lane held \
  --intent 'held lock must fail closed' --files lock-sabotage.test 2>&1)"
lock_rc=$?
set -e
lock_elapsed_ms=$((($(date +%s%N) - lock_started_ns) / 1000000))
kill "$lock_holder_pid" 2>/dev/null || true
wait "$lock_holder_pid" 2>/dev/null || true
lock_holder_pid=''
[[ "$lock_rc" -ne 0 ]] || fail 'held lock sabotage was allowed'
grep -q 'coordination state is being changed; retry the claim' <<< "$lock_output" || \
  fail "held lock refusal omitted its retry reason: $lock_output"
((lock_elapsed_ms >= 100 && lock_elapsed_ms < 1500)) || \
  fail "held lock timeout was not bounded: ${lock_elapsed_ms}ms"
lock_snapshot_after="$(snapshot_coord_state)"
[[ "$lock_snapshot_after" == "$lock_snapshot_before" ]] || \
  fail 'held lock refusal mutated coordination state'

output="$(cd "$REPO" && bin/sounio-coord install-runtime)"
first_id="$(sed -n 's/^INSTALLED runtime_id=\([^ ]*\).*/\1/p' <<< "$output")"
[[ -n "$first_id" ]] || fail 'installer did not return the first runtime id'
grep -q "^ACTIVATED runtime_id=$first_id " <<< "$output" || fail 'first runtime was not activated'
first_manifest="$RUNTIME_ROOT/versions/$first_id/manifest"
first_source_sha="$(git -C "$REPO" rev-parse --short=12 HEAD)"
grep -q "^source_sha=$first_source_sha$" "$first_manifest" || \
  fail 'runtime manifest source SHA does not identify the committed source'
grep -q '^source_state=clean$' "$first_manifest" || \
  fail 'runtime manifest omitted clean source provenance'
coord_runtime_sha="$(sha256sum "$RUNTIME_ROOT/versions/$first_id/bin/sounio-coord-runtime" | awk '{print $1}')"
loom_runtime_sha="$(sha256sum "$RUNTIME_ROOT/versions/$first_id/bin/sounio-loom-runtime" | awk '{print $1}')"
loom_custody_transfer_sha="$(sha256sum "$RUNTIME_ROOT/versions/$first_id/bin/sounio-loom-custody-transfer-runtime" | awk '{print $1}')"
grep -qx "coord_runtime_sha256=$coord_runtime_sha" "$first_manifest" || \
  fail 'runtime manifest did not pin the coordination runtime executable'
grep -qx "loom_runtime_sha256=$loom_runtime_sha" "$first_manifest" || \
  fail 'runtime manifest did not pin the compiled OCaml Loom executable'
grep -qx "loom_custody_transfer_runtime_sha256=$loom_custody_transfer_sha" \
  "$first_manifest" || \
  fail 'runtime manifest did not pin the frozen Sounio custody-transfer executable'
grep -q '^capability=loom-native-hook-binary-attestation-v1$' "$first_manifest" || \
  fail 'runtime manifest omitted native hook binary attestation'

active_before_tamper="$(readlink -f "$RUNTIME_ROOT/current")"
for tamper_binary in sounio-coord-runtime sounio-loom-runtime \
  sounio-loom-custody-transfer-runtime; do
  binary="$RUNTIME_ROOT/versions/$first_id/bin/$tamper_binary"
  saved_binary="$TEST_ROOT/$tamper_binary.saved"
  cp -p "$binary" "$saved_binary"
  printf x >> "$binary"
  set +e
  tamper_output="$(cd "$REPO" && scripts/dev/install_sounio_coord_runtime.sh \
    --runtime-dir "$RUNTIME_ROOT" --activate "$first_id" 2>&1)"
  tamper_rc=$?
  set -e
  [[ "$tamper_rc" -ne 0 && "$tamper_output" == *'binary hash mismatch'* ]] || \
    fail "activation accepted tampered $tamper_binary: rc=$tamper_rc output=$tamper_output"
  [[ "$(readlink -f "$RUNTIME_ROOT/current")" == "$active_before_tamper" ]] || \
    fail 'failed binary activation changed the current runtime link'
  cp -p "$saved_binary" "$binary"
done
git -C "$REPO" ls-tree -r --name-only "$first_source_sha" | \
  grep -qx 'stdlib/coordination/loom_witness_epoch_handoff.sio' || \
  fail 'runtime source SHA omits the frame-9015 source'
git -C "$REPO" ls-tree -r --name-only "$first_source_sha" | \
  grep -qx 'stdlib/coordination/loom_witness_epoch_transparency.sio' || \
  fail 'runtime source SHA omits the frame-9016 source'
git -C "$REPO" ls-tree -r --name-only "$first_source_sha" | \
  grep -qx 'stdlib/coordination/loom_custody_transfer.sio' || \
  fail 'runtime source SHA omits the frame-9040 source'

printf '\n# dirty runtime source control\n' >> \
  "$REPO/stdlib/coordination/loom_witness_epoch_handoff.sio"
set +e
dirty_output="$(cd "$REPO" && bin/sounio-coord install-runtime 2>&1)"
dirty_rc=$?
set -e
[[ "$dirty_rc" -ne 0 ]] || fail 'installer accepted a dirty runtime source bundle'
grep -q 'runtime source bundle has uncommitted changes' <<< "$dirty_output" || \
  fail 'dirty runtime source refusal omitted its provenance reason'
git -C "$REPO" show HEAD:stdlib/coordination/loom_witness_epoch_handoff.sio > \
  "$REPO/stdlib/coordination/loom_witness_epoch_handoff.sio"
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-coord-causal-runtime" ]] || \
  fail 'installed runtime omitted the causal receipt verifier'
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-agentd-runtime" ]] || \
  fail 'installed runtime omitted the detached agent supervisor'
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-loom-runtime" ]] || \
  fail 'installed runtime omitted the OCaml Loom kernel'
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-loom-language-authority-runtime" ]] || \
  fail 'installed runtime omitted the frozen Sounio language authority'
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-loom-custody-transfer-runtime" ]] || \
  fail 'installed runtime omitted the frozen Sounio custody-transfer authority'
grep -q '^loom_custody_transfer_semantics_sha256=5f53d3edcb6731c5b0f4e58ff7b27d251e6c0b40eda8c68366e48b17e596f55c$' \
  "$first_manifest" || \
  fail 'installed runtime omitted frozen custody-transfer semantics'
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-loom-lane-health-runtime" && \
  -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-loom-lane-health-parity-runtime" ]] || \
  fail 'installed runtime omitted the frozen Sounio lane-health executables'
grep -q '^loom_lane_health_semantics_sha256=5eb48f9cb214f6018569fb24e1e419b3e800dccde2e6e8d775246f4c05e4c93f$' \
  "$first_manifest" || fail 'installed runtime omitted frozen lane-health semantics'
grep -q '^capability=loom-native-agent-hook-v1$' "$first_manifest" || \
  fail 'installed runtime omitted the native-agent-hook capability'
grep -q '^loom_language_authority_semantics_sha256=16e283166d29d6b18ed690b000e2eb595a7d965e4357553a8380714486429fff$' \
  "$first_manifest" || fail 'installed native hook is not bound to frozen Sounio semantics'
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-loom-continuity-runtime" ]] || \
  fail 'installed runtime omitted the native Sounio continuity adapter'
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-loom-obligation-runtime" ]] || \
  fail 'installed runtime omitted the native Sounio obligation adapter'
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-loom-epistemic-runtime" ]] || \
  fail 'installed runtime omitted the native Sounio epistemic adapter'
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-loom-attention-runtime" ]] || \
  fail 'installed runtime omitted the native Sounio attention adapter'
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-loom-portfolio-runtime" ]] || \
  fail 'installed runtime omitted the native Sounio portfolio adapter'
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-loom-contingent-runtime" ]] || \
  fail 'installed runtime omitted the native Sounio contingent-policy adapter'
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-loom-outcome-authority-runtime" ]] || \
  fail 'installed runtime omitted the native Sounio outcome-authority adapter'
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-loom-witness-mesh-runtime" ]] || \
  fail 'installed runtime omitted the native Sounio witness-mesh adapter'
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-loom-witness-mesh-v1-runtime" ]] || \
  fail 'installed runtime omitted the native Sounio witness-mesh-v1 adapter'
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-loom-witness-epoch-handoff-runtime" ]] || \
  fail 'installed runtime omitted the native Sounio witness-epoch-handoff adapter'
[[ -x "$RUNTIME_ROOT/versions/$first_id/bin/sounio-loom-witness-epoch-transparency-runtime" ]] || \
  fail 'installed runtime omitted the native Sounio witness-epoch-transparency adapter'
grep -q '^loom_witness_mesh_language=Sounio$' \
  "$RUNTIME_ROOT/versions/$first_id/manifest" || \
  fail 'installed runtime omitted the witness-mesh language declaration'
grep -q '^loom_witness_mesh_frame=9013$' \
  "$RUNTIME_ROOT/versions/$first_id/manifest" || \
  fail 'installed runtime omitted the witness-mesh frame declaration'
grep -q '^loom_witness_mesh_v1_language=Sounio$' \
  "$RUNTIME_ROOT/versions/$first_id/manifest" || \
  fail 'installed runtime omitted the witness-mesh-v1 language declaration'
grep -q '^loom_witness_mesh_v1_frame=9014$' \
  "$RUNTIME_ROOT/versions/$first_id/manifest" || \
  fail 'installed runtime omitted the witness-mesh-v1 frame declaration'
grep -q '^loom_witness_epoch_handoff_language=Sounio$' \
  "$RUNTIME_ROOT/versions/$first_id/manifest" || \
  fail 'installed runtime omitted the witness-epoch-handoff language declaration'
grep -q '^loom_witness_epoch_handoff_frame=9015$' \
  "$RUNTIME_ROOT/versions/$first_id/manifest" || \
  fail 'installed runtime omitted the witness-epoch-handoff frame declaration'
grep -q '^loom_witness_epoch_transparency_language=Sounio$' \
  "$RUNTIME_ROOT/versions/$first_id/manifest" || \
  fail 'installed runtime omitted the witness-epoch-transparency language declaration'
grep -q '^loom_witness_epoch_transparency_frame=9016$' \
  "$RUNTIME_ROOT/versions/$first_id/manifest" || \
  fail 'installed runtime omitted the witness-epoch-transparency frame declaration'
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
  loom-native-hook-binary-attestation-v1 \
  loom-transactional-custody-transfer-v1 \
  loom-truthful-lane-health-v1 loom-nondestructive-health-reconcile-v1 \
  loom-native-sounio-continuity-v1 \
  loom-durable-obligation-v1 \
  loom-epistemic-machine-v0 loom-epistemic-arrow-projection-v0 \
  loom-attention-compiler-v0 loom-attention-linear-resource-v0 \
  loom-pareto-portfolio-attention-v0 loom-atomic-multi-resource-attention-v0 \
  loom-robust-contingent-policy-v0 loom-atomic-outcome-resource-handoff-v0 \
  loom-signed-outcome-authority-v0 loom-linear-outcome-evidence-v0 \
  loom-journal-head-bound-consume-v0 \
  loom-external-witness-mesh-v0 loom-quorum-intersection-checkpoint-v0 \
  loom-rollback-detection-through-checkpoint-v0 \
  loom-external-witness-mesh-v1 loom-three-of-four-witness-quorum-v1 \
  loom-one-dishonest-honest-intersection-v1 \
  loom-one-fault-anchor-and-verify-availability-v1 \
  loom-proof-carrying-witness-epoch-handoff-v0 \
  loom-joint-old-new-witness-quorum-v0 \
  loom-atomic-witness-epoch-activation-v0 \
  loom-witness-epoch-crash-recovery-v0 \
  loom-external-epoch-transparency-v0 \
  loom-materialized-merkle-prefix-verification-v0 \
  loom-witnessed-split-view-refusal-v0 \
  loom-latest-quorum-witnessed-epoch-rollback-refusal-v0 \
  loom-transparency-unreachable-fail-closed-v0 \
  loom-post-activation-request-bridge-v1 \
  loom-recoverable-control-service-v1 \
  loom-beagle-coordination-endpoint-v1 loom-separate-pod-inbox-replay-v1 \
  loom-signed-continuity-receipt-v2 loom-principal-independence-v1 \
  loom-independent-measurement-v1 \
  loom-observation-authority-v1 \
  loom-journal-authority-quorum-v1 \
  loom-cross-node-replay-v1 \
  loom-exclusive-input-lease-v1 loom-read-only-gui-v1 \
  loom-fusion-cockpit-v1 loom-authority-overlay-v1 loom-authority-overlay-v2 \
  coord-cockpit-snapshot-v1 loom-persistent-provider-custody-v1 \
  coord-reply-command-v1 loom-coord-transport-v1 \
  coord-generation-scoped-wake-v1 \
  loom-recoverable-guardian-v1 loom-kernel-recovery-v1 loom-dual-journal-v1 \
  loom-persistent-fleet-catalog-v1 loom-post-pod-reconcile-v1 \
  loom-fleet-custody-catalog-v2 loom-fleet-custody-catalog-v3 \
  loom-conflict-free-active-adoption-v1 \
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
grep -q '^runtime_version=2026.08.27.35$' <<< "$output" || \
  fail 'shared Loom kernel version diverged from its runtime bundle'
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
set +e
output="$(cd "$SECOND" && SOUNIO_COORD_DIR="$STATE" timeout 3 \
  bin/sounio-coord obligation-supervise --interval-seconds 1 2>&1)"
duplicate_supervisor_status=$?
set -e
((duplicate_supervisor_status == 73)) && \
  grep -q 'state=duplicate-leader' <<< "$output" || \
  fail 'raw supervisor start was not fenced by the lifetime leader lock'
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
cp "$ROOT_DIR/scripts/dev/install_sounio_coord_runtime.sh" "$ALT/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_runtime.sh" "$ALT/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_agent_hook_runtime.py" "$ALT/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_causal_runtime.py" "$ALT/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_agentd.py" "$ALT/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_fleet.py" "$ALT/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_fleetd.py" "$ALT/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_fleet_tla_sabotage.py" "$ALT/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_fleet_trace_verify.py" "$ALT/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/build_sounio_loom.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_language_authority.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_custody_transfer.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_lane_health.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_lane_health_parity.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_continuity_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_obligation_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_epistemic_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_attention_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_portfolio_attention_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_contingent_policy_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_outcome_authority_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_witness_mesh_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_witness_mesh_v1_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_witness_epoch_handoff_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_witness_epoch_transparency_adapter.sh" \
  "$ALT/scripts/dev/"
mkdir -p "$ALT/tools/loom/src"
cp "$ROOT_DIR/tools/loom/dune-project" "$ALT/tools/loom/"
cp "$ROOT_DIR/tools/loom/language_authority_main.sio" \
  "$ROOT_DIR/tools/loom/language_authority.freeze.v1" "$ALT/tools/loom/"
cp "$ROOT_DIR/tools/loom/custody_transfer_main.sio" \
  "$ROOT_DIR/tools/loom/custody_transfer.freeze.v1" "$ALT/tools/loom/"
cp "$ROOT_DIR/tools/loom/lane_health_main.sio" \
  "$ROOT_DIR/tools/loom/lane_health_parity_main.sio" \
  "$ROOT_DIR/tools/loom/lane_health.freeze.v1" \
  "$ROOT_DIR/tools/loom/lane_health.ocaml.v1" "$ALT/tools/loom/"
cp "$ROOT_DIR/tools/loom/continuity_adapter_main.sio" "$ALT/tools/loom/"
cp "$ROOT_DIR/tools/loom/obligation_adapter_main.sio" "$ALT/tools/loom/"
cp "$ROOT_DIR/tools/loom/epistemic_adapter_main.sio" "$ALT/tools/loom/"
cp "$ROOT_DIR/tools/loom/attention_adapter_main.sio" "$ALT/tools/loom/"
cp "$ROOT_DIR/tools/loom/portfolio_attention_adapter_main.sio" "$ALT/tools/loom/"
cp "$ROOT_DIR/tools/loom/contingent_policy_adapter_main.sio" "$ALT/tools/loom/"
cp "$ROOT_DIR/tools/loom/outcome_authority_adapter_main.sio" "$ALT/tools/loom/"
cp "$ROOT_DIR/tools/loom/witness_mesh_adapter_main.sio" "$ALT/tools/loom/"
cp "$ROOT_DIR/tools/loom/witness_mesh_v1_adapter_main.sio" "$ALT/tools/loom/"
cp "$ROOT_DIR/tools/loom/witness_epoch_handoff_adapter_main.sio" \
  "$ALT/tools/loom/"
cp "$ROOT_DIR/tools/loom/epoch_transparency_adapter_main.sio" \
  "$ALT/tools/loom/"
cp "$ROOT_DIR/tools/loom/src/dune" "$ROOT_DIR/tools/loom/src/loom.ml" \
  "$ROOT_DIR/tools/loom/src/loom_arrow.ml" \
  "$ROOT_DIR/tools/loom/src/loom_epistemic.ml" \
  "$ROOT_DIR/tools/loom/src/loom_exec.ml" \
  "$ROOT_DIR/tools/loom/src/loom_hook.ml" \
  "$ROOT_DIR/tools/loom/src/loom_lane_health.ml" \
  "$ROOT_DIR/tools/loom/src/loom_witness.ml" \
  "$ROOT_DIR/tools/loom/src/loom_witness_epoch.ml" \
  "$ROOT_DIR/tools/loom/src/loom_witness_transparency.ml" \
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
cp "$ROOT_DIR/stdlib/coordination/loom_language_authority.sio" \
  "$ALT/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_custody_transfer.sio" \
  "$ALT/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_lane_health.sio" \
  "$ALT/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_obligation.sio" \
  "$ALT/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_epistemic_machine.sio" \
  "$ALT/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_attention_compiler.sio" \
  "$ALT/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_portfolio_attention.sio" \
  "$ALT/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_contingent_policy.sio" \
  "$ALT/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_outcome_authority.sio" \
  "$ALT/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_witness_mesh.sio" \
  "$ALT/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_witness_mesh_v1.sio" \
  "$ALT/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_witness_epoch_handoff.sio" \
  "$ALT/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_witness_epoch_transparency.sio" \
  "$ALT/stdlib/coordination/"
mkdir -p "$ALT/stdlib/crypto"
cp "$ROOT_DIR/stdlib/crypto/sha256.sio" "$ALT/stdlib/crypto/"
cp "$ROOT_DIR/formal/tla/SounioFleet.tla" "$ROOT_DIR/formal/tla/SounioFleet.cfg" \
  "$ALT/formal/tla/"
sed -i 's/^SOUNIO_COORD_RUNTIME_VERSION=.*/SOUNIO_COORD_RUNTIME_VERSION=2026.08.23.8-test/' \
  "$ALT/scripts/dev/sounio_coord_runtime.sh"
sed -i 's/^let runtime_version = .*/let runtime_version = "2026.08.23.8-test"/' \
  "$ALT/tools/loom/src/loom.ml"
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
cp "$ROOT_DIR/scripts/dev/install_sounio_coord_runtime.sh" "$BAD/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_agentd.py" "$BAD/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_fleet.py" "$BAD/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_coord_fleetd.py" "$BAD/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_fleet_tla_sabotage.py" "$BAD/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/sounio_fleet_trace_verify.py" "$BAD/scripts/dev/"
cp "$ROOT_DIR/scripts/dev/build_sounio_loom.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_language_authority.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_custody_transfer.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_lane_health.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_lane_health_parity.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_continuity_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_obligation_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_epistemic_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_attention_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_portfolio_attention_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_contingent_policy_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_outcome_authority_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_witness_mesh_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_witness_mesh_v1_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_witness_epoch_handoff_adapter.sh" \
  "$ROOT_DIR/scripts/dev/build_sounio_loom_witness_epoch_transparency_adapter.sh" \
  "$BAD/scripts/dev/"
mkdir -p "$BAD/tools/loom/src"
cp "$ROOT_DIR/tools/loom/dune-project" "$BAD/tools/loom/"
cp "$ROOT_DIR/tools/loom/language_authority_main.sio" \
  "$ROOT_DIR/tools/loom/language_authority.freeze.v1" "$BAD/tools/loom/"
cp "$ROOT_DIR/tools/loom/custody_transfer_main.sio" \
  "$ROOT_DIR/tools/loom/custody_transfer.freeze.v1" "$BAD/tools/loom/"
cp "$ROOT_DIR/tools/loom/lane_health_main.sio" \
  "$ROOT_DIR/tools/loom/lane_health_parity_main.sio" \
  "$ROOT_DIR/tools/loom/lane_health.freeze.v1" \
  "$ROOT_DIR/tools/loom/lane_health.ocaml.v1" "$BAD/tools/loom/"
cp "$ROOT_DIR/tools/loom/continuity_adapter_main.sio" "$BAD/tools/loom/"
cp "$ROOT_DIR/tools/loom/obligation_adapter_main.sio" "$BAD/tools/loom/"
cp "$ROOT_DIR/tools/loom/epistemic_adapter_main.sio" "$BAD/tools/loom/"
cp "$ROOT_DIR/tools/loom/attention_adapter_main.sio" "$BAD/tools/loom/"
cp "$ROOT_DIR/tools/loom/portfolio_attention_adapter_main.sio" "$BAD/tools/loom/"
cp "$ROOT_DIR/tools/loom/contingent_policy_adapter_main.sio" "$BAD/tools/loom/"
cp "$ROOT_DIR/tools/loom/outcome_authority_adapter_main.sio" "$BAD/tools/loom/"
cp "$ROOT_DIR/tools/loom/witness_mesh_adapter_main.sio" "$BAD/tools/loom/"
cp "$ROOT_DIR/tools/loom/witness_mesh_v1_adapter_main.sio" "$BAD/tools/loom/"
cp "$ROOT_DIR/tools/loom/witness_epoch_handoff_adapter_main.sio" \
  "$BAD/tools/loom/"
cp "$ROOT_DIR/tools/loom/epoch_transparency_adapter_main.sio" \
  "$BAD/tools/loom/"
cp "$ROOT_DIR/tools/loom/src/dune" "$ROOT_DIR/tools/loom/src/loom.ml" \
  "$ROOT_DIR/tools/loom/src/loom_arrow.ml" \
  "$ROOT_DIR/tools/loom/src/loom_epistemic.ml" \
  "$ROOT_DIR/tools/loom/src/loom_hook.ml" \
  "$ROOT_DIR/tools/loom/src/loom_lane_health.ml" \
  "$ROOT_DIR/tools/loom/src/loom_witness.ml" \
  "$ROOT_DIR/tools/loom/src/loom_witness_epoch.ml" \
  "$ROOT_DIR/tools/loom/src/loom_witness_transparency.ml" \
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
cp "$ROOT_DIR/stdlib/coordination/loom_language_authority.sio" \
  "$BAD/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_custody_transfer.sio" \
  "$BAD/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_lane_health.sio" \
  "$BAD/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_obligation.sio" \
  "$BAD/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_epistemic_machine.sio" \
  "$BAD/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_attention_compiler.sio" \
  "$BAD/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_portfolio_attention.sio" \
  "$BAD/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_contingent_policy.sio" \
  "$BAD/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_outcome_authority.sio" \
  "$BAD/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_witness_mesh.sio" \
  "$BAD/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_witness_mesh_v1.sio" \
  "$BAD/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_witness_epoch_handoff.sio" \
  "$BAD/stdlib/coordination/"
cp "$ROOT_DIR/stdlib/coordination/loom_witness_epoch_transparency.sio" \
  "$BAD/stdlib/coordination/"
mkdir -p "$BAD/stdlib/crypto"
cp "$ROOT_DIR/stdlib/crypto/sha256.sio" "$BAD/stdlib/crypto/"
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
cp -a "$RUNTIME_ROOT/versions/$first_id" \
  "$RUNTIME_ROOT/versions/lane-health-authority-omitted"
sed -i 's/^runtime_id=.*/runtime_id=lane-health-authority-omitted/' \
  "$RUNTIME_ROOT/versions/lane-health-authority-omitted/manifest"
rm -f \
  "$RUNTIME_ROOT/versions/lane-health-authority-omitted/bin/sounio-loom-lane-health-runtime"
if (cd "$REPO" && bin/sounio-coord install-runtime \
    --activate lane-health-authority-omitted) >/dev/null 2>&1; then
  fail 'installer activated truthful lane health without its Sounio authority executable'
fi
output="$(cd "$REPO" && bin/sounio-coord runtime-info)"
grep -q "^runtime_id=$first_id$" <<< "$output" || \
  fail 'failed lane-health activation changed the current runtime'
cp -a "$RUNTIME_ROOT/versions/$first_id" \
  "$RUNTIME_ROOT/versions/portfolio-adapter-omitted"
sed -i 's/^runtime_id=.*/runtime_id=portfolio-adapter-omitted/' \
  "$RUNTIME_ROOT/versions/portfolio-adapter-omitted/manifest"
rm -f "$RUNTIME_ROOT/versions/portfolio-adapter-omitted/bin/sounio-loom-portfolio-runtime"
if (cd "$REPO" && bin/sounio-coord install-runtime \
    --activate portfolio-adapter-omitted) >/dev/null 2>&1; then
  fail 'installer activated a declared frame-9010 runtime without its adapter'
fi
output="$(cd "$REPO" && bin/sounio-coord runtime-info)"
grep -q "^runtime_id=$first_id$" <<< "$output" || fail 'failed activation changed the current runtime'

cp -a "$RUNTIME_ROOT/versions/$first_id" \
  "$RUNTIME_ROOT/versions/contingent-adapter-omitted"
sed -i 's/^runtime_id=.*/runtime_id=contingent-adapter-omitted/' \
  "$RUNTIME_ROOT/versions/contingent-adapter-omitted/manifest"
rm -f "$RUNTIME_ROOT/versions/contingent-adapter-omitted/bin/sounio-loom-contingent-runtime"
if (cd "$REPO" && bin/sounio-coord install-runtime \
    --activate contingent-adapter-omitted) >/dev/null 2>&1; then
  fail 'installer activated a declared frame-9011 runtime without its adapter'
fi
output="$(cd "$REPO" && bin/sounio-coord runtime-info)"
grep -q "^runtime_id=$first_id$" <<< "$output" || \
  fail 'failed contingent activation changed the current runtime'

cp -a "$RUNTIME_ROOT/versions/$first_id" \
  "$RUNTIME_ROOT/versions/outcome-authority-adapter-omitted"
sed -i 's/^runtime_id=.*/runtime_id=outcome-authority-adapter-omitted/' \
  "$RUNTIME_ROOT/versions/outcome-authority-adapter-omitted/manifest"
rm -f "$RUNTIME_ROOT/versions/outcome-authority-adapter-omitted/bin/sounio-loom-outcome-authority-runtime"
if (cd "$REPO" && bin/sounio-coord install-runtime \
    --activate outcome-authority-adapter-omitted) >/dev/null 2>&1; then
  fail 'installer activated a declared frame-9012 runtime without its adapter'
fi
output="$(cd "$REPO" && bin/sounio-coord runtime-info)"
grep -q "^runtime_id=$first_id$" <<< "$output" || \
  fail 'failed outcome-authority activation changed the current runtime'

cp -a "$RUNTIME_ROOT/versions/$first_id" \
  "$RUNTIME_ROOT/versions/outcome-authority-root-omitted"
sed -i 's/^runtime_id=.*/runtime_id=outcome-authority-root-omitted/' \
  "$RUNTIME_ROOT/versions/outcome-authority-root-omitted/manifest"
sed -i '/^capability=loom-signed-outcome-authority-v0$/d' \
  "$RUNTIME_ROOT/versions/outcome-authority-root-omitted/manifest"
if (cd "$REPO" && bin/sounio-coord install-runtime \
    --activate outcome-authority-root-omitted) >/dev/null 2>&1; then
  fail 'installer activated derived outcome-evidence capabilities without their root capability'
fi
output="$(cd "$REPO" && bin/sounio-coord runtime-info)"
grep -q "^runtime_id=$first_id$" <<< "$output" || \
  fail 'failed outcome-authority dependency activation changed the current runtime'

cp -a "$RUNTIME_ROOT/versions/$first_id" \
  "$RUNTIME_ROOT/versions/witness-mesh-adapter-omitted"
sed -i 's/^runtime_id=.*/runtime_id=witness-mesh-adapter-omitted/' \
  "$RUNTIME_ROOT/versions/witness-mesh-adapter-omitted/manifest"
rm -f "$RUNTIME_ROOT/versions/witness-mesh-adapter-omitted/bin/sounio-loom-witness-mesh-runtime"
if (cd "$REPO" && bin/sounio-coord install-runtime \
    --activate witness-mesh-adapter-omitted) >/dev/null 2>&1; then
  fail 'installer activated a declared frame-9013 runtime without its adapter'
fi
output="$(cd "$REPO" && bin/sounio-coord runtime-info)"
grep -q "^runtime_id=$first_id$" <<< "$output" || \
  fail 'failed witness-mesh activation changed the current runtime'

cp -a "$RUNTIME_ROOT/versions/$first_id" \
  "$RUNTIME_ROOT/versions/witness-mesh-root-omitted"
sed -i 's/^runtime_id=.*/runtime_id=witness-mesh-root-omitted/' \
  "$RUNTIME_ROOT/versions/witness-mesh-root-omitted/manifest"
sed -i '/^capability=loom-external-witness-mesh-v0$/d' \
  "$RUNTIME_ROOT/versions/witness-mesh-root-omitted/manifest"
if (cd "$REPO" && bin/sounio-coord install-runtime \
    --activate witness-mesh-root-omitted) >/dev/null 2>&1; then
  fail 'installer activated derived witness-mesh capabilities without their root capability'
fi
output="$(cd "$REPO" && bin/sounio-coord runtime-info)"
grep -q "^runtime_id=$first_id$" <<< "$output" || \
  fail 'failed witness-mesh dependency activation changed the current runtime'

cp -a "$RUNTIME_ROOT/versions/$first_id" \
  "$RUNTIME_ROOT/versions/witness-mesh-v1-adapter-omitted"
sed -i 's/^runtime_id=.*/runtime_id=witness-mesh-v1-adapter-omitted/' \
  "$RUNTIME_ROOT/versions/witness-mesh-v1-adapter-omitted/manifest"
rm -f "$RUNTIME_ROOT/versions/witness-mesh-v1-adapter-omitted/bin/sounio-loom-witness-mesh-v1-runtime"
if (cd "$REPO" && bin/sounio-coord install-runtime \
    --activate witness-mesh-v1-adapter-omitted) >/dev/null 2>&1; then
  fail 'installer activated a declared frame-9014 runtime without its adapter'
fi
output="$(cd "$REPO" && bin/sounio-coord runtime-info)"
grep -q "^runtime_id=$first_id$" <<< "$output" || \
  fail 'failed witness-mesh-v1 activation changed the current runtime'

cp -a "$RUNTIME_ROOT/versions/$first_id" \
  "$RUNTIME_ROOT/versions/witness-mesh-v1-root-omitted"
sed -i 's/^runtime_id=.*/runtime_id=witness-mesh-v1-root-omitted/' \
  "$RUNTIME_ROOT/versions/witness-mesh-v1-root-omitted/manifest"
sed -i '/^capability=loom-external-witness-mesh-v1$/d' \
  "$RUNTIME_ROOT/versions/witness-mesh-v1-root-omitted/manifest"
if (cd "$REPO" && bin/sounio-coord install-runtime \
    --activate witness-mesh-v1-root-omitted) >/dev/null 2>&1; then
  fail 'installer activated derived witness-mesh-v1 capabilities without their root capability'
fi
output="$(cd "$REPO" && bin/sounio-coord runtime-info)"
grep -q "^runtime_id=$first_id$" <<< "$output" || \
  fail 'failed witness-mesh-v1 dependency activation changed the current runtime'

cp -a "$RUNTIME_ROOT/versions/$first_id" \
  "$RUNTIME_ROOT/versions/witness-epoch-handoff-adapter-omitted"
sed -i 's/^runtime_id=.*/runtime_id=witness-epoch-handoff-adapter-omitted/' \
  "$RUNTIME_ROOT/versions/witness-epoch-handoff-adapter-omitted/manifest"
rm -f "$RUNTIME_ROOT/versions/witness-epoch-handoff-adapter-omitted/bin/sounio-loom-witness-epoch-handoff-runtime"
if (cd "$REPO" && bin/sounio-coord install-runtime \
    --activate witness-epoch-handoff-adapter-omitted) >/dev/null 2>&1; then
  fail 'installer activated a declared frame-9015 runtime without its adapter'
fi
output="$(cd "$REPO" && bin/sounio-coord runtime-info)"
grep -q "^runtime_id=$first_id$" <<< "$output" || \
  fail 'failed witness-epoch-handoff activation changed the current runtime'

cp -a "$RUNTIME_ROOT/versions/$first_id" \
  "$RUNTIME_ROOT/versions/witness-epoch-handoff-root-omitted"
sed -i 's/^runtime_id=.*/runtime_id=witness-epoch-handoff-root-omitted/' \
  "$RUNTIME_ROOT/versions/witness-epoch-handoff-root-omitted/manifest"
sed -i '/^capability=loom-proof-carrying-witness-epoch-handoff-v0$/d' \
  "$RUNTIME_ROOT/versions/witness-epoch-handoff-root-omitted/manifest"
if (cd "$REPO" && bin/sounio-coord install-runtime \
    --activate witness-epoch-handoff-root-omitted) >/dev/null 2>&1; then
  fail 'installer activated derived witness-epoch capabilities without their root capability'
fi
output="$(cd "$REPO" && bin/sounio-coord runtime-info)"
grep -q "^runtime_id=$first_id$" <<< "$output" || \
  fail 'failed witness-epoch dependency activation changed the current runtime'

cp -a "$RUNTIME_ROOT/versions/$first_id" \
  "$RUNTIME_ROOT/versions/witness-epoch-transparency-adapter-omitted"
sed -i 's/^runtime_id=.*/runtime_id=witness-epoch-transparency-adapter-omitted/' \
  "$RUNTIME_ROOT/versions/witness-epoch-transparency-adapter-omitted/manifest"
rm -f "$RUNTIME_ROOT/versions/witness-epoch-transparency-adapter-omitted/bin/sounio-loom-witness-epoch-transparency-runtime"
if (cd "$REPO" && bin/sounio-coord install-runtime \
    --activate witness-epoch-transparency-adapter-omitted) >/dev/null 2>&1; then
  fail 'installer activated a declared frame-9016 runtime without its adapter'
fi
output="$(cd "$REPO" && bin/sounio-coord runtime-info)"
grep -q "^runtime_id=$first_id$" <<< "$output" || \
  fail 'failed witness-epoch-transparency activation changed the current runtime'

cp -a "$RUNTIME_ROOT/versions/$first_id" \
  "$RUNTIME_ROOT/versions/witness-epoch-transparency-root-omitted"
sed -i 's/^runtime_id=.*/runtime_id=witness-epoch-transparency-root-omitted/' \
  "$RUNTIME_ROOT/versions/witness-epoch-transparency-root-omitted/manifest"
sed -i '/^capability=loom-external-epoch-transparency-v0$/d' \
  "$RUNTIME_ROOT/versions/witness-epoch-transparency-root-omitted/manifest"
if (cd "$REPO" && bin/sounio-coord install-runtime \
    --activate witness-epoch-transparency-root-omitted) >/dev/null 2>&1; then
  fail 'installer activated derived transparency capabilities without their root capability'
fi
output="$(cd "$REPO" && bin/sounio-coord runtime-info)"
grep -q "^runtime_id=$first_id$" <<< "$output" || \
  fail 'failed transparency dependency activation changed the current runtime'

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
