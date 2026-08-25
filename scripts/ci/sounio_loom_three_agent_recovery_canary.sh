#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
RUN_ID="${SOUNIO_LOOM_CANARY_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RUN_ROOT="${SOUNIO_LOOM_CANARY_ROOT:-$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-three-agent.${RUN_ID}.XXXXXX")}"
STATE_DIR="$RUN_ROOT/state"
EVIDENCE_DIR="$RUN_ROOT/evidence"
LOOM="$ROOT_DIR/bin/loom"
WAIT_SECONDS="${SOUNIO_LOOM_CANARY_WAIT_SECONDS:-240}"
SLEEP_SECONDS="${SOUNIO_LOOM_CANARY_AGENT_SLEEP_SECONDS:-12}"

AGENTS=(codex grok minimax)
declare -A lanes workdirs receipts tokens
declare -A kernels_before kernels_after guardians harnesses instances
declare -A providers models
lanes[codex]='real-codex'
lanes[grok]='real-grok'
lanes[minimax]='real-minimax'
providers[codex]='codex'
providers[grok]='grok'
providers[minimax]='opencode'
models[minimax]='minimax/MiniMax-M2.7'

mkdir -p "$STATE_DIR" "$EVIDENCE_DIR/snapshots" "$RUN_ROOT/work"

fail() {
  local message="$*"
  printf 'sounio-loom-three-agent-canary: FAIL: %s\n' "$message" >&2
  printf '%s\n' "$message" > "$EVIDENCE_DIR/failure.txt"
  exit 1
}

field() {
  local name="$1" value
  value="$(cat)"
  sed -n "s/.* ${name}=\([^ ]*\).*/\1/p" <<< "$value"
}

loom() {
  SOUNIO_COORD_RUNTIME_MODE=local "$LOOM" "$@"
}

lane_status() {
  local agent="$1"
  loom status --state-dir "$STATE_DIR" --cwd "${workdirs[$agent]}" \
    --agent "$agent" --lane "${lanes[$agent]}"
}

stop_lanes() {
  local agent
  for agent in "${AGENTS[@]}"; do
    [[ -n "${workdirs[$agent]:-}" ]] || continue
    loom stop --state-dir "$STATE_DIR" --cwd "${workdirs[$agent]}" \
      --agent "$agent" --lane "${lanes[$agent]}" >/dev/null 2>&1 || true
  done
}
trap stop_lanes EXIT

for command in codex grok opencode jq gzip; do
  command -v "$command" >/dev/null 2>&1 || fail "required command is missing: $command"
done
[[ -x "$LOOM" ]] || fail "Loom launcher is missing: $LOOM"

"$ROOT_DIR/scripts/dev/build_sounio_loom.sh" > "$EVIDENCE_DIR/build.log" 2>&1
loom runtime-version > "$EVIDENCE_DIR/runtime.txt"
grep -q '^language=OCaml$' "$EVIDENCE_DIR/runtime.txt" || \
  fail 'selected Loom runtime is not OCaml'

started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
for agent in "${AGENTS[@]}"; do
  workdirs[$agent]="$RUN_ROOT/work/$agent"
  receipts[$agent]="${workdirs[$agent]}/agent-receipt.txt"
  tokens[$agent]="LOOM_REAL_${agent^^}_${RUN_ID//[^A-Za-z0-9]/_}"
  mkdir -p "${workdirs[$agent]}"
done

jq -n \
  --arg schema 'sounio.loom.three-agent-canary.prereg.v1' \
  --arg run_id "$RUN_ID" \
  --arg started_utc "$started_utc" \
  --arg hypothesis 'Three real agent CLIs continue while all disposable Loom kernels are absent, then replay their work under recovered kernels without changing guardian, harness, or instance identity.' \
  --arg control 'Each agent must create a unique receipt through its own Bash tool; a textual imitation without the receipt fails.' \
  --arg acceptance 'All three receipts and replay tokens exist; every recovered kernel PID changes; every guardian PID, harness PID, and instance ID remains unchanged.' \
  --arg launch_surface 'loom-provider-abi-v1' \
  --arg codex "$(command -v codex)" \
  --arg grok "$(command -v grok)" \
  --arg minimax "$(command -v opencode)" \
  '{schema:$schema,run_id:$run_id,started_utc:$started_utc,hypothesis:$hypothesis,control:$control,acceptance:$acceptance,launch_surface:$launch_surface,executables:{codex:$codex,grok:$grok,minimax_via_opencode:$minimax}}' \
  > "$EVIDENCE_DIR/prereg.json"

agent_prompt() {
  local agent="$1"
  printf "Use the Bash tool to run exactly this command and do nothing else: sleep %s; printf '%%s\\n' '%s' | tee '%s'. After the command finishes, reply with exactly %s." \
    "$SLEEP_SECONDS" "${tokens[$agent]}" "${receipts[$agent]}" "${tokens[$agent]}"
}

start_agent() {
  local agent="$1" prompt session_id
  local -a arguments
  prompt="$(agent_prompt "$agent")"
  session_id="$(cat /proc/sys/kernel/random/uuid)"
  arguments=(
    provider-start
    --provider "${providers[$agent]}"
    --state-dir "$STATE_DIR"
    --agent "$agent"
    --lane "${lanes[$agent]}"
    --session-id "$session_id"
    --cwd "${workdirs[$agent]}"
    --prompt "$prompt"
    --isolate-context
    --unsafe-auto
  )
  if [[ -n "${models[$agent]:-}" ]]; then
    arguments+=(--model "${models[$agent]}")
  fi
  loom "${arguments[@]}"
}

for agent in "${AGENTS[@]}"; do
  start_agent "$agent" >> "$EVIDENCE_DIR/start.log"
done

for agent in "${AGENTS[@]}"; do
  status="$(lane_status "$agent")"
  printf '=== %s ===\n%s\n' "$agent" "$status" >> "$EVIDENCE_DIR/pre-crash-status.txt"
  [[ "$(field state <<< "$status")" == active ]] || fail "$agent did not become active"
  kernels_before[$agent]="$(field daemon_pid <<< "$status")"
  guardians[$agent]="$(field guardian_pid <<< "$status")"
  harnesses[$agent]="$(field harness_pid <<< "$status")"
  instances[$agent]="$(field instance_id <<< "$status")"
  for pid in "${kernels_before[$agent]}" "${guardians[$agent]}" "${harnesses[$agent]}"; do
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] || fail "$agent omitted a process identity"
    kill -0 "$pid" 2>/dev/null || fail "$agent process is not alive before the crash: $pid"
  done
done

crash_started_ns="$(date +%s%N)"
for agent in "${AGENTS[@]}"; do
  loom crash-kernel --state-dir "$STATE_DIR" --cwd "${workdirs[$agent]}" \
    --agent "$agent" --lane "${lanes[$agent]}" --at now >> "$EVIDENCE_DIR/crash.log"
done

for agent in "${AGENTS[@]}"; do
  kill -0 "${guardians[$agent]}" 2>/dev/null || fail "$agent guardian died with its kernel"
  kill -0 "${harnesses[$agent]}" 2>/dev/null || fail "$agent CLI died with its kernel"
  inventory="$(loom list --state-dir "$STATE_DIR" --cwd "${workdirs[$agent]}")"
  guardian_status="$(loom guardian-status --state-dir "$STATE_DIR" \
    --cwd "${workdirs[$agent]}" --agent "$agent" --lane "${lanes[$agent]}")"
  printf '=== %s inventory ===\n%s\n=== %s guardian ===\n%s\n' \
    "$agent" "$inventory" "$agent" "$guardian_status" \
    >> "$EVIDENCE_DIR/kernel-absent-status.txt"
  grep -q "LOOM_SESSION state=recoverable agent=$agent lane=${lanes[$agent]} " \
    <<< "$inventory" || \
    fail "$agent did not become recoverable after kernel loss"
  [[ "$(field bridge_clients <<< "$guardian_status")" == 0 ]] || \
    fail "$agent guardian retained the dead kernel bridge"
done

for agent in "${AGENTS[@]}"; do
  loom recover --state-dir "$STATE_DIR" --cwd "${workdirs[$agent]}" \
    --agent "$agent" --lane "${lanes[$agent]}" >> "$EVIDENCE_DIR/recover.log"
done
recovered_ns="$(date +%s%N)"
recovery_ms="$(( (recovered_ns - crash_started_ns) / 1000000 ))"

for agent in "${AGENTS[@]}"; do
  status="$(lane_status "$agent")"
  printf '=== %s ===\n%s\n' "$agent" "$status" >> "$EVIDENCE_DIR/post-recovery-status.txt"
  [[ "$(field state <<< "$status")" == active ]] || fail "$agent kernel did not recover"
  kernels_after[$agent]="$(field daemon_pid <<< "$status")"
  [[ "${kernels_after[$agent]}" != "${kernels_before[$agent]}" ]] || \
    fail "$agent reused the dead kernel PID"
  [[ "$(field guardian_pid <<< "$status")" == "${guardians[$agent]}" ]] || \
    fail "$agent guardian identity changed"
  [[ "$(field harness_pid <<< "$status")" == "${harnesses[$agent]}" ]] || \
    fail "$agent CLI identity changed"
  [[ "$(field instance_id <<< "$status")" == "${instances[$agent]}" ]] || \
    fail "$agent instance identity changed"
done

deadline=$((SECONDS + WAIT_SECONDS))
for agent in "${AGENTS[@]}"; do
  while [[ ! -s "${receipts[$agent]}" && "$SECONDS" -lt "$deadline" ]]; do
    sleep 1
  done
  [[ -s "${receipts[$agent]}" ]] || fail "$agent did not create its tool receipt"
  grep -Fxq "${tokens[$agent]}" "${receipts[$agent]}" || \
    fail "$agent receipt has the wrong token"
done

for agent in "${AGENTS[@]}"; do
  replay=''
  for _ in $(seq 1 60); do
    replay="$(loom snapshot --state-dir "$STATE_DIR" --cwd "${workdirs[$agent]}" \
      --agent "$agent" --lane "${lanes[$agent]}" --cursor 0 2>/dev/null || true)"
    [[ "$replay" == *"${tokens[$agent]}"* ]] && break
    sleep 1
  done
  printf '%s\n' "$replay" > "$EVIDENCE_DIR/snapshots/$agent.txt"
  [[ "$replay" == *"${tokens[$agent]}"* ]] || fail "$agent token is absent from replay"
  gzip -n "$EVIDENCE_DIR/snapshots/$agent.txt"
done

completed_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
jq -n \
  --arg schema 'sounio.loom.three-agent-canary.outcome.v1' \
  --arg run_id "$RUN_ID" \
  --arg completed_utc "$completed_utc" \
  --argjson recovery_ms "$recovery_ms" \
  --arg codex_instance "${instances[codex]}" \
  --arg grok_instance "${instances[grok]}" \
  --arg minimax_instance "${instances[minimax]}" \
  '{schema:$schema,run_id:$run_id,completed_utc:$completed_utc,pass:true,launch_surface:"loom-provider-abi-v1",real_agents:["codex","grok","minimax"],presentation_kernels_destroyed:3,recovery_ms:$recovery_ms,identity_preserved:true,tool_receipts:3,replay_tokens:3,instances:{codex:$codex_instance,grok:$grok_instance,minimax:$minimax_instance}}' \
  > "$EVIDENCE_DIR/outcome.json"

(cd "$EVIDENCE_DIR" && sha256sum prereg.json runtime.txt pre-crash-status.txt \
  kernel-absent-status.txt post-recovery-status.txt snapshots/*.txt.gz outcome.json) \
  > "$EVIDENCE_DIR/sha256.txt"

printf 'sounio-loom-three-agent-canary: PASS agents=codex,grok,minimax kernels_destroyed=3 identity_preserved=3 receipts=3 replay_tokens=3 recovery_ms=%s evidence=%s\n' \
  "$recovery_ms" "$EVIDENCE_DIR"
