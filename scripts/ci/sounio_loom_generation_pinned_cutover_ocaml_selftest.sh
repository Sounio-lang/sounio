#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
GIT_COMMON="$(git -C "$ROOT_DIR" rev-parse --git-common-dir)"
SHARED_RUNTIME_ROOT="$GIT_COMMON/sounio-coord-runtime"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-generation-pin.XXXXXX")"
CHILDREN=()

cleanup() {
  local pid
  for pid in "${CHILDREN[@]}"; do
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
  rm -rf "$WORK"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-generation-pinned-cutover-ocaml-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

expect_denied() {
  local label="$1"
  shift
  set +e
  local output
  output="$("$@" 2>&1)"
  local rc=$?
  set -e
  ((rc != 0)) || fail "$label unexpectedly allowed"
  [[ -n "$output" ]] || fail "$label failed without a reason"
}

(
  flock -x 9
  dune build --root "$ROOT_DIR/tools/loom" src/loom.exe >/dev/null
) 9>"$ROOT_DIR/tools/loom/_build/.dune-build.lock"
bash "$ROOT_DIR/scripts/dev/build_sounio_loom_generation_pinned_cutover.sh" >/dev/null
[[ -x "$LOOM" ]] || fail 'OCaml runtime is absent'
POLICY_ROOT="$WORK/policy"
mkdir -p "$POLICY_ROOT/tools/loom"
cp "$ROOT_DIR/tools/loom/generation_pinned_cutover.freeze.v1" \
  "$POLICY_ROOT/tools/loom/generation_pinned_cutover.freeze.v1"
chmod 600 "$POLICY_ROOT/tools/loom/generation_pinned_cutover.freeze.v1"

OLD_RUNTIME="$(basename "$(readlink -f "$SHARED_RUNTIME_ROOT/current")")"
CANDIDATE_RUNTIME="$(basename "$(readlink -f "$SHARED_RUNTIME_ROOT/native-next")")"
if [[ "$OLD_RUNTIME" == "$CANDIDATE_RUNTIME" ]]; then
  CANDIDATE_RUNTIME="$OLD_RUNTIME"
  OLD_RUNTIME=""
  for pin in "$GIT_COMMON/sounio-coord-state/generation-runtime-pins"/*.pin; do
    [[ -f "$pin" ]] || continue
    pinned_runtime="$(sed -n 's/^runtime_id=//p' "$pin")"
    if [[ -n "$pinned_runtime" && "$pinned_runtime" != "$CANDIDATE_RUNTIME" &&
          -d "$SHARED_RUNTIME_ROOT/versions/$pinned_runtime" ]]; then
      OLD_RUNTIME="$pinned_runtime"
      break
    fi
  done
  [[ -n "$OLD_RUNTIME" ]] || fail 'no installed pinned runtime differs from current'
fi
RUNTIME_ROOT="$WORK/runtime"
mkdir -p "$RUNTIME_ROOT"
ln -s "$SHARED_RUNTIME_ROOT/versions" "$RUNTIME_ROOT/versions"
ln -s "versions/$OLD_RUNTIME" "$RUNTIME_ROOT/current"
ln -s "versions/$CANDIDATE_RUNTIME" "$RUNTIME_ROOT/native-next"

sleep 300 &
LIVE_PID=$!
CHILDREN+=("$LIVE_PID")
STAT_TEXT="$(<"/proc/$LIVE_PID/stat")"
STAT_TAIL="${STAT_TEXT##*) }"
read -r -a STAT_FIELDS <<<"$STAT_TAIL"
LIVE_START="${STAT_FIELDS[19]}"
BOOT_ID="$(tr -d '\n' </proc/sys/kernel/random/boot_id)"
PID_NAMESPACE="$(readlink /proc/self/ns/pid)"
HOST="$(hostname)"

new_state() {
  local name="$1"
  local state="$WORK/$name"
  mkdir -p "$state/process-presences" "$state/hook-capabilities"
  printf '%s\n' \
    'presence_id=test--generation-pin' 'agent=test' 'lane=generation-pin' \
    'session_id=generation-pin-session' 'generation=7' 'harness=codex' \
    "worktree=$ROOT_DIR" "host=$HOST" "boot_id=$BOOT_ID" \
    "pid_namespace=$PID_NAMESPACE" "pid=$LIVE_PID" "pid_start=$LIVE_START" \
    'last_seen_epoch=1' 'ttl_seconds=1' \
    >"$state/process-presences/test--generation-pin.presence"
  chmod 600 "$state/process-presences/test--generation-pin.presence"
  printf '%s\n' "$state"
}

run_seal() {
  local state="$1"
  shift
  SOUNIO_LOOM_HOOK_TEST_MODE=1 SOUNIO_COORD_DIR="$state" \
    SOUNIO_LOOM_GENERATION_PIN_POLICY_ROOT="$POLICY_ROOT" \
    SOUNIO_COORD_RUNTIME_DIR="$RUNTIME_ROOT" \
    "$LOOM" hook-generation-pin-seal --source-root "$ROOT_DIR" \
      --git-common "$GIT_COMMON" --old-runtime "$OLD_RUNTIME" \
      --candidate-runtime "$CANDIDATE_RUNTIME" "$@"
}

STATE="$(new_state control)"
OUTPUT="$(run_seal "$STATE")"
[[ "$OUTPUT" == *'SOUNIO_GENERATION_PINNED_CUTOVER CUTOVER_READY semantic_authority=Sounio action=9048'* ]] ||
  fail "control did not reach CUTOVER_READY: $OUTPUT"
PIN="$STATE/generation-runtime-pins/test--generation-pin.pin"
ACTIVATION="$STATE/generation-runtime-pins/activation.v1"
[[ -f "$PIN" && -f "$ACTIVATION" ]] || fail 'control omitted pin or activation receipt'
[[ "$(stat -c '%a' "$PIN")" == 600 ]] || fail 'pin mode is not 0600'
grep -q "^runtime_id=$OLD_RUNTIME$" "$PIN" || fail 'legacy process was not pinned to old current'

PIN_SHA="$(sha256sum "$PIN" | awk '{print $1}')"
ACTIVATION_SHA="$(sha256sum "$ACTIVATION" | awk '{print $1}')"
sed -i 's/^last_seen_epoch=.*/last_seen_epoch=2/' \
  "$STATE/process-presences/test--generation-pin.presence"
run_seal "$STATE" >/dev/null
[[ "$PIN_SHA" == "$(sha256sum "$PIN" | awk '{print $1}')" ]] || fail 'idempotent retry rewrote pin'
[[ "$ACTIVATION_SHA" == "$(sha256sum "$ACTIVATION" | awk '{print $1}')" ]] ||
  fail 'idempotent retry rewrote activation receipt'

PYTHON_STATE="$(new_state python-oracle)"
expect_denied python-oracle-attempt env SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_GENERATION_PIN_POLICY_ROOT="$POLICY_ROOT" \
  SOUNIO_COORD_DIR="$PYTHON_STATE" SOUNIO_COORD_RUNTIME_DIR="$RUNTIME_ROOT" \
  "$LOOM" hook-generation-pin-seal --source-root "$ROOT_DIR" \
  --git-common "$GIT_COMMON" --old-runtime python --candidate-runtime "$CANDIDATE_RUNTIME"
[[ ! -e "$PYTHON_STATE/generation-runtime-pins/activation.v1" ]] ||
  fail 'Python oracle attempt wrote activation receipt'

MISSING_STATE="$(new_state missing-policy)"
expect_denied missing-policy env SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_GENERATION_PIN_POLICY_ROOT="$WORK/absent-policy" \
  SOUNIO_COORD_DIR="$MISSING_STATE" SOUNIO_COORD_RUNTIME_DIR="$RUNTIME_ROOT" \
  "$LOOM" hook-generation-pin-seal --source-root "$ROOT_DIR" \
  --git-common "$GIT_COMMON" --old-runtime "$OLD_RUNTIME" --candidate-runtime "$CANDIDATE_RUNTIME"

DRIFT_STATE="$(new_state selector-drift)"
expect_denied selector-drift env SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_GENERATION_PIN_POLICY_ROOT="$POLICY_ROOT" \
  SOUNIO_COORD_DIR="$DRIFT_STATE" SOUNIO_COORD_RUNTIME_DIR="$RUNTIME_ROOT" \
  "$LOOM" hook-generation-pin-seal --source-root "$ROOT_DIR" \
  --git-common "$GIT_COMMON" --old-runtime "$CANDIDATE_RUNTIME" --candidate-runtime "$OLD_RUNTIME"

MALFORMED_STATE="$(new_state malformed-pin)"
run_seal "$MALFORMED_STATE" >/dev/null
printf '%s\n' 'schema=loom-generation-runtime-pin-v1' 'schema=duplicate' \
  >"$MALFORMED_STATE/generation-runtime-pins/test--generation-pin.pin"
expect_denied malformed-pin run_seal "$MALFORMED_STATE"

LOCK_STATE="$(new_state lock-timeout)"
SOUNIO_LOOM_GENERATION_PIN_TEST_HOLD_LOCK_SECONDS=4 run_seal "$LOCK_STATE" >/dev/null &
LOCK_PID=$!
CHILDREN+=("$LOCK_PID")
sleep 0.25
expect_denied lock-timeout run_seal "$LOCK_STATE"
wait "$LOCK_PID"
CHILDREN=("$LIVE_PID")

[[ ! -s "$WORK/forbidden.log" ]] || fail 'a disposable oracle executed'
grep -q 'SOUNIO_LOOM_GENERATION_PIN_FORWARD_TARGET=' \
  "$ROOT_DIR/tools/loom/src/loom_hook_generation_pin.ml" ||
  fail 'forward target marker is absent'
! grep -q 'SOUNIO_LOOM_GENERATION_PIN_FORWARDED=1' \
  "$ROOT_DIR/tools/loom/src/loom_hook_generation_pin.ml" ||
  fail 'legacy boolean forward marker breaks immutable target compatibility'
grep -q 'generation-pin-forward-target-drift' \
  "$ROOT_DIR/tools/loom/src/loom_hook_generation_pin.ml" ||
  fail 'forward target drift is not fail-closed'
grep -q 'generation-pin-recursive-forward' \
  "$ROOT_DIR/tools/loom/src/loom_hook_generation_pin.ml" ||
  fail 'recursive forward is not fail-closed'
printf 'SOUNIO_LOOM_GENERATION_PINNED_CUTOVER_OCAML_SELFTEST PASS cases=11 python_oracle_executed=false semantic_authority=Sounio action=9048\n'
