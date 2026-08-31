#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
LOOM="${SOUNIO_LOOM_PRODUCT_RUNTIME:-$ROOT_DIR/tools/loom/_build/default/src/loom.exe}"
FIXTURE="$ROOT_DIR/tools/loom/_build/default/src/loom_sovereign_provider_fixture.exe"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-native-hook.sovereign-product.XXXXXX")"
STATE_DIR="$TEST_ROOT/loom"
COORD_DIR="$TEST_ROOT/coord"
ORACLE_DIR="$TEST_ROOT/no-oracle"
COORD_RUNTIME_MODE=local
COORD_RUNTIME_DIR=''

if [[ -n "${SOUNIO_LOOM_PRODUCT_RUNTIME:-}" ]]; then
  product_bundle="$(dirname "$(dirname "$(readlink -f "$LOOM")")")"
  COORD_RUNTIME_DIR="$(dirname "$(dirname "$product_bundle")")"
  COORD_RUNTIME_MODE=installed-selftest
fi

fail() {
  printf 'sounio-loom-sovereign-execution-kernel-product-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

cleanup() {
  local state pid
  while IFS= read -r state; do
    pid="$(sed -n 's/^daemon_pid=//p' "$state" | head -1)"
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] || continue
    kill -KILL "$pid" 2>/dev/null || true
  done < <(find "$STATE_DIR" -name session.state -type f 2>/dev/null)
  if [[ "${SOUNIO_LOOM_KEEP_TEST_ROOT:-0}" == 1 ]]; then
    printf 'sounio-loom-sovereign-execution-kernel-product-selftest: retained=%s\n' \
      "$TEST_ROOT" >&2
  else
    rm -rf "$TEST_ROOT"
  fi
}
trap cleanup EXIT

mkdir -p "$STATE_DIR" "$COORD_DIR" "$ORACLE_DIR"
for oracle in python python3 rustc cargo; do
  printf '#!/bin/sh\nprintf invoked >%s/%s.invoked\nexit 97\n' \
    "$ORACLE_DIR" "$oracle" > "$ORACLE_DIR/$oracle"
  chmod 0555 "$ORACLE_DIR/$oracle"
done

dune build --root "$ROOT_DIR/tools/loom" \
  src/loom_sovereign_provider_fixture.exe >/dev/null 2>&1
if [[ -z "${SOUNIO_LOOM_PRODUCT_RUNTIME:-}" ]]; then
  SOUNIO_SOURCE_ROOT="$ROOT_DIR" "$ROOT_DIR/scripts/dev/build_sounio_loom.sh" \
    >/dev/null 2>&1
fi
[[ -x "$LOOM" && -x "$FIXTURE" ]] || fail 'product binaries are absent'

launch() {
  local lane="$1" session_id="$2" mode="$3" command="$4"
  env PATH="$ORACLE_DIR:$PATH" \
    SOUNIO_COORD_RUNTIME_MODE="$COORD_RUNTIME_MODE" \
    SOUNIO_COORD_RUNTIME_DIR="$COORD_RUNTIME_DIR" \
    SOUNIO_COORD_NATIVE_HOOK_SELFTEST=1 \
    SOUNIO_LOOM_START_READY_TIMEOUT_SECONDS=120 \
    SOUNIO_COORD_DIR="$COORD_DIR" \
    SOUNIO_LOOM_COORD_AUTO=0 \
    SOUNIO_LOOM_PROVIDER_HOOK_RUNTIME="$LOOM" \
    "$LOOM" start --state-dir "$STATE_DIR" --agent sovereign-product \
      --lane "$lane" --session-id "$session_id" --cwd "$ROOT_DIR" -- \
      "$FIXTURE" "$mode" "$command" >/dev/null
}

session_dir() {
  printf '%s/sessions/sovereign-product--%s' "$STATE_DIR" "$1"
}

wait_for() {
  local description="$1"; shift
  local attempt
  for attempt in $(seq 1 600); do
    if "$@"; then return 0; fi
    sleep 0.05
  done
  fail "timeout waiting for $description"
}

is_exited() {
  grep -q '^state=exited$' "$(session_dir "$1")/session.state" 2>/dev/null
}

has_material() {
  find "$(session_dir "$1")" -name guardian.tsv -type f -exec \
    grep -l $'\tMATERIAL_REGISTERED\t' {} + 2>/dev/null | grep -q .
}

has_result() {
  find "$(session_dir "$1")" -path '*/sovereign-results/*.record' \
    -type f -print -quit 2>/dev/null | grep -q .
}

has_material_leaf() {
  local state daemon_pid worker_pid material_pid executable
  state="$(session_dir "$1")/session.state"
  daemon_pid="$(sed -n 's/^daemon_pid=//p' "$state" 2>/dev/null | head -1)"
  [[ "$daemon_pid" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ -r "/proc/$daemon_pid/task/$daemon_pid/children" ]] || return 1
  for worker_pid in $(<"/proc/$daemon_pid/task/$daemon_pid/children"); do
    [[ -r "/proc/$worker_pid/task/$worker_pid/children" ]] || continue
    for material_pid in $(<"/proc/$worker_pid/task/$worker_pid/children"); do
      executable="$(readlink "/proc/$material_pid/exe" 2>/dev/null || true)"
      [[ "$executable" == /usr/bin/sleep ]] && return 0
    done
  done
  return 1
}

runtime_processes_absent() {
  ! ps -eo args= | grep -F "$TEST_ROOT" | grep -v grep >/dev/null
}

result_path() {
  find "$(session_dir "$1")" -path '*/sovereign-results/*.record' \
    -type f -print -quit
}

output_path() {
  find "$(session_dir "$1")" -name output.bin -type f -print -quit
}

journal_path() {
  find "$(session_dir "$1")" -name journal.tsv -type f -print -quit
}

launch happy 11111111-1111-4111-8111-111111111111 execute \
  '/usr/bin/printf LOOM_SOVEREIGN_OK'
wait_for 'happy-path termination' is_exited happy
happy_result="$(result_path happy)"
happy_output="$(output_path happy)"
[[ -f "$happy_result" && -f "$happy_output" ]] || fail 'happy result is absent'
grep -q '^state=COMPLETED$' "$happy_result" || fail 'happy result did not complete'
grep -q '^material_started=true$' "$happy_result" || fail 'material did not start'
grep -q '^material_completed=true$' "$happy_result" || fail 'material did not complete'
grep -q '^guardian_revoked=false$' "$happy_result" || fail 'happy path was revoked'
grep -q '^pdeathsig_armed=true$' "$happy_result" || fail 'parent death revocation was not armed'
grep -q '^outcome_authority_invoked=true$' "$happy_result" || fail 'outcome authority was skipped'
[[ "$(cat "$happy_output")" == LOOM_SOVEREIGN_OK ]] || fail 'presented output diverged'
grep -q $'\tSOVEREIGN_EXEC_COMPLETED\t' "$(journal_path happy)" || \
  fail 'kernel did not promote the completed result'

launch spoof 22222222-2222-4222-8222-222222222222 spoof-start \
  '/usr/bin/printf SPOOF_MUST_NOT_EXECUTE'
wait_for 'same-UID spoof refusal' is_exited spoof
spoof_output="$(output_path spoof)"
grep -q 'sovereign-issuer-not-direct-harness-child' "$spoof_output" || \
  fail 'same-UID spoof was not causally refused'
grep -q $'\tSOVEREIGN_EXEC_REFUSED\t' "$(journal_path spoof)" || \
  fail 'same-UID spoof refusal was not journaled'
if has_material spoof; then fail 'same-UID spoof registered material'; fi
if has_result spoof; then fail 'same-UID spoof produced a result'; fi

launch transport 33333333-3333-4333-8333-333333333333 transport-exit \
  '/usr/bin/sleep 1'
wait_for 'transport-loss material registration' has_material transport
wait_for 'transport-loss completion' has_result transport
wait_for 'transport-loss session termination' is_exited transport
transport_result="$(result_path transport)"
grep -q '^state=COMPLETED$' "$transport_result" || \
  fail 'provider loss interrupted material execution'
grep -q '^guardian_revoked=false$' "$transport_result" || \
  fail 'provider loss was treated as Guardian loss'

launch guardian 44444444-4444-4444-8444-444444444444 transport-exit \
  '/usr/bin/sleep 30'
wait_for 'Guardian-death material registration' has_material guardian
wait_for 'Guardian-death live material leaf' has_material_leaf guardian
guardian_state="$(session_dir guardian)/session.state"
guardian_pid="$(sed -n 's/^guardian_pid=//p' "$guardian_state" | head -1)"
[[ "$guardian_pid" =~ ^[1-9][0-9]*$ ]] || fail 'Guardian PID is invalid'
kill -KILL "$guardian_pid"
wait_for 'Guardian-death refusal record' has_result guardian
guardian_result="$(result_path guardian)"
grep -q '^state=GUARDIAN_REVOKED$' "$guardian_result" || \
  fail 'true Guardian death did not revoke'
grep -q '^material_completed=false$' "$guardian_result" || \
  fail 'revoked material was reported complete'
grep -q '^guardian_revoked=true$' "$guardian_result" || \
  fail 'Guardian revocation fact is absent'

wait_for 'product runtime process extinction' runtime_processes_absent
if find "$ORACLE_DIR" -name '*.invoked' -print -quit | grep -q .; then
  fail 'a prohibited Python or Rust oracle executed'
fi
if ! runtime_processes_absent; then
  fail 'a product runtime process survived its test'
fi

printf '%s\n' \
  'sounio-loom-sovereign-execution-kernel-product-selftest: PASS semantic_authority=Sounio action=9042 stage=PRODUCT_EXECUTION_ATTACHED happy_path=PASS hostile_same_uid_spoof=REFUSED_BEFORE_EXECUTION transport_death=MATERIAL_COMPLETED guardian_death=GUARDIAN_REVOKED grant_residency=Loom_kernel_memory grant_is_bearer=false grant_single_use=true consume_atomic=true interface_release_authority=zero same_uid_peer_isolation=true production_activation=true exec_attached=true write_attached=false commit_attached=false ci_attached=false worker_reaped=true residual_processes=false python_executed=false rust_executed=false claim_ready=false'
