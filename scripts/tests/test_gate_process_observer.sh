#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$ROOT/scripts/lib/gate_process_observer.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
export SOUNIO_GATE_OBSERVE_SECONDS=1
gate_observe_command success "$work/success.log" bash -c 'printf "before\n"; sleep 2; printf "after\n"' > "$work/observer.log"
printf 'before\nafter\n' > "$work/expected"
cmp "$work/expected" "$work/success.log"
grep -q 'GATE_OBSERVATION label=success' "$work/observer.log"
grep -q 'command_rc=0' "$work/observer.log"
pid="$(sed -n 's/.*observer_pid=\([0-9]*\).*/\1/p' "$work/observer.log")"
if kill -0 "$pid" 2>/dev/null; then echo 'observer leaked' >&2; exit 1; fi
for expected in 7 78 143; do
  if gate_observe_command failure "$work/failure.log" bash -c 'printf "original failure\n"; exit "$1"' bash "$expected" > "$work/failure-observer.log"; then
    echo 'failure became success' >&2; exit 1
  else actual=$?; fi
  [[ "$actual" -eq "$expected" ]]
  [[ "$(cat "$work/failure.log")" == 'original failure' ]]
done
export SOUNIO_GATE_OBSERVE_SECONDS=0
if gate_observe_command invalid "$work/invalid.log" touch "$work/executed"; then exit 1; else [[ "$?" -eq 64 ]]; fi
[[ ! -e "$work/executed" && ! -e "$work/invalid.log" ]]
echo 'gate process observer: PASS (live log, exact bytes, status0/7/78/143, cleanup, invalid interval)'
