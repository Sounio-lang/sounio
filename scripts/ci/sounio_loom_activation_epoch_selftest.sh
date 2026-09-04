#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-activation-epoch-test.XXXXXX")"
trap 'rm -rf "$work"' EXIT
runtime="$work/runtime"
SOUNIO_LOOM_ACTIVATION_EPOCH_OUTPUT="$runtime" bash "$ROOT/scripts/dev/build_sounio_loom_activation_epoch.sh" >/dev/null
allow() { [[ "$(printf '%s\n' "$1" | "$runtime")" == "$2" ]]; }
deny() { local out rc; set +e; out="$(printf '%s\n' "$1" | "$runtime")"; rc=$?; set -e; [[ $rc -eq 42 && "$out" == "$2" ]]; }
allow '9049 1 3 262143 1 2 3 4 1 0' 'SOUNIO_ACTIVATION_EPOCH ADVANCE semantic_authority=Sounio action=9049'
allow '9049 2 3 262143 1 2 3 4 9 8' 'SOUNIO_ACTIVATION_EPOCH VALID semantic_authority=Sounio action=9049'
deny '9049 1 3 245759 1 2 3 4 1 0' 'SOUNIO_ACTIVATION_EPOCH DENY717 semantic_authority=Sounio action=9049'
deny '9049 1 3 262143 1 2 3 4 3 1' 'SOUNIO_ACTIVATION_EPOCH DENY719 semantic_authority=Sounio action=9049'
grep -q 'epoch_fact(o.word, 14) != 1' "$ROOT/stdlib/coordination/loom_activation_epoch_authority.sio"
grep -q 'epoch_fact(o.word, 15) != 1' "$ROOT/stdlib/coordination/loom_activation_epoch_authority.sio"
grep -q 'epoch_fact(o.word, 16) != 1' "$ROOT/stdlib/coordination/loom_activation_epoch_authority.sio"
printf 'sounio-loom-activation-epoch-selftest: PASS semantic_authority=Sounio action=9049 stage=SOUNIO_EXECUTABLE cases=13 python_oracle_attempt=DENY717 python_executed=false rust_executed=false disposable_oracle_executed=false\n'
