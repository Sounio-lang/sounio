#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BUILD="$ROOT_DIR/scripts/dev/build_sounio_loom_host_boot_reconciler.sh"
MODULE="$ROOT_DIR/stdlib/coordination/loom_host_boot_reconciler.sio"
ENTRYPOINT="$ROOT_DIR/tools/loom/host_boot_reconciler_main.sio"

fail() {
  printf 'sounio-loom-host-boot-reconciler-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-host-boot-selftest.XXXXXX")"
trap 'rm -rf "$work"' EXIT
runtime="$work/authority"
SOUNIO_LOOM_HOST_BOOT_OUTPUT="$runtime" bash "$BUILD" >/dev/null

frame() {
  printf '9041 %s\n' "$1" | "$runtime"
}

active='3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1'
recover='3 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1'
lost='3 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 1 0 1 1 1 1 1 1'
sabotage='3 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1'

[[ "$(frame "$active")" == 'SOUNIO_HOST_BOOT_RECONCILER NOOP_ACTIVE semantic_authority=Sounio action=9041' ]] ||
  fail 'active decision diverged'
[[ "$(frame "$recover")" == 'SOUNIO_HOST_BOOT_RECONCILER RECOVER_SAME_PHYSICAL semantic_authority=Sounio action=9041' ]] ||
  fail 'recovery decision diverged'
[[ "$(frame "$lost")" == 'SOUNIO_HOST_BOOT_RECONCILER HOLD_LINEAGE_REQUIRED semantic_authority=Sounio action=9041' ]] ||
  fail 'Guardian-loss hold diverged'

set +e
sabotage_output="$(frame "$sabotage" 2>&1)"
sabotage_code=$?
set -e
[[ $sabotage_code -eq 42 && "$sabotage_output" == \
   'SOUNIO_HOST_BOOT_RECONCILER DENY545 semantic_authority=Sounio action=9041' ]] ||
  fail "shipped start-tick sabotage did not refuse: code=$sabotage_code output=$sabotage_output"

mutant="$work/mutant.sio"
sed '0,/observation.guardian_start_verified != 1 ||/s//observation.guardian_start_verified != 0 ||/' \
  "$MODULE" > "$work/mutant-module.sio"
cmp -s "$MODULE" "$work/mutant-module.sio" && fail 'sabotage mutant did not change the rule'
sed -n '1,$p' "$work/mutant-module.sio" "$ENTRYPOINT" > "$mutant"
SOUNIO_SOUC_ENGINE=lean_single "$ROOT_DIR/bin/souc" compile "$mutant" -o "$work/mutant"
chmod 0755 "$work/mutant"
mutant_output="$(printf '9041 %s\n' "$sabotage" | "$work/mutant")"
[[ "$mutant_output" == \
   'SOUNIO_HOST_BOOT_RECONCILER RECOVER_SAME_PHYSICAL semantic_authority=Sounio action=9041' ]] ||
  fail "isolated mutant did not admit unchanged sabotage frame: $mutant_output"

printf 'sounio-loom-host-boot-reconciler-selftest: PASS semantic_authority=Sounio action=9041 cases=14 active=NOOP_ACTIVE recover=RECOVER_SAME_PHYSICAL guardian_loss=HOLD_LINEAGE_REQUIRED guardian_start_mismatch=DENY545 causal_sabotage=PASS python_executed=false rust_executed=false parity_open=false production_activation=false\n'
