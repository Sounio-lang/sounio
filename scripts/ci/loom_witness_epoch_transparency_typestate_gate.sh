#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
SOUC="${SOUNIO_LOOM_WITNESS_EPOCH_TRANSPARENCY_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_WITNESS_EPOCH_TRANSPARENCY_ENGINE:-lean_single}"
STDLIB="$ROOT_DIR/stdlib"
MODULE="$STDLIB/coordination/loom_witness_epoch_transparency.sio"
WITNESS="$ROOT_DIR/tests/compiler/loom_witness_epoch_transparency_typestate_check.sio"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-witness-epoch-transparency-typestate.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

fail() {
  printf 'loom-witness-epoch-transparency-typestate: FAIL: %s\n' "$*" >&2
  exit 1
}

compose_module_source() {
  local module="$1" source="$2" output="$3"
  {
    sed -n '1,$p' "$module"
    awk '
      /^use coordination::loom_witness_epoch_transparency::\{/ { skip=1; next }
      skip && /^\}/ { skip=0; next }
      !skip { print }
    ' "$source"
  } > "$output"
}

expect_rejection() {
  local label="$1" source="$2" code="$3" rc=0
  local log="$WORK/$label.log"
  set +e
  SOUNIO_STDLIB_PATH="$STDLIB" "$SOUC" check "$source" >"$log" 2>&1
  rc=$?
  set -e
  [[ "$rc" -eq 1 ]] || { sed -n '1,240p' "$log" >&2; fail "$label rc=$rc"; }
  rg -q "error\[$code" "$log" || {
    sed -n '1,240p' "$log" >&2
    fail "$label omitted $code"
  }
}

[[ -x "$SOUC" ]] || fail "compiler is missing: $SOUC"

combined="$WORK/typestate.sio"
compose_module_source "$MODULE" "$WITNESS" "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" run "$combined" >"$WORK/runtime.log" 2>&1 || {
  sed -n '1,280p' "$WORK/runtime.log" >&2
  fail 'linear epoch transparency proof did not run'
}
rg -Fxq \
  'loom-witness-epoch-transparency-typestate: PASS linear=1 handoff=1 checkpoint=1 transparency=1 host_receipt=1' \
  "$WORK/runtime.log" || {
  sed -n '1,280p' "$WORK/runtime.log" >&2
  fail 'linear proof omitted its exact receipt'
}

expect_rejection private-host-receipt \
  "$ROOT_DIR/tests/compile-fail/loom_witness_epoch_transparency_host_receipt_private.sio" E176

reuse="$WORK/reuse.sio"
compose_module_source "$MODULE" \
  "$ROOT_DIR/tests/compile-fail/loom_witness_epoch_transparency_quorum_reuse.sio" \
  "$reuse"
expect_rejection authority-reuse "$reuse" E039

"$ROOT_DIR/scripts/ci/sounio_loom_witness_epoch_transparency_adapter_selftest.sh" \
  >"$WORK/adapter.log"
rg -q 'same_frame_sabotages=8' "$WORK/adapter.log" || {
  cat "$WORK/adapter.log" >&2
  fail 'adapter sabotage controls were not executed'
}

printf 'loom-witness-epoch-transparency-typestate: PASS privacy=E176 linear=E039 frame=9016 same_frame_sabotages=8 rollback_below_latest_quorum_witnessed=REFUSED freeze_claim=NONE availability_claim=NONE\n'
