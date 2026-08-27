#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
SOUC="${SOUNIO_LOOM_WITNESS_MESH_V1_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_WITNESS_MESH_V1_ENGINE:-lean_single}"
STDLIB="$ROOT_DIR/stdlib"
MODULE="$STDLIB/coordination/loom_witness_mesh_v1.sio"
WITNESS="$ROOT_DIR/tests/compiler/loom_witness_mesh_v1_typestate_check.sio"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-witness-v1-typestate.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

fail() {
  printf 'loom-witness-mesh-v1-typestate: FAIL: %s\n' "$*" >&2
  exit 1
}

compose_source() {
  local source="$1" output="$2"
  {
    sed -n '1,$p' "$MODULE"
    awk '
      /^use coordination::loom_witness_mesh_v1::\{/ { skip=1; next }
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
combined="$WORK/loom_witness_mesh_v1_typestate.sio"
compose_source "$WITNESS" "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" run "$combined" \
  >"$WORK/runtime.log" 2>&1 || {
  sed -n '1,280p' "$WORK/runtime.log" >&2
  fail 'linear witness-mesh-v1 proof did not run'
}
rg -Fxq \
  'loom-witness-mesh-v1-typestate: PASS linear=1 shares=3 checkpoint=1 host_receipt=1 quorum=3/4' \
  "$WORK/runtime.log" || {
  sed -n '1,280p' "$WORK/runtime.log" >&2
  fail 'linear witness-mesh-v1 proof omitted its exact receipt'
}

expect_rejection private-host-receipt \
  "$ROOT_DIR/tests/compile-fail/loom_witness_mesh_v1_host_receipt_private.sio" E176

reuse="$WORK/reuse.sio"
compose_source \
  "$ROOT_DIR/tests/compile-fail/loom_witness_mesh_v1_receipt_reuse.sio" \
  "$reuse"
expect_rejection receipt-reuse "$reuse" E039

printf 'loom-witness-mesh-v1-typestate: PASS privacy=E176 linear=E039 shares=3 frame=9014\n'
