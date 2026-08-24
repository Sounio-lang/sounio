#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC="${SOUNIO_LOOM_CONTINUITY_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_CONTINUITY_ENGINE:-lean_single}"
STDLIB="$ROOT_DIR/stdlib"
MODULE="$STDLIB/coordination/loom_continuity.sio"
WITNESS="$ROOT_DIR/tests/compiler/loom_continuity_typestate_check.sio"
PRIVACY="$ROOT_DIR/tests/compiler/loom_continuity_privacy"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-continuity.XXXXXX")"

cleanup() {
  rm -rf "$WORK"
}
trap cleanup EXIT

fail() {
  printf 'loom-continuity-typestate: FAIL: %s\n' "$*" >&2
  exit 1
}

compose_witness() {
  local module="$1" output="$2"
  {
    cat "$module"
    awk '
      /^use coordination::loom_continuity::\{/ { skip=1; next }
      skip && /^\}/ { skip=0; next }
      !skip { print }
    ' "$WITNESS"
  } > "$output"
}

expect_rejection() {
  local label="$1" source="$2" code="$3" rc=0
  local log="$WORK/$label.log"
  set +e
  SOUNIO_STDLIB_PATH="$STDLIB" "$SOUC" check "$source" >"$log" 2>&1
  rc=$?
  set -e
  [[ "$rc" -eq 1 ]] || {
    cat "$log" >&2
    fail "$label must reject with rc=1, got rc=$rc"
  }
  [[ "$(rg -c "error\[$code" "$log")" -eq 1 ]] || {
    cat "$log" >&2
    fail "$label must emit exactly one $code"
  }
}

[[ -x "$SOUC" ]] || fail "compiler is missing: $SOUC"

modular_log="$WORK/modular.log"
SOUNIO_STDLIB_PATH="$STDLIB" "$SOUC" check "$WITNESS" >"$modular_log" 2>&1 || {
  cat "$modular_log" >&2
  fail 'imported Loom continuity witness did not typecheck'
}

combined="$WORK/loom_continuity_kernel.sio"
compose_witness "$MODULE" "$combined"
runtime_log="$WORK/runtime.log"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" run "$combined" >"$runtime_log" 2>&1 || {
  cat "$runtime_log" >&2
  fail 'single-module Loom continuity witness did not run'
}
rg -Fxq \
  'loom-continuity-typestate: PASS initial=1 clean=1 pod=1 predecessor=refused count=refused kind=refused authority=refused' \
  "$runtime_log" || {
    cat "$runtime_log" >&2
    fail 'Loom continuity witness omitted its exact receipt'
  }

expect_rejection private-constructor \
  "$PRIVACY/loom_continuity_private_struct_main.sio" E176
expect_rejection wrong-state-promotion \
  "$PRIVACY/loom_continuity_wrong_state_main.sio" E009

mkdir -p "$WORK/stdlib/coordination"
cp "$MODULE" "$WORK/stdlib/coordination/loom_continuity.sio"
mutation_count="$(rg -c '^    if observed\.predecessor_semantic_head_token == 0 \{ return None \}$' \
  "$WORK/stdlib/coordination/loom_continuity.sio")"
[[ "$mutation_count" -eq 1 ]] || \
  fail "expected one Pod predecessor guard before mutation, got $mutation_count"
awk '
  BEGIN { changed=0 }
  !changed && $0 == "    if observed.predecessor_semantic_head_token == 0 { return None }" {
    print "    if false { return None }"
    changed=1
    next
  }
  { print }
  END { if (changed != 1) exit 42 }
' "$WORK/stdlib/coordination/loom_continuity.sio" > "$WORK/mutated.sio" || \
  fail 'could not apply the targeted predecessor-guard mutation'
mv "$WORK/mutated.sio" "$WORK/stdlib/coordination/loom_continuity.sio"

sabotage_program="$WORK/sabotage_kernel.sio"
compose_witness "$WORK/stdlib/coordination/loom_continuity.sio" "$sabotage_program"
sabotage_log="$WORK/sabotage.log"
set +e
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" run "$sabotage_program" >"$sabotage_log" 2>&1
sabotage_rc=$?
set -e
if [[ "$sabotage_rc" -eq 0 ]]; then
  cat "$sabotage_log" >&2
  fail 'removing the Pod predecessor guard did not expose the negative witness'
fi

echo "loom-continuity-typestate: PASS positive_engine=$ENGINE private=E176 wrong-state=E009 sabotage-predecessor-guard=1"
