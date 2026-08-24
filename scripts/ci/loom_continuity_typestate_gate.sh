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
  compose_source "$module" "$WITNESS" "$output"
}

compose_source() {
  local module="$1" source="$2" output="$3"
  {
    cat "$module"
    awk '
      /^use coordination::loom_continuity::\{/ { skip=1; next }
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
  [[ "$rc" -eq 1 ]] || {
    cat "$log" >&2
    fail "$label must reject with rc=1, got rc=$rc"
  }
  local count
  count="$(rg -c "error\[$code" "$log" || true)"
  if [[ "$code" == E039 || "$code" == E175 ]]; then
    [[ "$count" -ge 1 ]] || {
      cat "$log" >&2
      fail "$label must emit at least one $code"
    }
    return
  fi
  [[ "$count" -eq 1 ]] || {
    cat "$log" >&2
    fail "$label must emit exactly one $code"
  }
}

expect_composed_rejection() {
  local label="$1" source="$2" code="$3"
  local program="$WORK/$label.sio"
  compose_source "$MODULE" "$source" "$program"
  expect_rejection "$label" "$program" "$code"
}

[[ -x "$SOUC" ]] || fail "compiler is missing: $SOUC"

combined="$WORK/loom_continuity_kernel.sio"
compose_witness "$MODULE" "$combined"
runtime_log="$WORK/runtime.log"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" run "$combined" >"$runtime_log" 2>&1 || {
  cat "$runtime_log" >&2
  fail 'single-module Loom continuity witness did not run'
}
rg -Fxq \
  'loom-continuity-typestate: PASS host_seal=1 linear=1 initial=1 clean=1 pod=1 predecessor=refused count=refused kind=refused authority=refused' \
  "$runtime_log" || {
    cat "$runtime_log" >&2
    fail 'Loom continuity witness omitted its exact receipt'
  }

expect_rejection private-constructor \
  "$PRIVACY/loom_continuity_private_struct_main.sio" E176
expect_composed_rejection wrong-state-promotion \
  "$PRIVACY/loom_continuity_wrong_state_main.sio" E009
expect_rejection unsealed-host-admission \
  "$PRIVACY/loom_continuity_unsealed_admission_main.sio" E175
expect_composed_rejection linear-reuse \
  "$PRIVACY/loom_continuity_linear_reuse_main.sio" E039

cp -a "$STDLIB" "$WORK/visibility-stdlib"
visibility_module="$WORK/visibility-stdlib/coordination/loom_continuity.sio"
visibility_count="$(rg -c '^fn admit_runtime_continuity\($' "$visibility_module")"
[[ "$visibility_count" -eq 1 ]] || \
  fail "expected one private host admission before sabotage, got $visibility_count"
awk '
  BEGIN { changed=0 }
  !changed && $0 == "fn admit_runtime_continuity(" {
    print "pub fn admit_runtime_continuity("
    changed=1
    next
  }
  { print }
  END { if (changed != 1) exit 42 }
' "$visibility_module" > "$WORK/visibility-mutated.sio" || \
  fail 'could not make host admission public for sabotage control'
mv "$WORK/visibility-mutated.sio" "$visibility_module"
visibility_log="$WORK/visibility-sabotage.log"
set +e
SOUNIO_STDLIB_PATH="$WORK/visibility-stdlib" "$SOUC" check \
  "$PRIVACY/loom_continuity_unsealed_admission_main.sio" \
  >"$visibility_log" 2>&1
visibility_rc=$?
set -e
if rg -q 'error\[E175' "$visibility_log"; then
  cat "$visibility_log" >&2
  fail 'making host admission public did not remove the E175 refusal'
fi
if [[ "$visibility_rc" -ne 0 ]] && ! rg -q 'error\[E039' "$visibility_log"; then
  cat "$visibility_log" >&2
  fail 'visibility sabotage failed for a reason other than the known modular linear baseline'
fi

mkdir -p "$WORK/predecessor-stdlib/coordination"
cp "$MODULE" "$WORK/predecessor-stdlib/coordination/loom_continuity.sio"
predecessor_module="$WORK/predecessor-stdlib/coordination/loom_continuity.sio"
mutation_count="$(rg -c '^    if observed\.predecessor_semantic_head_token == 0 \{ return None \}$' \
  "$predecessor_module")"
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
' "$predecessor_module" > "$WORK/predecessor-mutated.sio" || \
  fail 'could not apply the targeted predecessor-guard mutation'
mv "$WORK/predecessor-mutated.sio" "$predecessor_module"

sabotage_program="$WORK/sabotage_kernel.sio"
compose_witness "$predecessor_module" "$sabotage_program"
sabotage_log="$WORK/sabotage.log"
set +e
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" run "$sabotage_program" >"$sabotage_log" 2>&1
sabotage_rc=$?
set -e
if [[ "$sabotage_rc" -eq 0 ]]; then
  cat "$sabotage_log" >&2
  fail 'removing the Pod predecessor guard did not expose the negative witness'
fi

echo "loom-continuity-typestate: PASS positive_engine=$ENGINE negative_engine=madaros host-seal=E175 private=E176 wrong-state=E009 linear-reuse=E039 sabotage-host-seal=1 sabotage-predecessor-guard=1"
