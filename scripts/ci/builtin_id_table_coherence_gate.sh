#!/usr/bin/env bash
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/gate_artifact.sh"
# Builtin-id table coherence.
#
# A builtin name reaches machine code through FOUR independent tables in
# self-hosted/native/codegen_x86_linux.sio:
#
#   native_v2_builtin_id_for_func_ref   name -> id
#   native_v2_builtin_id_for_name       name -> id
#   native_v2_emit_builtin_by_id_into   id   -> emitter
#   native_v2_builtin_returns_float     id   -> ABI
#
# Nothing in the language ties them together. They are hand-written parallel
# lists and they drift silently: a renumbering that updates the two producers
# and misses the consumer type-checks clean, builds clean, and miscompiles.
#
# Measured 2026-08-23 on this branch. ffi_* were renumbered out of a collision
# with main's live builtins in the two producers only. The stale consumer rows
# then sat ABOVE main's own rows in a first-match-wins dispatch and shadowed
# them: id 33 read_offset64 emitted ffi_system, id 34 write_offset64 emitted
# ffi_exit, id 35 read_offset64 emitted ffi_abort. Those are the memory
# primitives, so essentially every program miscompiled -- five unrelated FFI
# fixtures built rc=0 and SIGSEGVed before printing a line, while the same
# fixtures against the unmodified compiler ran rc=0.
#
# Two agreeing tables read exactly like consistency. This gate counts the
# tables instead of assuming them.
set -euo pipefail
cd "$(dirname "$0")/../.."
# The anti-vacuity primitives. This gate reads its facts out of source with
# regexes, and a regex that stops matching reports on ZERO rows and calls that
# agreement. Not hypothetical: on 2026-08-23 a comment containing `{ on }`
# truncated a non-greedy extraction in a sibling gate and silently dropped six
# entries from the set being checked. So the count is floored, not trusted.
. "$(dirname "${BASH_SOURCE[0]}")/../lib/gate_assert.sh"
gate_name "builtin_id_table_coherence"
CG=self-hosted/native/codegen_x86_linux.sio
ART=artifacts/gates/builtin_id_table_coherence.v1.json
mkdir -p "$(dirname "$ART")"

run() { python3 scripts/ci/lib/builtin_id_coherence.py "$1"; }

# Positive control FIRST. A checker that has never failed has measured nothing:
# if the sabotaged copy passes, the gate is inspecting nothing and must not be
# allowed to report green on the real file.
SAB=$(mktemp); trap 'rm -f "$SAB"' EXIT
sed 's/if builtin_id == 47 {/if builtin_id == 37 {/' "$CG" | gate_write_artifact "$SAB"
if run "$SAB" >/dev/null 2>&1; then
  echo "CONTROL_FAIL: the sabotaged table passed. This gate inspects nothing."
  printf '{"status":"fail","reason":"positive control did not fire","metrics":{"total":0,"passed":0,"failed":1,"not_run":0}}\n' | gate_write_artifact "$ART"
  exit 1
fi
echo "control: sabotaged table rejected, as required"

if out=$(run "$CG" 2>&1); then
  echo "$out"
  n=$(printf '%s' "$out" | grep -cE '^ *[a-z_]+ +producers=' || true)
  # A green over nothing is the failure this gate exists to prevent elsewhere,
  # so it must not be able to commit it itself. Ten is the count on the day the
  # floor was set: the seven original ffi_* plus sqrt, floor and ceil. The range
  # only grows, so a drop means the extraction broke, not that builtins left.
  require_min_count "$n" 10 "ffi_* builtins compared across the tables"
  printf '{"status":"pass","metrics":{"total":%s,"passed":%s,"failed":0,"not_run":0}}\n' "$n" "$n" | gate_write_artifact "$ART"
  exit 0
fi
echo "$out"
printf '{"status":"fail","metrics":{"total":0,"passed":0,"failed":1,"not_run":0}}\n' | gate_write_artifact "$ART"
exit 1
