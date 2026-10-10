#!/usr/bin/env bash
# scripts/ci/native_reloc_unknown_kind_gate.sh
#
# Pins the native backend's refusal to emit an ELF when a relocation has a kind
# apply_relocations_into cannot patch (anything but 1 call, 2 .rodata, 3 .data,
# 4 fn-address). Such a site used to stay at its rel32 placeholder and the ELF
# was still written, so the program jumped through garbage at run time.
#
# No emitter produces an unknown kind, so no user program reaches the refusal,
# and tests/run-pass/native_reloc_unknown_kind_refuses.sio can only pin the
# predicate and the recorder (importing the x86 backend into a run-pass test
# overruns the 30s budget). Removing the branch in apply_relocations_into would
# leave that test green. This gate closes the gap in two halves:
#
#   static  the branch exists inside apply_relocations_into (calls the recorder
#           and latches reloc_overflow); the single refusal point
#           native_reloc_refuse_if_invalid (native/frame.sio) prints the unknown
#           kind and the site count and returns 20 for an unknown kind and for
#           reloc_overflow; and EVERY route from apply_relocations_into to a file
#           write passes through it. The routes are DERIVED from the code by
#           scripts/ci/native_reloc_refusal_routes.py (callers of
#           apply_relocations_into, followed through wrappers that relocate the
#           caller's NativeCompiler, to every writer they reach) -- not a fixed
#           list, so a new route or writer is checked without editing this gate.
#   live    (1) self-test T70r (native_v2_reloc_unknown_kind_selftest) in a
#           Madaros built from current source. T70r builds a NativeCompiler, adds
#           a kind-9 relocation, runs the real apply_relocations_into, and fails if
#           the flag, the counter or the untouched placeholder are wrong; a
#           known-kind control checks nothing else refuses. Removing the branch
#           turns T70r into "FAIL: T70r".
#           (2) scripts/ci/native_reloc_sret_refusal_check.sh: the sret witness
#           route driven with an injected kind-9 relocation must report rc 20 and
#           leave NO output file; the uninjected control must write an ELF that
#           exits 14. Fails on d52b82d4, the #2803 head first reviewed.
#
# Env:
#   SOUNIO_NATIVE_RELOC_UNKNOWN_KIND_MADAROS  Madaros built from current source
#     (scripts/ci/build_modular_madaros.sh). When set, the live half is
#     REQUIRED: T70r must print OK. When unset, the live half is NOT_RUN and
#     only the static half is certified.
#
# `madaros --self-test` crashes later in the run on current main (well after
# T70r; recorded in scripts/ci/epistemic_egraph_rewrite_gate.sh). That crash is
# not this gate's concern: the log up to the crash is what is read, and an empty
# log is a failure, not a pass.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9
. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "native_reloc_unknown_kind_gate"

SRC="self-hosted/native/codegen_x86_linux.sio"
FRAME="self-hosted/native/frame.sio"
require_file "$SRC"
require_file "$FRAME"

# Body of one top-level fn: from its `fn <name>(` line to the first `}` in
# column 0 after it.
fn_body() {
  awk -v name="$2" '
    !inside && $0 ~ "^(pub )?fn " name "\\(" { inside = 1 }
    inside { print }
    inside && /^}/ { exit }
  ' "$1"
}

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "--- static ---"

fn_body "$SRC" apply_relocations_into >"$TMP/apply"
require_nonempty_file "$TMP/apply" "apply_relocations_into not found in $SRC"
# The unknown-kind branch: the predicate guard, the recorder, and the latch,
# in that order and inside the same else-if arm.
awk '
  /else if !native_v2_reloc_kind_is_known\(kind_code\)/ { arm = 1; next }
  arm && /nc_reloc_note_unknown_kind\(kind_code\)/ { rec = 1 }
  arm && rec && /\(\*nc\)\.reloc_overflow = true/ { ok = 1; exit }
  arm && /^        }/ { exit }
  END { exit ok ? 0 : 1 }
' "$TMP/apply" \
  || gate_fail "apply_relocations_into has no unknown-kind arm that records the kind and latches reloc_overflow"
echo "PASS  static:apply_relocations_into records an unknown kind and latches reloc_overflow"

fn_body "$FRAME" native_v2_reloc_kind_is_known >"$TMP/pred"
require_nonempty_file "$TMP/pred" "native_v2_reloc_kind_is_known not found in $FRAME"
grep -Eq 'kind_code >= 1 && kind_code <= 4' "$TMP/pred" \
  || gate_fail "native_v2_reloc_kind_is_known no longer accepts exactly kinds 1..4"
echo "PASS  static:native_v2_reloc_kind_is_known accepts exactly 1..4"

fn_body "$FRAME" native_reloc_refuse_if_invalid >"$TMP/helper"
require_nonempty_file "$TMP/helper" "native_reloc_refuse_if_invalid not found in $FRAME"
# Unknown kind: names the first kind and the site count, returns 20. Overflow:
# returns 20. Both before the fall-through `0`.
awk '
  /if NC_RELOC_UNKNOWN_KIND_COUNT > 0 \{/ { blk = 1; next }
  blk && /print_int\(NC_RELOC_UNKNOWN_KIND_FIRST\)/ { first = 1 }
  blk && /print_int\(NC_RELOC_UNKNOWN_KIND_COUNT\)/ { count = 1 }
  blk && /return 20/ { ret = 1 }
  blk && /^    }/ { blk = 0; done = 1 }
  /if \(\*nc\)\.reloc_overflow \{/ { ob = 1; next }
  ob && /return 20/ { oret = 1 }
  ob && /^    }/ { ob = 0 }
  END { exit (done && first && count && ret && oret) ? 0 : 1 }
' "$TMP/helper" \
  || gate_fail "native_reloc_refuse_if_invalid does not return 20 (naming kind and site count) on an unknown kind, or does not return 20 on reloc_overflow"
echo "PASS  static:native_reloc_refuse_if_invalid refuses rc 20 on unknown kind (naming kind and site count) and on reloc_overflow"

# No second copy of the refusal: outside the helper, nothing tests the counter.
dups="$(grep -rn --include='*.sio' 'NC_RELOC_UNKNOWN_KIND_COUNT > 0' self-hosted | grep -v "^$FRAME:" || true)"
[[ -z "$dups" ]] || gate_fail "refusal duplicated outside native_reloc_refuse_if_invalid: $dups"
echo "PASS  static:no duplicated refusal outside native_reloc_refuse_if_invalid"

echo "--- routes (derived from callers of apply_relocations_into) ---"
python3 "$ROOT_DIR/scripts/ci/native_reloc_refusal_routes.py" "$ROOT_DIR" >"$TMP/routes" 2>&1
routes_rc=$?
cat "$TMP/routes"
[[ "$routes_rc" == 0 ]] || gate_fail "a route from apply_relocations_into reaches a write without native_reloc_refuse_if_invalid (see FAIL lines above)"
routes="$(grep -c '^PASS  route' "$TMP/routes" || true)"
require_min_count "$routes" 1 "routes from apply_relocations_into to a write"
echo "PASS  static:$routes route(s) to a write, every one through native_reloc_refuse_if_invalid"

echo "--- live ---"
MADAROS="${SOUNIO_NATIVE_RELOC_UNKNOWN_KIND_MADAROS:-}"
if [[ -z "$MADAROS" ]]; then
  echo "NOT_RUN  live:T70r + sret refusal check (set SOUNIO_NATIVE_RELOC_UNKNOWN_KIND_MADAROS to a Madaros built via scripts/ci/build_modular_madaros.sh)"
  gate_pass "static half only (live T70r NOT_RUN)"
  exit 0
fi
require_executable "$MADAROS"
# The self-test overflows the default stack: at 8 MiB and at the 16 MiB GitHub
# runners give, it dies around T60, before T70r; at 512 MiB it reaches T110.
# Same soft limit scripts/ci/madaros_changed_tests_gate.sh sets for Madaros.
stack_kb="${SOUNIO_NATIVE_RELOC_UNKNOWN_KIND_STACK_KB:-524288}"
stack_soft="$(ulimit -S -s 2>/dev/null || true)"
require_nonempty "$stack_soft" "could not read the soft stack limit"
if [[ "$stack_soft" != "unlimited" ]] && (( stack_soft < stack_kb )); then
  ulimit -S -s "$stack_kb" 2>/dev/null \
    || gate_fail "could not raise the soft stack limit to ${stack_kb} KiB (hard=$(ulimit -H -s 2>/dev/null || echo unavailable)); T70r is not reachable below it"
fi
echo "stack soft_before_kb=$stack_soft soft_after_kb=$(ulimit -S -s)"
madaros_rc=0
timeout 120 "$MADAROS" --self-test >"$TMP/selftest.log" 2>&1 || madaros_rc=$?
require_nonempty_file "$TMP/selftest.log" "self-test log is empty (rc=$madaros_rc): the instrument produced no evidence"
if grep -q "FAIL: T70r" "$TMP/selftest.log"; then
  grep "T70r" "$TMP/selftest.log" >&2
  gate_fail "self-test T70r failed: apply_relocations_into no longer refuses an unknown kind"
elif grep -q "T70r OK" "$TMP/selftest.log"; then
  echo "PASS  live:T70r (self-test rc=$madaros_rc)"
else
  tail -n 5 "$TMP/selftest.log" >&2
  gate_fail "self-test T70r was not reached (rc=$madaros_rc) -- the live half is required when a Madaros is given"
fi

echo "--- live: sret route, injected unknown kind ---"
bash "$ROOT_DIR/scripts/ci/native_reloc_sret_refusal_check.sh" "$MADAROS" \
  || gate_fail "sret route with an injected unknown relocation kind did not refuse (rc 20, no output file)"

gate_pass "unknown relocation kinds are refused statically (every derived route) and live (T70r, sret route)"
