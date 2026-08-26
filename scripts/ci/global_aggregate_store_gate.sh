#!/usr/bin/env bash
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/gate_artifact.sh"
# scripts/ci/global_aggregate_store_gate.sh
#
# Two reproductions of one defect family in the bootstrap seed: a global with an
# AGGREGATE type does not round-trip.
#
#   lean_single_global_aggregate_store.sio       struct -> global ARRAY element
#                                                is a silent NO-OP
#   lean_single_global_struct_store_overrun.sio  struct -> PLAIN global OVERRUNS
#                                                into the globals declared after it
#
# WHY THIS IS A GATE AND NOT JUST A COMMENT. The second one reads back as 0 when
# the struct's tail is zero padding, and a global that reads 0 is
# indistinguishable from a counter that was never incremented. Instrumentation
# built on such a global reports "nothing ever happened" with total confidence,
# and everything concluded from it is void rather than merely unproven. That is
# not a hypothetical: self-hosted/ir/lower.sio carried this exact shape and its
# error counter read 0 for every compile across three commits.
#
# WHAT THIS GATE ASSERTS. Not "the bug exists" — a gate that demands a bug stay
# broken is a gate that fights its own fix. It asserts the DIVERGENCE is where we
# recorded it, in both directions:
#
#   - Madaros must be CORRECT on both files. A regression there is a real
#     regression and is red.
#   - lean_single is EXPECTED to be broken. If it starts passing, this gate goes
#     red as PROGRESS, the same ratchet as madaros_fixed_point_gate.sh: the seed
#     was fixed, the flat-array workaround is no longer forced, and that fact
#     must be recorded rather than absorbed silently.
#
# The files themselves are `//@ ignore` in the test suite, and must stay that
# way: the suite's own stage2 IS lean_single, so they would be a permanent red
# there with no way to act on it. This gate is where they get to mean something.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "global_aggregate_store"

CASES=(
  tests/known_failures/lean_single_global_aggregate_store.sio
  tests/known_failures/lean_single_global_struct_store_overrun.sio
)

LEAN="${SOUNIO_LEAN_SINGLE_BIN:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
MADAROS="${MADAROS_BIN:-}"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/globagg.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

TOTAL=0
PASSED=0
FAILED=0
NOT_RUN=0

# Compile with the engine's own argv and run; echo the verdict token the file
# prints. "OK" / "BROKEN" are the only two the sources emit, so anything else
# (a compile failure, a segfault, an empty run) is reported as its own token
# rather than silently folded into BROKEN.
run_case() {
  local engine="$1" bin="$2" src="$3"
  # Separate statement: within one `local`, $src is not yet assigned when a
  # later initialiser on the same line expands it.
  local out="$WORK/$(basename "$src" .sio).$engine.elf"
  local log="$WORK/$(basename "$src" .sio).$engine.log"

  case "$engine" in
    lean)    ( ulimit -s 524288 2>/dev/null || true; "$bin" "$src" "$out" ) >"$log" 2>&1 ;;
    madaros) ( ulimit -s 524288 2>/dev/null || true; "$bin" build "$src" "$out" ) >"$log" 2>&1 ;;
  esac
  if [[ ! -s "$out" ]]; then printf 'COMPILE_FAILED'; return; fi

  chmod +x "$out"
  local run_out
  run_out="$("$out" 2>&1)"
  # The two files use different verdict tokens ("OK" vs
  # "GLOBAL_AGGREGATE_STORE_OK"). Match the suffix rather than editing the
  # sources — they are reproductions, and a reproduction that gets reformatted
  # to suit its harness is one edit away from no longer reproducing anything.
  if grep -qE '(^|_)OK$' <<<"$run_out"; then printf 'OK'
  elif grep -qE '(^|_)BROKEN' <<<"$run_out"; then printf 'BROKEN'
  else printf 'NO_VERDICT'; fi
}

echo "GLOBAL_AGGREGATE_STORE_V1"
echo "lean_single $LEAN"
echo "madaros     ${MADAROS:-<unset>}"
echo

for SRC in "${CASES[@]}"; do
  echo "== $SRC"
  if [[ ! -f "$SRC" ]]; then
    gate_fail "$SRC is missing. These two files ARE the record of this defect; without them the only trace of it is a commit message."
  fi

  grep -q '^//@ ignore' "$SRC" \
    || gate_fail "$SRC lost its '//@ ignore'. The test suite's stage2 is lean_single, so this file would be a permanent red there — it belongs to this gate, not to the suite."

  # ── lean_single: expected broken ────────────────────────────────────────────
  TOTAL=$((TOTAL + 1))
  if [[ -x "$LEAN" ]]; then
    LEAN_V="$(run_case lean "$LEAN" "$SRC")"
    echo "   lean_single  $LEAN_V (recorded: BROKEN)"
    case "$LEAN_V" in
      BROKEN)
        PASSED=$((PASSED + 1)) ;;
      OK)
        FAILED=$((FAILED + 1))
        gate_fail "lean_single now PASSES $SRC. This is PROGRESS and it is red on purpose: the seed's global-aggregate defect is fixed, so the parallel-flat-array workaround in self-hosted/ir/lower.sio and self-hosted/ir/ir.sio is no longer forced. Record that here before the fact is absorbed silently." ;;
      *)
        FAILED=$((FAILED + 1))
        gate_fail "lean_single produced '$LEAN_V' on $SRC — neither verdict. The file did not compile or did not run, so it measured nothing at all; see $WORK." ;;
    esac
  else
    NOT_RUN=$((NOT_RUN + 1))
    echo "   lean_single  SKIP (no binary at $LEAN)"
  fi

  # ── Madaros: expected correct ───────────────────────────────────────────────
  TOTAL=$((TOTAL + 1))
  if [[ -n "$MADAROS" && -x "$MADAROS" ]]; then
    MAD_V="$(run_case madaros "$MADAROS" "$SRC")"
    echo "   madaros      $MAD_V (recorded: OK)"
    if [[ "$MAD_V" == "OK" ]]; then
      PASSED=$((PASSED + 1))
    else
      FAILED=$((FAILED + 1))
      gate_fail "Madaros produced '$MAD_V' on $SRC, but Madaros is the engine that gets this RIGHT — it is the reason we know the defect is the seed's and not the language's. A regression here removes the only working reference. See $WORK."
    fi
  else
    NOT_RUN=$((NOT_RUN + 1))
    echo "   madaros      SKIP (set MADAROS_BIN to a raw Madaros ELF)"
  fi
  echo
done

if [[ "$PASSED" -eq 0 ]]; then
  gate_fail "no arm ran: neither engine was available, so this gate measured nothing. A gate that finds none of its subject must not report success."
fi

LEAN_MEASURED=$([[ -x "$LEAN" ]] && echo "BROKEN (measured)" || echo "not measured")
MAD_MEASURED=$([[ -n "$MADAROS" && -x "$MADAROS" ]] && echo "OK (measured)" || echo "not measured")

ART_DIR="${SOUNIO_ARTIFACT_DIR:-$ROOT_DIR/artifacts/gates}"
mkdir -p "$ART_DIR"
cat <<JSON | gate_write_artifact "$ART_DIR/global_aggregate_store.json"
{
  "gate": "global_aggregate_store",
  "status": "pass",
  "metrics": {
    "total": $TOTAL,
    "passed": $PASSED,
    "failed": $FAILED,
    "not_run": $NOT_RUN
  },
  "witness": {
    "quantity": "GLOBAL_AGGREGATE_STORE_DIVERGENCE",
    "lean_single": "$LEAN_MEASURED",
    "madaros": "$MAD_MEASURED",
    "note": "seed drops struct stores into global arrays and overruns from plain struct globals into the next global; workaround is parallel flat arrays"
  }
}
JSON

# Name what was actually measured. Saying "lean_single broken, Madaros correct"
# when the Madaros arms were skipped is the exact vacuity this library exists to
# prevent — and it would read as a two-engine result to whoever greps the log.
COVERAGE="lean_single ✓, madaros ✓"
[[ -z "$MADAROS" || ! -x "$MADAROS" ]] && COVERAGE="lean_single ✓, madaros NOT MEASURED (MADAROS_BIN unset)"
[[ ! -x "$LEAN" ]] && COVERAGE="lean_single NOT MEASURED, madaros ✓"

gate_pass "global-aggregate store divergence is where it was recorded. $COVERAGE. $PASSED/$TOTAL arms checked, $NOT_RUN skipped. Artifact: $ART_DIR/global_aggregate_store.json"
