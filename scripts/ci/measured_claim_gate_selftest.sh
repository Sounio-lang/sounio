#!/usr/bin/env bash
# measured_claim_gate_selftest.sh — prove the gate fails when it should.
#
# The defect this whole line of work is about is machinery that reports success
# without checking anything. On the same sweep that motivated the gate, two
# compile-fail fixtures were found matching their own FILENAMES: keyed on the
# bare string `E040`, while the compiler echoes `source: .../rust_let_mut_e040.sio`.
# They passed. They would have passed against a compiler that printed nothing.
#
# A gate with no exercised failure path is that same defect wearing a gate's
# name. So: four controls, two of which must FAIL, and the run is only OK if
# each behaves as stated.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9
GATE="$ROOT_DIR/scripts/ci/measured_claim_gate.sh"
[[ -x "$GATE" ]] || { echo "SELFTEST FAIL: gate not executable at $GATE" >&2; exit 1; }

W="$(mktemp -d /tmp/measured_claim_selftest.XXXXXX)"
trap 'rm -rf "$W"' EXIT
printf 'id\tdescription\tclaim_cmd\tmeasure_cmd\n' > "$W/hdr"
printf 'id\tdescription\tclaim_cmd\tmeasure_cmd\tfix_cmd\n' > "$W/hdr5"

run_case() {  # run_case <name> <expect pass|fail> <claims-file> <baseline-file> [needle]
  local name="$1" expect="$2" claims="$3" base="$4" needle="${5:-}" out rc
  out="$(SOUNIO_MEASURED_CLAIMS="$claims" SOUNIO_MEASURED_CLAIMS_BASELINE="$base" bash "$GATE" 2>&1)"; rc=$?
  local got=pass; [[ $rc -ne 0 ]] && got=fail
  if [[ "$got" != "$expect" ]]; then
    echo "SELFTEST FAIL: $name expected $expect, got $got (rc=$rc)" >&2
    echo "$out" | sed 's/^/    /' >&2
    return 1
  fi
  if [[ -n "$needle" ]] && ! grep -qF -- "$needle" <<<"$out"; then
    echo "SELFTEST FAIL: $name behaved correctly but never said '$needle'" >&2
    echo "$out" | sed 's/^/    /' >&2
    return 1
  fi
  echo "  ok  $name ($expect)"
}

: > "$W/empty_baseline"

# GREEN — a claim that agrees must pass.
{ cat "$W/hdr"; printf 'agree\ttwo equals two\techo 7\techo 7\n'; } > "$W/c_agree"
run_case "agreeing claim passes" pass "$W/c_agree" "$W/empty_baseline" "ok        agree" || exit 1

# RED — a claim that disagrees and is NOT baselined must fail. This is the case
# every stale number in the sweep would have hit.
{ cat "$W/hdr"; printf 'drift\tratchet-shaped drift\techo 471\techo 468\n'; } > "$W/c_drift"
run_case "drifted claim fails" fail "$W/c_drift" "$W/empty_baseline" "claimed 471, measured 468" || exit 1

# GREEN — the same drift, baselined, is reported and tolerated.
printf 'drift\n' > "$W/base_drift"
run_case "baselined drift is excused" pass "$W/c_drift" "$W/base_drift" "baselined drift" || exit 1

# RED — the failure mode that would make this gate vacuous: a claim command that
# reads nothing. Comparing "" to "" must never be a pass.
{ cat "$W/hdr"; printf 'silent\tkey renamed away\tgrep -oE "\\"gone\\":[0-9]+" /dev/null\techo 5\n'; } > "$W/c_silent"
run_case "unreadable claim fails, not passes" fail "$W/c_silent" "$W/empty_baseline" "produced NOTHING" || exit 1

# RED — a baselined row does NOT excuse an unreadable claim. Baselining is for
# known disagreement, not for a check that stopped running.
printf 'silent\n' > "$W/base_silent"
run_case "baseline does not excuse unreadable" fail "$W/c_silent" "$W/base_silent" "could not be read" || exit 1

# GREEN — a fixed row still passes and says so, so the baseline can be pruned.
{ cat "$W/hdr"; printf 'fixed\tagrees again\techo 3\techo 3\n'; } > "$W/c_fixed"
printf 'fixed\n' > "$W/base_fixed"
run_case "fixed row tells you to prune the baseline" pass "$W/c_fixed" "$W/base_fixed" "FIXED" || exit 1

# RED — a drift must print the command that repairs it. A gate that says a
# number is wrong and leaves the reader to find the regeneration flag is why
# seven pull requests and three agents each ran the same search on 2026-08-28.
{ cat "$W/hdr5"; printf 'withfix\tdrift with a fix\techo 471\techo 468\tbash scripts/ci/regen_me.sh --refresh\n'; } > "$W/c_fix"
run_case "drift names its fix command" fail "$W/c_fix" "$W/empty_baseline" "FIX:     bash scripts/ci/regen_me.sh --refresh" || exit 1

# RED — a four-column row still works, and says the fix is missing rather than
# printing an empty FIX line that reads as "no action needed".
run_case "legacy row admits it has no fix" fail "$W/c_drift" "$W/empty_baseline" "no fix_cmd recorded" || exit 1

# RED — a table that parses to zero rows is not a green gate.
cat "$W/hdr" > "$W/c_empty"
run_case "empty table fails" fail "$W/c_empty" "$W/empty_baseline" "zero rows" || exit 1

echo "MEASURED_CLAIM_GATE_SELFTEST_OK: 9 controls, 6 of them RED, each behaved as stated"
