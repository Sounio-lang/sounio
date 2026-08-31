#!/usr/bin/env bash
# run_gate_list.sh — run a named batch of gate scripts and account for them
# under one CI job, so a workflow can name a WAVE instead of one script at a
# time.
#
# Usage:
#   printf '%s\n' scripts/ci/foo_gate.sh '!scripts/ci/bar_gate.sh' \
#     | bash scripts/ci/run_gate_list.sh <wave-name>
#
# Input: one gate path per line on stdin. Blank lines and lines starting
# with `#` are ignored. A leading `!` marks a gate KNOWN to fail on main
# today.
#
# The `!` convention is scripts/ci/madaros_witness_gate.sh's, carried over
# unchanged because it already gets the incentive right for a batch that
# mixes healthy and known-broken gates:
#   - a `!` gate still RUNS every time — a known defect that stops being
#     exercised is a defect nobody is watching, not a defect that is fixed;
#   - a `!` gate FAILING does not fail this job — that is the known state;
#   - a `!` gate PASSING DOES fail this job, with a message that says to
#     drop the `!` — a "known-failing" gate that quietly starts passing is a
#     stale claim sitting in the repo, not good news suppressed politely.
# A gate with no `!` must pass, full stop; this is how green here stays
# meaningful instead of drifting into "green unless known-red, in which
# case also green".
#
# What green here means, and does not mean: it means every named gate in
# this wave RAN and returned the rc its `!` status predicted. It does NOT
# mean the subsystem each gate checks is healthy — read the gate, not the
# wave, for that.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
# shellcheck source=scripts/lib/gate_assert.sh
. scripts/lib/gate_assert.sh

WAVE="${1:?usage: run_gate_list.sh <wave-name> < gate-list}"
gate_name "run_gate_list_${WAVE}"

ART_DIR="${GATE_ARTIFACT_DIR:-artifacts/gates}"
mkdir -p "$ART_DIR"
OUT_JSON="$ART_DIR/${WAVE}.json"

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
total=0; passed=0; failed=0; known_failing=0

while IFS= read -r line; do
  case "$line" in ''|'#'*) continue ;; esac
  known=0
  case "$line" in '!'*) known=1; line="${line#!}" ;; esac
  gate="$line"

  require_nonempty "$gate" "blank gate path after stripping '!'"
  [[ -f "$gate" ]] || gate_fail "listed but absent: $gate"

  total=$((total + 1))
  log="$W/$(printf '%s' "$gate" | tr '/' '_').log"
  rcfile="$W/$(printf '%s' "$gate" | tr '/' '_').rc"

  set +e
  bash "$gate" >"$log" 2>&1
  echo "$?" > "$rcfile"
  set -e
  require_rc_file "$rcfile"
  rc="$(cat "$rcfile")"

  if [[ "$rc" -eq 0 ]]; then
    if [[ "$known" -eq 1 ]]; then
      echo "[run-gate-list:$WAVE] FAIL $gate -- marked known-failing (!) but PASSED; drop the '!'"
      tail -n 20 "$log" | sed 's/^/[run-gate-list:'"$WAVE"']      /'
      failed=$((failed + 1))
    else
      echo "[run-gate-list:$WAVE] ok   $gate"
      passed=$((passed + 1))
    fi
  else
    if [[ "$known" -eq 1 ]]; then
      echo "[run-gate-list:$WAVE] known $gate -- still failing (rc=$rc), as declared"
      known_failing=$((known_failing + 1))
    else
      echo "[run-gate-list:$WAVE] FAIL $gate -- rc=$rc"
      tail -n 40 "$log" | sed 's/^/[run-gate-list:'"$WAVE"']      /'
      failed=$((failed + 1))
    fi
  fi
done

# A wave that named nothing must be red, not a silent no-op green.
require_min_count "$total" 1 "gates listed for wave '$WAVE'"

st=pass
[[ $failed -eq 0 ]] || st=fail
cat > "$OUT_JSON" <<JSON
{"status":"$st","wave":"$WAVE","metrics":{"total":$total,"passed":$passed,"failed":$failed,"not_run":$known_failing}}
JSON
echo "status=$st"
echo "metrics {total=$total, passed=$passed, failed=$failed, not_run=$known_failing}"
echo "[run-gate-list:$WAVE] known-failing: $known_failing (drop the '!' when fixed)"
echo "artifact=$OUT_JSON"
[[ $st == pass ]]
