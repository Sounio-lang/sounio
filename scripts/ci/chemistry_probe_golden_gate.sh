#!/usr/bin/env bash
# Golden-output gate for four of the five Sounio replica-side probes behind
# benchmarks/chemistry/RESULTS.md (§2.6 adiabatic defect, §7.3 reverse-rate
# defect, §7.7 stagnation test), run under lean_single -- the engine every
# number in RESULTS.md was produced on. A diff means a number in the paper
# moved: re-derive it, do not regenerate the golden to make the gate pass
# (CLAUDE.md §6.6).
#   regenerate (only after the change is understood):
#     REGEN=1 bash scripts/ci/chemistry_probe_golden_gate.sh
#
# What this gate does NOT prove, stated so its green is not over-read: the
# probes run under the COMMITTED bin/souc-lean-single-x86_64, which lags the
# source in self-hosted/ (CLAUDE.md principle 15). A lowering or codegen change
# on main therefore cannot move these numbers until that ELF is refreshed --
# measured 2026-09-03, when main's ir/lower.sio and codegen_x86_linux.sio
# changes merged in and all five probes matched byte for byte, with the ELF's
# md5 unchanged. The md5 is printed below for exactly that reason: it says
# which compiler produced the goldens, so a green run cannot be mistaken as
# evidence about a compiler the run never used.
#
# WHY THIS IS PARALLEL, WHY THE PER-PROBE TIMEOUTS LOOK THE WAY THEY DO, AND
# WHY rep_adiabatic_bug ISN'T HERE BY DEFAULT.
#
# gbs_oracle and h2_ignition_uq_demo were added to PROBES on 2026-09-02/03
# (d04c6715f8, 79812c2016), five months after the gate shipped with a single
# `timeout 1500` (25 min) sized for the original three light probes -- neither
# that per-probe timeout nor the job's timeout-minutes was ever revisited.
# Measured directly (not assumed) on this engine: rep_traj_bug ~153s,
# rep_stagnation ~346s, gbs_oracle ~32 min, h2_ignition_uq_demo ~45 min. All
# four are independent processes writing to separate files with no shared
# state, so they run concurrently here -- wall time becomes roughly
# max(probe time), not sum(probe time). Per-probe ceilings stay in place as a
# real hang detector, sized at ~2.5x the slowest measured run of each to allow
# for a slower CI runner than this box. That is why this gate went from
# cancelled-or-timed-out on 56 of its last 60 runs to reliably green: not a
# code regression, a CI budget that was never grown to match what got added
# to it.
#
# rep_adiabatic_bug is a different problem and stays out of every automated
# trigger, not just this default list. It integrates 2.4e6 RK4 steps (2
# setpoints x 3 forms x 400,000 steps, each a 29-reaction x 10-species rate
# calc). Measured directly, not extrapolated from a guess: reducing
# max_steps to 2,000 (12,000 total steps) and timing a real run gave 471.8s,
# i.e. ~39.3ms/step; linear over the real 2.4e6 steps that is ~94,400s, ~26.2
# HOURS. A 6-hour isolated run (no CPU contention) reached only ~23% of that
# and produced no output -- confirming the cost is real per-step work, not a
# hang or an accidental blowup, and ruling out "just parallelize it too" or
# "give the nightly job more budget": GitHub Actions caps a single hosted-
# runner job at 360 minutes, a hard ceiling this probe cannot meet under any
# schedule, isolated or not. Per-step cost is a real question (self-hosted's
# codegen, or the chemistry module's rate-constant/thermo lookups) that this
# CI-timeout fix does not investigate.
#
# So: no PR trigger, no push trigger, no nightly schedule, no workflow_dispatch
# -- a workflow step destined to fail its own job timeout on every invocation
# is worse than no automation at all. Verify it by hand when touching the
# adiabatic probe or the chemistry module it exercises, budgeting a full day:
#   SOUNIO_CHEM_PROBES="rep_adiabatic_bug" bash scripts/ci/chemistry_probe_golden_gate.sh
#   SOUNIO_CHEM_PROBES="rep_adiabatic_bug" REGEN=1 bash scripts/ci/chemistry_probe_golden_gate.sh
#
# SOUNIO_CHEM_PROBES: space-separated subset to run in place of the default
# four, e.g. to run just one probe or to add rep_adiabatic_bug back in for a
# manual check.
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 1
export SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib"
SOUC="${SOUC:-$ROOT_DIR/bin/souc}"
GOLD="$ROOT_DIR/benchmarks/chemistry/golden"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/chem-probe-golden.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
# Bare default excludes rep_adiabatic_bug on purpose (see above): nobody
# invoking this without arguments should accidentally start a ~26-hour run.
DEFAULT_PROBES=(rep_traj_bug rep_stagnation gbs_oracle h2_ignition_uq_demo)
if [[ -n "${SOUNIO_CHEM_PROBES:-}" ]]; then
  read -r -a PROBES <<<"$SOUNIO_CHEM_PROBES"
else
  PROBES=("${DEFAULT_PROBES[@]}")
fi

# FAIL CLOSED, not just on a probe mismatch -- on having no probes to run at
# all. `-n "${SOUNIO_CHEM_PROBES:-}"` is true for a whitespace-only value
# (" "), and `read -a` on whitespace-only input splits to a ZERO-element
# array: PROBES=() reaches the loop below, the loop body never runs, FAILS
# stays 0, and the gate printed "all 0 probes match" and exited 0 -- a golden
# check that verified nothing, reporting success. A CI config value is part
# of this gate's evidence contract (it decides what got measured), so an
# input that produces no selection must be as loud as a probe that diverges.
if [[ ${#PROBES[@]} -eq 0 ]]; then
  echo "[chem-golden] FAIL: SOUNIO_CHEM_PROBES selected zero probes (raw value: '${SOUNIO_CHEM_PROBES:-}')" >&2
  exit 1
fi

# Seconds. Override per probe with SOUNIO_CHEM_TIMEOUT_<PROBE_UPPERCASE>=N --
# for ANY probe, not just rep_adiabatic_bug (a prior version only wired the
# override through for that one entry, so the documented knob was a no-op for
# the four probes this gate actually runs by default; unknown probes fall
# back to 1500, the original, adequate for the two cheap probes this gate
# started with).
# rep_adiabatic_bug's own ceiling is a hang detector for the rare manual run,
# not a target: ~26.2h measured, so 172800s (48h, ~1.8x margin) here. Nothing
# automated ever reaches this entry -- see SOUNIO_CHEM_PROBES above.
declare -A PROBE_TIMEOUT_S=(
  [rep_traj_bug]=900
  [rep_stagnation]=1200
  [rep_adiabatic_bug]=172800
  [gbs_oracle]=4800
  [h2_ignition_uq_demo]=6600
)

# probe_timeout_s <probe> -> the effective timeout: SOUNIO_CHEM_TIMEOUT_<P>
# uppercased (with any non-[A-Za-z0-9_] char, i.e. none expected in a probe
# name, mapped to _) if set, else PROBE_TIMEOUT_S[<probe>], else 1500.
probe_timeout_s() {
  local p="$1" var
  var="SOUNIO_CHEM_TIMEOUT_${p^^}"
  var="${var//[^A-Za-z0-9_]/_}"
  if [[ -n "${!var:-}" ]]; then
    printf '%s' "${!var}"
  else
    printf '%s' "${PROBE_TIMEOUT_S[$p]:-1500}"
  fi
}

FAILS=0
echo "[chem-golden] lean_single: $(md5sum "$ROOT_DIR/bin/souc-lean-single-x86_64" 2>/dev/null | cut -c1-8) bin/souc-lean-single-x86_64"
echo "[chem-golden] probes: ${PROBES[*]}"

declare -A PID
for p in "${PROBES[@]}"; do
  src="examples/chemistry/$p.sio"
  t="$(probe_timeout_s "$p")"
  (
    SOUNIO_SOUC_ENGINE=lean_single timeout "$t" "$SOUC" run "$src" >"$WORK/$p.txt" 2>"$WORK/$p.err"
    echo $? >"$WORK/$p.rc"
  ) &
  PID[$p]=$!
  echo "[chem-golden] started $p (pid ${PID[$p]}, timeout ${t}s)"
done
for p in "${PROBES[@]}"; do
  wait "${PID[$p]}"
done

for p in "${PROBES[@]}"; do
  gold="$GOLD/$p.lean_single.txt"
  out="$WORK/$p.txt"
  rc="$(cat "$WORK/$p.rc" 2>/dev/null || echo 1)"
  if [[ "$rc" != "0" ]]; then
    echo "[chem-golden] FAIL $p: run exited non-zero (rc=$rc)" >&2
    tail -5 "$WORK/$p.err" >&2
    FAILS=$((FAILS + 1))
    continue
  fi
  if [[ "${REGEN:-0}" == "1" ]]; then
    cp "$out" "$gold"
    echo "[chem-golden] wrote $gold"
    continue
  fi
  if [[ ! -f "$gold" ]]; then
    echo "[chem-golden] FAIL $p: no golden at $gold" >&2
    FAILS=$((FAILS + 1))
    continue
  fi
  if diff -u "$gold" "$out" >"$WORK/$p.diff"; then
    echo "[chem-golden] ok   $p ($(wc -l <"$out") lines identical)"
  else
    echo "[chem-golden] FAIL $p: output differs from golden" >&2
    head -40 "$WORK/$p.diff" >&2
    FAILS=$((FAILS + 1))
  fi
done
# Defense in depth alongside the empty-selection check above: this is what
# actually ran, measured independently of PROBES (a `.rc` file only exists if
# the probe's subshell reached its `echo $? >"$WORK/$p.rc"` line), so a
# regression that reintroduces a zero-probe run some other way -- not
# necessarily through SOUNIO_CHEM_PROBES -- is still caught here rather than
# read as "0 probes, 0 fails, green".
EXECUTED=0
for p in "${PROBES[@]}"; do
  [[ -f "$WORK/$p.rc" ]] && EXECUTED=$((EXECUTED + 1))
done
if [[ $EXECUTED -eq 0 ]]; then
  echo "[chem-golden] FAIL: 0 of ${#PROBES[@]} selected probe(s) actually executed" >&2
  exit 1
fi

[[ $FAILS -eq 0 ]] && echo "[chem-golden] all ${#PROBES[@]} probes match ($EXECUTED executed)" || echo "[chem-golden] $FAILS probe(s) diverge ($EXECUTED executed)" >&2
[[ $FAILS -eq 0 ]]
