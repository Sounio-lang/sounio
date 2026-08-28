#!/usr/bin/env bash
# scripts/ci/dissertation_confidence_gate_gate.sh
#
# Dissertation Contribution 2 — compile-time confidence gates — evidence gate.
#
# Demonstrates BOTH sides of the contract on the real rapamycin PBPK28 priors,
# now carried as Knowledge<f64, ε> so the literature evidence-quality ε flows
# into the EpistemicComplete gate:
#
#   1. run-pass  — a dosing summary declared at the HONEST ceiling
#                  (with Epistemic(400), the weakest prior) compiles and runs.
#   2. compile-fail — the SAME hepatic-clearance prior (ε=0.65) in a function
#                  demanding with Epistemic(950) is REJECTED at compile time
#                  (EpistemicComplete violation, conf=650 < 950, no binary).
#
# Uses the self-hosted souc CLI convention `souc <src> <out>` (NOT `souc run`),
# then executes the emitted ELF. CPU-only, a few seconds. Self-skips if souc is
# missing or non-Linux/x86-64.
#
# Exit 0 = PASS. Exit 1 = FAIL.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[confidence-gate] SKIP: Linux-only gate" >&2
  exit 0
fi
case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *) echo "[confidence-gate] SKIP: x86-64 Linux-only gate" >&2; exit 0 ;;
esac

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

STAGE_DIR="$(mktemp -d /tmp/dissertation_confidence_gate_XXXXXX)"
PASS_SRC="tests/run-pass/dissertation_pbpk28_confidence_gate.sio"
FAIL_SRC="tests/compile-fail/dissertation_pbpk28_overclaim.sio"

echo "=== Dissertation Confidence-Gate Gate (Contribution 2) ==="
echo "  souc=$SOUC_BIN"

fails=0

# ── 1. run-pass: honest ceiling compiles + runs ──────────────────────────────
echo ""
echo "[honest_ceiling] $PASS_SRC"
PASS_ELF="$STAGE_DIR/confidence_pass.elf"
PASS_LOG="$STAGE_DIR/confidence_pass.log"
if [[ ! -f "$PASS_SRC" ]]; then
  echo "  FAIL: source missing"; fails=$((fails + 1))
else
  set +e
  "$SOUC_BIN" "$PASS_SRC" "$PASS_ELF" >"$PASS_LOG" 2>&1
  c_rc=$?
  set -e
  [[ -f "$PASS_ELF" ]] && chmod +x "$PASS_ELF"
  if [[ $c_rc -ne 0 || ! -x "$PASS_ELF" ]]; then
    echo "  FAIL: honest ceiling did not compile (rc=$c_rc)"
    tail -5 "$PASS_LOG" | sed 's/^/    /'
    fails=$((fails + 1))
  else
    set +e
    run_out="$("$PASS_ELF" 2>&1)"
    r_rc=$?
    set -e
    if [[ $r_rc -ne 0 ]]; then
      echo "  FAIL: emitted binary did not run (rc=$r_rc)"; fails=$((fails + 1))
    elif ! grep -q "PBPK28_CONFIDENCE_GATE_PASS" <<<"$run_out"; then
      echo "  FAIL: no PBPK28_CONFIDENCE_GATE_PASS marker"
      echo "$run_out" | sed 's/^/    /'
      fails=$((fails + 1))
    else
      echo "  PASS (compiled at Epistemic(400), ran, marker present)"
    fi
  fi
fi

# ── 2. compile-fail: over-claim must be REJECTED by the gate ─────────────────
echo ""
echo "[overclaim_reject] $FAIL_SRC"
FAIL_ELF="$STAGE_DIR/confidence_fail.elf"
FAIL_LOG="$STAGE_DIR/confidence_fail.log"
if [[ ! -f "$FAIL_SRC" ]]; then
  echo "  FAIL: source missing"; fails=$((fails + 1))
else
  rm -f "$FAIL_ELF"
  set +e
  "$SOUC_BIN" "$FAIL_SRC" "$FAIL_ELF" >"$FAIL_LOG" 2>&1
  f_rc=$?
  set -e
  if [[ $f_rc -eq 0 || -f "$FAIL_ELF" ]]; then
    echo "  FAIL: over-claim compiled clean (confidence gate not enforcing)"
    fails=$((fails + 1))
  elif ! grep -qE "EpistemicComplete violation" "$FAIL_LOG"; then
    echo "  FAIL: rejected but not by the EpistemicComplete gate (rc=$f_rc)"
    tail -5 "$FAIL_LOG" | sed 's/^/    /'
    fails=$((fails + 1))
  else
    echo "  PASS (rejected by EpistemicComplete gate, conf=650 < 950, no binary)"
  fi
fi

# ── 3. soundness: non-literal ε cannot satisfy the gate (no confidence laundering) ──
echo ""
NL_NAME="nonliteral_eps_reject"
NL_SRC="tests/compile-fail/knowledge_nonliteral_eps_ungated.sio"
NL_LOG="$STAGE_DIR/nonliteral_eps.log"
echo "[$NL_NAME] $NL_SRC"
NL_ELF="$STAGE_DIR/nonliteral_eps.elf"
if [[ ! -f "$NL_SRC" ]]; then
  echo "  FAIL: source missing"; fails=$((fails + 1))
else
  rm -f "$NL_ELF"
  set +e
  "$SOUC_BIN" "$NL_SRC" "$NL_ELF" >"$NL_LOG" 2>&1
  n_rc=$?
  set -e
  if [[ $n_rc -eq 0 || -f "$NL_ELF" ]]; then
    echo "  FAIL: non-literal ε compiled clean (confidence laundering hole open)"
    fails=$((fails + 1))
  elif ! grep -qE "EpistemicComplete violation" "$NL_LOG"; then
    echo "  FAIL: rejected but not by the EpistemicComplete gate (rc=$n_rc)"
    tail -5 "$NL_LOG" | sed 's/^/    /'
    fails=$((fails + 1))
  else
    echo "  PASS (non-literal ε rejected, conf=1 < N, no laundering)"
  fi
fi

echo ""
if [[ $fails -ne 0 ]]; then
  echo "dissertation_confidence_gate_gate: FAIL ($fails check(s) failed)"
  exit 1
fi
echo "dissertation_confidence_gate_gate: PASS (compile-time confidence gates enforce on real rapamycin priors)"
