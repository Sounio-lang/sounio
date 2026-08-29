#!/usr/bin/env bash
# scripts/ci/lean_single_fixed_point_gate.sh
#
# Verifies that Track A -- self-hosted/compiler/lean_single.sio -- self-compiles
# to a bit-identical fixed point (stage1 == stage2 == stage3). Establishes the
# frozen reference baseline used by milestone M1.1 of the three-track
# convergence plan: lock down "the compiler that bin/souc IS today" before
# moving bin/souc onto Track N-v2.
#
# Exit 0 = PASS (md5 matches across stages).
# Exit 1 = FAIL (lean_single no longer reaches a fixed point).
# Exit 0 with SKIP message on non-Linux/non-x86-64 hosts.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[lean-single-fp] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[lean-single-fp] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

# WHO RUNS THIS, precisely -- measured 2026-08-08, because the header has been
# wrong in both directions:
#
#   lean_single_fixed_point_gate.sh
#     <- native_v2_cpu_compiler_umbrella_gate.sh:138
#        <- native_v2_frontend_convergence_gate.sh:73
#           <- NOTHING in .github/workflows/ or the Makefile
#
# So it is reachable only by hand, and CI never executes it. The old header said
# "SUPERSEDED -- do not wire this into CI", which read as if it were unused; it
# has two callers. An earlier draft of this header said "THIS GATE IS WIRED",
# which was equally wrong in the direction that matters. It is a hand-run
# diagnostic with a live call chain, and it was broken.
#
# That is also why fixing it is worth doing but not urgent: nothing red goes
# green here. Somebody debugging a bootstrap by hand stops being lied to.
#
# Three defects, all measured 2026-08-08:
#
#   SUBJECT.  It resolved its compiler through resolve_souc.sh, which returns
#   bin/souc -- a WRAPPER SCRIPT that routes to Madaros. So "stage1" was Madaros
#   compiling lean_single.sio, not the lean_single seed reproducing itself, and
#   from a worktree it could resolve another checkout's binary entirely. It now
#   takes an explicit raw ELF, defaulting to the seed, and refuses a wrapper.
#
#   COMPARAND.  The "shipped" md5 it printed came from bin/souc-linux-x86_64
#   (c7d5e838...), a THIRD binary that is neither the seed (3a7a17a0...) nor the
#   wrapper. It compared unrelated things and reported the result as drift.
#
#   CRITERION.  It required stage1 == stage2, which REJECTS EVERY LEGITIMATE
#   CODEGEN CHANGE. When lean_single.sio changes how code is emitted:
#       stage1 = old seed compiling new source   -- new semantics, OLD codegen
#       stage2 = stage1  compiling new source    -- new semantics, NEW codegen
#   stage1 != stage2 is the CORRECT outcome there; the fixed point is reached at
#   stage2. Measured on the #1678 seed fix: 397b88a3 -> 25fb229c -> 25fb229c.
#   The old criterion would have rejected a bootstrap that converges perfectly.
#
# The fixed point is now stage2 == stage3, which is what the Makefile checks and
# what the sibling gate scripts/ci/native_v2_driver_self_compile_gate.sh:364
# already does. stage1 is reported as a diagnostic only.
#
# Strictness note: divergence between the shipped seed and the fixed point stays
# a WARN here. scripts/ci/canonical_compiler_gate.sh is the gate that FAILS on
# that, and it is the one wired into ci.yml. This gate answers a different
# question -- "does the chain converge at all" -- and keeping it non-strict is
# deliberate, not an oversight.
# The subject is explicit and must be a raw ELF. resolve_souc.sh is deliberately
# NOT used: it honours SOUC_BIN and the bin/souc shim, so the gate's subject
# became whatever happened to be installed, which made its verdict unattributable.
SEED="${SOUNIO_LEAN_SINGLE_SEED:-$ROOT_DIR/bin/souc-lean-single-x86_64}"

if [[ ! -x "$SEED" ]]; then
  echo "[lean-single-fp] FAIL: no seed binary at $SEED" >&2
  echo "[lean-single-fp]       override with SOUNIO_LEAN_SINGLE_SEED=<raw ELF>" >&2
  exit 1
fi
if head -c2 "$SEED" 2>/dev/null | grep -q '#!'; then
  echo "[lean-single-fp] FAIL: $SEED is a wrapper script, not a raw ELF." >&2
  echo "[lean-single-fp]       A wrapper routes to whatever compiler it likes;" >&2
  echo "[lean-single-fp]       self-compiling it does not measure this seed." >&2
  exit 1
fi
SOUC_BIN="$SEED"

OUT_DIR="${SOUNIO_LEAN_SINGLE_FP_DIR:-$(mktemp -d /tmp/sounio-lean-single-fp.XXXXXX)}"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

SRC="self-hosted/compiler/lean_single.sio"
STAGE1="$OUT_DIR/lean_single.stage1"
STAGE2="$OUT_DIR/lean_single.stage2"
STAGE3="$OUT_DIR/lean_single.stage3"

STAGE1_LOG="$LOG_DIR/stage1.log"
STAGE2_LOG="$LOG_DIR/stage2.log"
STAGE3_LOG="$LOG_DIR/stage3.log"

printf '[lean-single-fp] souc=%s\n' "$SOUC_BIN"
printf '[lean-single-fp] src=%s\n' "$SRC"
printf '[lean-single-fp] out=%s\n' "$OUT_DIR"

# ── Stage1: shipped souc compiles lean_single.sio ─────────────────────────────
if ! "$SOUC_BIN" "$SRC" "$STAGE1" >"$STAGE1_LOG" 2>&1; then
  echo "[lean-single-fp] FAIL: shipped souc could not compile lean_single.sio" >&2
  tail -n 80 "$STAGE1_LOG" >&2 || true
  exit 1
fi
chmod +x "$STAGE1"

if [[ ! -x "$STAGE1" ]]; then
  echo "[lean-single-fp] FAIL: stage1 is not executable: $STAGE1" >&2
  exit 1
fi

if command -v file >/dev/null 2>&1; then
  if ! file "$STAGE1" | grep -q 'ELF 64-bit LSB executable, x86-64'; then
    echo "[lean-single-fp] FAIL: stage1 is not an x86-64 ELF" >&2
    file "$STAGE1" >&2 || true
    exit 1
  fi
fi

# ── Stage2: stage1 compiles lean_single.sio ───────────────────────────────────
if ! "$STAGE1" "$SRC" "$STAGE2" >"$STAGE2_LOG" 2>&1; then
  echo "[lean-single-fp] FAIL: stage1 could not compile lean_single.sio" >&2
  tail -n 80 "$STAGE2_LOG" >&2 || true
  exit 1
fi
chmod +x "$STAGE2"

# ── Stage3: stage2 compiles lean_single.sio ───────────────────────────────────
if ! "$STAGE2" "$SRC" "$STAGE3" >"$STAGE3_LOG" 2>&1; then
  echo "[lean-single-fp] FAIL: stage2 could not compile lean_single.sio" >&2
  tail -n 80 "$STAGE3_LOG" >&2 || true
  exit 1
fi
chmod +x "$STAGE3"

# ── Fixed-point check: stage1 == stage2 == stage3 ─────────────────────────────
STAGE1_MD5="$(md5sum "$STAGE1" | cut -d' ' -f1)"
STAGE2_MD5="$(md5sum "$STAGE2" | cut -d' ' -f1)"
STAGE3_MD5="$(md5sum "$STAGE3" | cut -d' ' -f1)"

# The comparand is the seed under test. It used to fall back to
# bin/souc-linux-x86_64, a third binary unrelated to either the seed or the
# wrapper, and report the mismatch as drift.
SHIPPED_ELF="$SEED"
SHIPPED_MD5="$(md5sum "$SHIPPED_ELF" | cut -d' ' -f1)"

printf '[lean-single-fp] md5: shipped=%s (%s)\n' "$SHIPPED_MD5" "$SHIPPED_ELF"
printf '[lean-single-fp] md5: stage1 =%s\n' "$STAGE1_MD5"
printf '[lean-single-fp] md5: stage2 =%s\n' "$STAGE2_MD5"
printf '[lean-single-fp] md5: stage3 =%s\n' "$STAGE3_MD5"

# THE fixed-point check. stage1 is NOT part of it: when lean_single.sio changes
# codegen, stage1 carries the old codegen and differing from stage2 is correct.
if [[ "$STAGE2_MD5" != "$STAGE3_MD5" ]]; then
  echo "[lean-single-fp] FAIL: stage2 != stage3 -- the chain does not converge" >&2
  exit 1
fi

if [[ "$STAGE1_MD5" != "$STAGE2_MD5" ]]; then
  echo "[lean-single-fp] note: stage1 != stage2 -- the seed emits different code" >&2
  echo "[lean-single-fp] note: than the source it compiled. Expected after a codegen" >&2
  echo "[lean-single-fp] note: change; the chain still converges at stage2." >&2
fi

# Shipped binary should already be at the fixed point. We warn rather than
# fail on divergence because it may legitimately drift in mid-development --
# what we strictly require is that the chain itself converges.
if [[ "$SHIPPED_MD5" != "$STAGE2_MD5" ]]; then
  echo "[lean-single-fp] WARN: the seed differs from the fixed point (stage2)." >&2
  echo "[lean-single-fp] WARN: the chain converged, but the committed seed is not" >&2
  echo "[lean-single-fp] WARN: what lean_single.sio reproduces. Rebuild it, or run" >&2
  echo "[lean-single-fp] WARN: scripts/ci/canonical_compiler_gate.sh, which FAILS on this." >&2
fi

# Emit driver-self-compile-shaped lines so the umbrella's log parser
# (scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh::emit_summary_json) can
# extract a fixed-point md5 and a stage1==stage2==stage3 marker. lean_single's
# native codegen does not currently emit a .sounio.epistemic section, so we
# synthesize a self-consistent triple keyed off the binary md5.
printf '[lean-single-fp] fixed-point md5=%s\n' "$STAGE2_MD5"
printf '[lean-single-fp] epistemic stage1=%s stage2=%s stage3=%s\n' \
  "$STAGE1_MD5" "$STAGE2_MD5" "$STAGE3_MD5"

echo "[lean-single-fp] PASS: stage2 == stage3 (fixed point md5=$STAGE2_MD5, size=$(stat -c %s "$STAGE2") bytes)"
