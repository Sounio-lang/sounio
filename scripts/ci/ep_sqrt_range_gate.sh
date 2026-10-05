#!/usr/bin/env bash
# ep_sqrt is a square root across the f64 range, not only near 1.
#
# stdlib/epistemic/knowledge.sio ran ten Heron steps from y = x. Heron is
# quadratic only near the root; from y = x it spends ~log2(sqrt(x)) steps just
# getting close, so a fixed step count bounds the range over which the result is
# a square root at all. Measured 2026-09-09, relative residual |y*y - x| / x:
# accurate for x in about [1e-4, 1e4], 0.68 at 1e6, 95 at 1e8, and worse beyond.
#
# It is not a private helper: ep_std, ep_sqrt_ep, ep_rel_uncertainty and
# ep_budget are all thin wrappers over ep_sqrt(variance), so a variance of 1e-6
# -- an ordinary measurement uncertainty -- made ep_std report a standard
# deviation 29.6% off. That is why this has a gate rather than only a fixture.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# Always pin this worktree's stdlib (never inherit a foreign SOUNIO_STDLIB_PATH).
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
unset SOUNIO_SOUC_ENGINE || true
SOUC="${SOUC:-$ROOT/bin/souc}"
SRC="tests/run-pass/ep_sqrt_range.sio"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
ELF="$OUT/ep_sqrt_range.elf"

echo "== ep_sqrt_range_gate =="

if ! "$SOUC" compile "$SRC" -o "$ELF" >"$OUT/compile.log" 2>&1; then
  echo "FAIL: compile"
  tail -40 "$OUT/compile.log" || true
  exit 1
fi
chmod +x "$ELF"

# Capture rc BEFORE anything else runs: `$?` inside an `if !` body is the `if`'s
# own status, not the command's, and reading it there reports 0 for a crash.
set +e
"$ELF" >"$OUT/run.log" 2>&1
RUN_RC=$?
set -e
if [ "$RUN_RC" -ne 0 ]; then
  echo "FAIL: run (rc $RUN_RC)"
  cat "$OUT/run.log" || true
  exit 1
fi

grep -q 'EP_SQRT_RANGE_OK' "$OUT/run.log" || {
  echo "FAIL: missing sentinel"
  cat "$OUT/run.log" || true
  exit 1
}

# The four ep_sqrt copies must not drift apart again: identical bodies, all
# range-reduced. A copy that regresses to the unreduced form would pass the
# fixture above (which only exercises epistemic::knowledge) while silently
# returning nonsense to prob, simulation and fairness callers.
mapfile -t COPIES < <(grep -rl 'fn ep_sqrt(' stdlib --include='*.sio' | LC_ALL=C sort)
if [ "${#COPIES[@]}" -lt 2 ]; then
  echo "FAIL: expected several ep_sqrt copies, found ${#COPIES[@]}"
  exit 1
fi
REF=""
for f in "${COPIES[@]}"; do
  n=$(grep -n 'fn ep_sqrt(' "$f" | head -1 | cut -d: -f1)
  body=$(awk -v s="$n" 'NR>s{print} NR>s && /^}$/{exit}' "$f" | md5sum | cut -d' ' -f1)
  if ! grep -q 'scale_back' <(awk -v s="$n" 'NR>s{print} NR>s && /^}$/{exit}' "$f"); then
    echo "FAIL: $f has an ep_sqrt without range reduction"
    exit 1
  fi
  if [ -z "$REF" ]; then REF="$body"; REFF="$f"; continue; fi
  if [ "$body" != "$REF" ]; then
    echo "FAIL: ep_sqrt body in $f differs from $REFF"
    exit 1
  fi
done
echo "  ${#COPIES[@]} ep_sqrt copies, identical and range-reduced"

echo "EP_SQRT_RANGE_GATE_OK"
