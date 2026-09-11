#!/usr/bin/env bash
# The GUM rules in epistemic::knowledge carry their covariance term when asked.
#
# ep_add/ep_sub/ep_mul/ep_div drop Cov(X,Y), so they are the independent case
# only. Nothing said so, and nothing offered the term, so the correlated use had
# no correct spelling: with x = {val 3.0, variance 0.25}, ep_mul(&x,&x) reported
# 4.5 where the truth is 9.0 and ep_square(&x), in the same module, reported 9.0.
# Understating variance claims more certainty than is held, which is the unsafe
# direction. The E230 noise-set machinery does not reach it -- Epistemic is a
# plain struct, not Knowledge<T>, so the checker accepts the correlated call.
#
# This gate pins the covariance-aware forms against their closed answers for a
# single variable (Cov(X,X) = Var(X)), and pins that cov = 0 still reproduces
# the independent rules byte-for-byte.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# Always pin this worktree's stdlib (never inherit a foreign SOUNIO_STDLIB_PATH).
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
unset SOUNIO_SOUC_ENGINE || true
SOUC="${SOUC:-$ROOT/bin/souc}"
SRC="tests/run-pass/ep_gum_covariance.sio"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
ELF="$OUT/ep_gum_cov.elf"

echo "== ep_gum_covariance_gate =="

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

grep -q 'EP_GUM_COV_OK' "$OUT/run.log" || {
  echo "FAIL: missing sentinel"
  cat "$OUT/run.log" || true
  exit 1
}

# The independence precondition must stay stated. These comments are the only
# thing standing between a caller and a silently understated variance, and a
# tidy-up that deletes them would leave the formulas looking like identities.
for marker in 'UNCORRELATED case only' 'ep_add_cov' 'ep_mul_cov'; do
  grep -qF "$marker" stdlib/epistemic/knowledge.sio || {
    echo "FAIL: stdlib/epistemic/knowledge.sio no longer mentions '$marker'"
    exit 1
  }
done

echo "MADAROS_EP_GUM_COVARIANCE_GATE_OK"
