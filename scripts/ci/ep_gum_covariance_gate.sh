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

# Refusal probes: every input outside the covariance-matrix domain must be REFUSED
# at construction, never turned into a variance. There is no run-fail test harness,
# so each probe is compiled and run here and must ABORT (non-zero exit) WITHOUT
# printing the escape sentinel. A clean exit or the sentinel means inconsistent
# evidence reached a variance again, and ep_merge reads a zero or NaN variance as
# near-certainty.
#   nonpsd   Var 0.25, 0.09, Cov 99 (99^2 >> 0.0225)
#   slack    Var 4, 4, Cov 4*(1 + 2e-11): the case the old 1e-10 slack accepted
#   negvar   Var -1, -1, Cov 0: Cov^2 <= Var X * Var Y holds, S is negative definite
#   nancov   Cov = inf - inf
#   infvar   Var X = 1e200 * 1e200: infinite, refused as non-finite
#   ovfprod  Var X 1e200, Var Y 1e200, Cov 0: finite variances whose product
#            overflows, so the bound cannot be evaluated
#   zerovar  Var X 0, Var Y 1, Cov 1e-300: the bound is exactly 0
probe() {
  local name="$1" op="$2" va="$3" vb="$4" cov="$5"
  local src="$OUT/ep_gum_cov_refuse_${name}.sio" elf="$OUT/ep_gum_cov_refuse_${name}.elf"
  cat > "$src" <<PROBE
//@ run-pass
use epistemic::knowledge::{Epistemic, ${op}}
fn main() -> i32 with IO, Div, Panic {
    let t = 1.0e200
    let big = t * t
    let x = Epistemic { val: 3.0, variance: ${va}, confidence: 900 }
    let y = Epistemic { val: 2.0, variance: ${vb}, confidence: 900 }
    // Outside the domain: must panic here, never reach the print below.
    let r = ${op}(&x, &y, ${cov})
    print("EP_GUM_COV_REFUSE_ESCAPED ")
    print_int(r.variance as i64)
    print("\n")
    return 0
}
PROBE
  if ! "$SOUC" compile "$src" -o "$elf" >"$OUT/probe_${name}_compile.log" 2>&1; then
    echo "FAIL: refusal probe '$name' did not compile"
    tail -40 "$OUT/probe_${name}_compile.log" || true
    exit 1
  fi
  chmod +x "$elf"
  set +e
  "$elf" >"$OUT/probe_${name}_run.log" 2>&1
  local rc=$?
  set -e
  if [ "$rc" -eq 0 ] || grep -q 'EP_GUM_COV_REFUSE_ESCAPED' "$OUT/probe_${name}_run.log"; then
    echo "FAIL: refusal probe '$name' was NOT refused"
    cat "$OUT/probe_${name}_run.log" || true
    exit 1
  fi
  echo "  refusal probe ok: $name (rc $rc)"
}
probe nonpsd  ep_sub_cov 0.25 0.09 99.0
probe slack   ep_sub_cov 4.0 4.0 "4.0 * (1.0 + 2.0e-11)"
probe negvar  ep_add_cov "0.0 - 1.0" "0.0 - 1.0" 0.0
probe nancov  ep_mul_cov 0.25 0.09 "big - big"
probe infvar  ep_div_cov big 0.09 0.0
probe ovfprod ep_add_cov 1.0e200 1.0e200 0.0
probe zerovar ep_add_cov 0.0 1.0 1.0e-300

echo "MADAROS_EP_GUM_COVARIANCE_GATE_OK"
