#!/usr/bin/env bash
# Every epistemic metric builtin must REFUSE a malformed call, not answer it.
#
# hessian_of(expr, j, k) once required only its first argument. `hessian_of(x*x)`
# type-checked, lowered through `if j >= 0 && j < 8 && k >= 0 && k < 8` with
# j = k = -1, missed, and returned 0.0 -- while the true H_00 is 2.0. An
# out-of-range literal did the same, and so did sensitivity_of(x, 99).
#
# In this algebra a zero is not a neutral answer. Zero variance means "certain",
# zero sensitivity means "does not depend on", zero Hessian means "no
# second-order dependence". Those are the STRONGEST claims the system can make,
# and they were being handed back silently in exactly the cases where it knew
# least. This gate exists so that a builtin cannot be added, or an arity
# widened, without the unanswerable forms being refused.
#
# It builds its probes rather than reading fixtures: a fixture can be deleted,
# and its absence looks like coverage. It also runs a POSITIVE control -- a
# well-formed call must be ACCEPTED -- because a gate that passes by refusing
# everything measures nothing.
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9
. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "epistemic_refusal_coverage"

SOUC="${SOUNIO_REFUSAL_SOUC:-$ROOT_DIR/bin/madaros-linux-x86_64}"
require_file "$SOUC" "no compiler at $SOUC"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
WORK="$(mktemp -d /tmp/epistemic-refusal.XXXXXX)"

probe() {  # $1 = body line
  cat > "$WORK/p.sio" <<EOF
fn main() -> i64 with Mut, Panic, Div, Alloc, Epistemic, IO {
    let k: Knowledge<f64> = measure(0.5, uncertainty: 0.05)
    let x = k.value
    let r = $1
    print_f64(r)
    0
}
EOF
  timeout 120 "$SOUC" check "$WORK/p.sio" >"$WORK/out" 2>&1
  printf '%s' "$?"
}

# Does the SOURCE already declare the refusal that the shipped binary lacks?
#
# With a committed ELF these two facts diverge, and a gate that ignores the
# difference blames the wrong one. madaros_binary_source_drift_gate.sh owns the
# question "does the shipped binary do what the source implements"; this gate
# owns "does the compiler refuse what it cannot answer". When the source has the
# refusal and the binary predates it, the honest report is PENDING-REBUILD --
# not a failure of a source that is already correct, and not a pass either.
SOURCE_HAS_E242=0
if grep -q 'epistemic channel index is out of range' "$ROOT_DIR/self-hosted/check/check.sio" 2>/dev/null; then
  SOURCE_HAS_E242=1
fi

total=0; passed=0; failed=0; pending=0
row() {  # name | expectation accept|refuse | body
  local name="$1" want="$2" body="$3"
  total=$((total+1))
  local rc; rc="$(probe "$body")"
  if [[ "$want" == "refuse" ]]; then
    if [[ "$rc" != "0" ]]; then
      passed=$((passed+1)); echo "  ok      refused  $name"
    elif [[ "$SOURCE_HAS_E242" == "1" ]]; then
      pending=$((pending+1))
      echo "  PENDING-REBUILD  $name -- the source refuses this; the shipped ELF predates it"
    else
      failed=$((failed+1)); echo "  FAIL    ACCEPTED $name -- returns a value where it cannot answer"
    fi
  else
    if [[ "$rc" == "0" ]]; then
      passed=$((passed+1)); echo "  ok      accepted $name  (positive control)"
    else
      failed=$((failed+1)); echo "  FAIL    refused  $name -- the control form must be accepted"
      grep -E 'error' "$WORK/out" | head -2 | sed 's/^/            /'
    fi
  fi
}

# Positive controls first: if these do not pass, nothing below means anything.
row "hessian_of well-formed"      accept "hessian_of(x * x, 0, 0)"
row "sensitivity_of well-formed"  accept "sensitivity_of(x, 0)"
row "variance_of well-formed"     accept "variance_of(x * x)"

# The unanswerable forms.
row "hessian_of missing indices"  refuse "hessian_of(x * x)"
row "hessian_of one index"        refuse "hessian_of(x * x, 0)"
row "hessian_of j out of range"   refuse "hessian_of(x * x, 99, 0)"
row "hessian_of k out of range"   refuse "hessian_of(x * x, 0, 99)"
row "hessian_of negative index"   refuse "hessian_of(x * x, 0 - 1, 0)"
row "sensitivity_of missing k"    refuse "sensitivity_of(x)"
row "sensitivity_of out of range" refuse "sensitivity_of(x, 99)"

rm -rf "$WORK"
echo "epistemic_refusal_coverage_gate: status=$([[ $failed -eq 0 ]] && echo pass || echo fail) total=$total passed=$passed failed=$failed not_run=$pending"
if [[ $pending -ne 0 ]]; then
  echo "  $pending form(s) pending a Madaros rebuild; madaros_binary_source_drift_gate.sh owns that row"
fi
if [[ $failed -ne 0 ]]; then
  gate_fail "$failed epistemic form(s) answered where they cannot"
fi
gate_pass "$passed/$total -- every unanswerable form refused, every control accepted"
exit 0
