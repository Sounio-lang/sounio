#!/usr/bin/env bash
# E175 residual closeout: complex+lorentz must check under Madaros, and EXP13
# dual-engine green. Lives under scripts/research/ (scripts/ci/ may be claimed).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"

WITNESS="$(mktemp /tmp/particle_e175_witness.XXXXXX.sio)"
trap 'rm -f "$WITNESS"' EXIT
cat >"$WITNESS" <<'EOF'
use complex::lib::{complex_new, complex_mul}
use particle_physics::lorentz::{lorentz_new, metric_eta, rapidity}
fn main() -> i32 with IO, Mut, Div, Panic {
    let z = complex_mul(complex_new(1.0, 2.0), complex_new(3.0, 4.0))
    let p = lorentz_new(100.0, 0.0, 0.0, 50.0)
    print_f64(rapidity(p) + metric_eta(0, 0) + z.re)
    print("\n")
    0
}
EOF

echo "== E175 witness: complex+lorentz Madaros check =="
set +e
SOUNIO_SOUC_ENGINE=madaros ./bin/souc check "$WITNESS" >/tmp/particle_e175_witness_out.txt 2>/tmp/particle_e175_witness_err.txt
rc=$?
set -e
if ! grep -q 'check: OK' /tmp/particle_e175_witness_out.txt && ! grep -q 'verdict=0' /tmp/particle_e175_witness_err.txt; then
  # Madaros prints verdict on stderr in some paths; accept either stream.
  if ! grep -qE 'verdict=0|check: OK' /tmp/particle_e175_witness_out.txt /tmp/particle_e175_witness_err.txt; then
    echo "complex+lorentz Madaros check failed rc=$rc" >&2
    tail -40 /tmp/particle_e175_witness_err.txt >&2
    exit 1
  fi
fi
if grep -q 'E175' /tmp/particle_e175_witness_out.txt /tmp/particle_e175_witness_err.txt; then
  echo "E175 still present on complex+lorentz witness" >&2
  grep 'E175' /tmp/particle_e175_witness_out.txt /tmp/particle_e175_witness_err.txt >&2
  exit 1
fi

echo "== EXP13 dual-engine (canonical CI gate) =="
if [[ -x "$ROOT/scripts/ci/particle_exp13_amplitude_honesty_gate.sh" ]]; then
  bash "$ROOT/scripts/ci/particle_exp13_amplitude_honesty_gate.sh"
else
  echo "missing scripts/ci/particle_exp13_amplitude_honesty_gate.sh" >&2
  exit 1
fi

echo "PARTICLE_E175_AMP_IMPORT_GATE_OK"
