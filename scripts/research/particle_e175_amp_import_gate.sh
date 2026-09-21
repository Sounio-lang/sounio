#!/usr/bin/env bash
# E175 trilogy gate: stdlib extern sweep + EXP13/14 dual-engine.
# Lives under scripts/research/ (scripts/ci/ may be claimed).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"

check_witness() {
  local label="$1"
  local src="$2"
  local out="/tmp/particle_e175_${label}_out.txt"
  local err="/tmp/particle_e175_${label}_err.txt"
  echo "== E175 witness: ${label} Madaros check =="
  set +e
  SOUNIO_SOUC_ENGINE=madaros ./bin/souc check "$src" >"$out" 2>"$err"
  local rc=$?
  set -e
  if ! grep -qE 'verdict=0|check: OK' "$out" "$err"; then
    echo "${label} Madaros check failed rc=${rc}" >&2
    tail -40 "$err" >&2
    exit 1
  fi
  if grep -q 'E175' "$out" "$err"; then
    echo "E175 still present on ${label} witness" >&2
    grep 'E175' "$out" "$err" >&2
    exit 1
  fi
}

WITNESS="$(mktemp /tmp/particle_e175_witness.XXXXXX.sio)"
PROP="$(mktemp /tmp/particle_e175_prop.XXXXXX.sio)"
cleanup() { rm -f "$WITNESS" "$PROP"; }
trap cleanup EXIT

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
check_witness lorentz_complex "$WITNESS"

cat >"$PROP" <<'EOF'
use complex::lib::{complex_new, complex_abs}
use particle_physics::propagator::{photon_propagator_sq}
use particle_physics::lorentz::{lorentz_new}
fn main() -> i32 with IO, Mut, Div, Panic {
    let p = lorentz_new(91.2, 0.0, 0.0, 0.0)
    let d = photon_propagator_sq(p)
    print_f64(complex_abs(complex_new(3.0, 4.0)) + d)
    print("\n")
    0
}
EOF
check_witness propagator_complex "$PROP"

echo "== EXP13 dual-engine =="
bash "$ROOT/scripts/ci/particle_exp13_amplitude_honesty_gate.sh"

echo '== EXP14 dual-engine with eemm_z_amplitude_nu =='
bash "$ROOT/scripts/research/particle_exp14_amp_xsec_gate.sh"

echo "PARTICLE_E175_TRILOGY_GATE_OK"
