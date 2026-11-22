#!/usr/bin/env bash
# #846: the per-module `exp` helpers in stdlib must agree with the shared
# math::pure::exp to 1e-12 relative over x in [-700, 700].
#
# The helpers were hand-rolled Taylor series around 0 (some with reflection only
# beyond |x| = 20, some with none), so for moderate negative x the alternating
# terms cancelled catastrophically: at x = -13.855 (the #841 witness) 32 of the
# 35 copies returned values off by 1e7..1e10, many of them negative, and five
# were already wrong at x = -5. They now delegate to math::pure::exp, which
# tests/run-pass/math_pure_exp_full_range.sio checks against libm.
#
# A run-pass fixture cannot test them in place: they are private module
# helpers and Madaros refuses to import a private fn. So this gate extracts each
# helper VERBATIM from its file (from `fn NAME(` to the first column-0 `}`),
# renames it to keep the bodies apart, gives it a uniform effect row, and
# evaluates it next to math::pure::exp in one generated program. Whatever the
# file says is what is measured: a helper that goes back to a local Taylor
# series fails here.
#
# Engine: bin/souc (Madaros) by default; set SOUNIO_SOUC_ENGINE=lean_single to
# run the same program through the seed. CI runs both.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
SOUC="${SOUC:-$ROOT/bin/souc}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT

echo "== stdlib_exp_helpers_gate (engine: ${SOUNIO_SOUC_ENGINE:-default}) =="

# file  helper  [local fns the helper calls, copied along]
# (the #846 inventory, paths as of 2026-10-07)
HELPERS="
stdlib/darwin_pbpk/compartments/brain.sio dpbpk_brain_exp
stdlib/darwin_pbpk/compartments/kidney.sio dpbpk_kidney_exp
stdlib/darwin_pbpk/compartments/liver.sio dpbpk_liver_exp
stdlib/darwin_pbpk/constants.sio dpbpk_exp
stdlib/darwin_pbpk/core/fractal_blood.sio dpbpk_fb_exp
stdlib/darwin_pbpk/core/ode_pbpk.sio ode_exp
stdlib/darwin_pbpk/core/rodgers_rowland.sio rr_exp
stdlib/darwin_pbpk/ddi/mechanistic_ddi.sio dpbpk_mechanistic_ddi_exp
stdlib/darwin_pbpk/ml/gnn_inference.sio gnn_exp
stdlib/darwin_pbpk/pd/hill_mtor.sio hill_exp
stdlib/darwin_pbpk/simulation.sio dpbpk_simulation_exp
stdlib/epistemic/klibanoff.sio kl_exp_small
stdlib/epistemic/ode.sio ode_exp
stdlib/epistemic/uncertain_eq.sio ueq_exp_core
stdlib/fmri/connectivity.sio fmri_exp_inner
stdlib/fmri/helpers.sio fmri_exp
stdlib/gpu/smooth.sio smooth_exp
stdlib/medlang/pk/multi_compartment.sio mc_exp
stdlib/medlang/population/simulation.sio ps_exp
stdlib/ml/classifier.sio clf_exp
stdlib/nn/activation.sio exp_val
stdlib/nn/dense_layer.sio exp_val
stdlib/nn/epistemic_backward.sio bk_exp
stdlib/nn/epistemic_layer.sio exp_el
stdlib/ode/solver.sio solver_exp_approx
stdlib/physics/thermo.sio thermo_exp thermo_pow2_bit thermo_pow2_pos thermo_pow2
stdlib/qnn/activation.sio qnn_exp
stdlib/snn/base.sio sed_exp
stdlib/snn/benchmark.sio sed_exp
stdlib/snn/pac_training.sio sed_exp
stdlib/snn/sedenion_layer_impl.sio sed_exp
stdlib/snn/sedenion_training.sio sed_exp
stdlib/ssm/lib.sio ssm_exp
stdlib/viz/epiviz.sio epiviz_exp
stdlib/autodiff/dual.sio exp_f64
stdlib/autodiff/epistemic_dual.sio exp_f64
stdlib/autodiff/grad.sio exp_f64
stdlib/autodiff/tape.sio exp_f64
stdlib/chemistry/equilibrium.sio exp_approx
stdlib/epistemic/combine.sio exp_f64
stdlib/epistemic/meta.sio exp_f64
stdlib/fmri/connectivity_epistemic.sio exp_f64
stdlib/fmri/preprocess.sio exp_f64
stdlib/ml/autodiff.sio exp_f64
stdlib/ml/gp.sio gp_exp_f64
stdlib/ml/tensor.sio tensor_exp_f64
stdlib/nn/autograd.sio exp_f64
stdlib/nn/dense2.sio exp_f64
stdlib/nn/dense_quaternion.sio exp_f64
stdlib/nn/differentiable.sio exp_diff
stdlib/nn/mlp_classifier.sio mlp_exp_f64
stdlib/nn/mlp_xor.sio exp_f64
stdlib/nn/quaternion.sio exp_f64
stdlib/particle_physics/pdf.sio exp_f64
stdlib/pbpk/covariate.sio cov_exp_f64
stdlib/pbpk/error_models.sio em_exp_f64
stdlib/pbpk/population.sio exp_f64
stdlib/prob/inference.sio inference_exp_f64
stdlib/prob/normal.sio exp_approx
stdlib/prob/primitives.sio prob_f64_exp
stdlib/snn/sedenion_layer_core.sio exp_f64
stdlib/stats/shapiro_wilk.sio sw_exp
"
# Helpers that clamp their argument to [-20, 20] by contract (activation
# saturation, PK underflow guard); swept on the grid points inside that range.
# The second batch of rows above and these five were found by
# the series lint below (2026-10-12): 35 more copies of the same bug, at
# x = -13.855 off by up to 4e10, some already wrong at x = -5.
CLAMP20="
stdlib/medlang/pk/one_compartment.sio pk1_exp
stdlib/medlang/pk/two_compartment.sio pk2_exp
stdlib/medlang/pk/three_compartment.sio pk3_exp
stdlib/qnn/attention.sio qatt_exp
stdlib/qnn/recurrent.sio qgru_exp
"
if [[ -n "${EXP_HELPERS_FILE:-}" ]]; then HELPERS="$(cat "$EXP_HELPERS_FILE")"; fi
# darwin_pbpk/scenarios/venlafaxine_xr.sio mer_exp_neg is exp(x) for x < 0 only
# (it returns 1 for x >= 0 by contract); it is checked on the negative half.
NEG_ONLY="stdlib/darwin_pbpk/scenarios/venlafaxine_xr.sio mer_exp_neg"

GRID="-700.0 -500.0 -100.0 -30.0 -20.0 -15.0 -13.855 -10.0 -5.0 -2.5 -1.0 -0.5 0.0 0.5 1.0 2.5 5.0 10.0 13.855 20.0 50.0 100.0 500.0 700.0"

NEG_START=0
for x in $GRID; do [[ "$x" == -* ]] && NEG_START=$((NEG_START + 1)); done
GATE_N=$(wc -w <<<"$GRID")

prog="$OUT/exp_helpers.sio"
calls="$OUT/calls.sio"
: >"$calls"
{
  echo "use math::pure::{exp}"
  echo
  # grid point i (the first NEG_START points are the negative half)
  echo "fn gate_x(i: i64) -> f64 {"
  i=0
  for x in $GRID; do
    xl="$x"; [[ "$x" == -* ]] && xl="0.0 - ${x#-}"
    echo "    if i == $i { return $xl }"
    i=$((i + 1))
  done
  echo "    0.0"
  echo "}"
  echo
} >"$prog"
cat >>"$prog" <<'SIO'
fn gate_rel_err(a: f64, b: f64) -> f64 {
    let d = a - b
    let ad = if d < 0.0 { 0.0 - d } else { d }
    let ab = if b < 0.0 { 0.0 - b } else { b }
    ad / ab
}

fn gate_bad(x: f64, got: f64) -> i64 with Mut, Div, Panic {
    let e = gate_rel_err(got, exp(x))
    if e <= 1.0e-12 { return 0 }
    1
}

fn gate_report(label: string, bad: i64, at: f64) -> i64 with IO, Mut, Div, Panic {
    if bad == 0 { return 0 }
    print("HELPER_FAIL ")
    print(label)
    print(" bad_points=")
    print_int(bad)
    print(" helper/exp@-13.855=")
    print_f64(at / exp(0.0 - 13.855))
    print("\n")
    return 1
}

SIO

n=0
extract() { # file fn -> body on stdout
  awk -v n="$2" '
    $0 ~ "^(pub )?fn " n "\\(" { p = 1 }
    p { print }
    p && /^}/ { exit }' "$1"
}
emit() { # lo hi file name [deps...]   (sweeps grid points [lo, hi))
  local lo="$1" hi="$2" file="$3" name="$4"; shift 4
  n=$((n + 1))
  local id="h${n}_${name}"
  [[ -f "$file" ]] || { echo "FAIL: $file missing"; exit 1; }
  # rename every copied fn (and its calls) with the helper's prefix, drop the
  # visibility, give the helper itself a uniform effect row
  local ren="s/\\b${name}\\(/${id}(/g" d
  for d in "$@"; do ren="$ren; s/\\b${d}\\(/h${n}_${d}(/g"; done
  for d in "$@"; do
    local db; db="$(extract "$file" "$d")"
    # a missing dep is not fatal by itself: if the helper calls it, the
    # generated program fails to compile and the gate fails there
    [[ -n "$db" ]] || { echo "note: fn $d not in $file"; continue; }
    printf '%s\n\n' "$(printf '%s\n' "$db" | sed -E "$ren; 1s/^pub //")" >>"$prog"
  done
  local body; body="$(extract "$file" "$name")"
  [[ -n "$body" ]] || { echo "FAIL: fn $name not found in $file"; exit 1; }
  body="$(printf '%s\n' "$body" | sed -E "$ren; 1s/^pub //; 1s/\\) -> f64( with [A-Za-z, ]+)? \\{/) -> f64 with Mut, Div, Panic {/")"
  printf '%s\n\n' "$body" >>"$prog"
  cat >>"$prog" <<SIO
fn gate_sweep_${id}() -> i64 with IO, Mut, Div, Panic {
    var bad: i64 = 0
    var i: i64 = ${lo}
    while i < ${hi} {
        let x = gate_x(i)
        bad = bad + gate_bad(x, ${id}(x))
        i = i + 1
    }
    gate_report("${file} ${name}", bad, ${id}(0.0 - 13.855))
}

SIO
  printf '    fails = fails + gate_sweep_%s()\n' "$id" >>"$calls"
}

while read -r -a row; do
  [[ ${#row[@]} -gt 0 ]] || continue
  emit 0 "$GATE_N" "${row[@]}"
done <<<"$HELPERS"
read -r -a row <<<"$NEG_ONLY"
emit 0 "$NEG_START" "${row[@]}"
# grid indices whose |x| <= 20
C_LO=-1; C_HI=0; i=0
for x in $GRID; do
  ax="${x#-}"; ax="${ax%%.*}"
  if [[ "$ax" -le 20 ]]; then [[ "$C_LO" -lt 0 ]] && C_LO=$i; C_HI=$((i + 1)); fi
  i=$((i + 1))
done
while read -r -a row; do
  [[ ${#row[@]} -gt 0 ]] || continue
  emit "$C_LO" "$C_HI" "${row[@]}"
done <<<"$CLAMP20"

cat >>"$prog" <<'SIO'
fn main() -> i32 with IO, Mut, Div, Panic {
    var fails: i64 = 0
SIO
cat "$calls" >>"$prog"
cat >>"$prog" <<'SIO'
    if fails == 0 {
        print("STDLIB_EXP_HELPERS_OK\n")
        return 0
    }
    print("STDLIB_EXP_HELPERS_FAIL helpers=")
    print_int(fails)
    print("\n")
    return 1
}
SIO

echo "helpers: $n, grid points: $GATE_N ($NEG_START negative)"
if ! "$SOUC" compile "$prog" -o "$OUT/exp_helpers.elf" >"$OUT/compile.log" 2>&1; then
  tail -n 30 "$OUT/compile.log"
  echo "FAIL: generated harness did not compile (kept at ${KEEP_HARNESS:-<set KEEP_HARNESS=path to keep>})"
  [[ -n "${KEEP_HARNESS:-}" ]] && cp "$prog" "$KEEP_HARNESS"
  exit 1
fi
set +e
chmod +x "$OUT/exp_helpers.elf"
"$OUT/exp_helpers.elf" >"$OUT/run.log" 2>&1
rc=$?
set -e
grep -v '^$' "$OUT/run.log" | tail -n 80
if [[ $rc -ne 0 ]] || ! grep -qx 'STDLIB_EXP_HELPERS_OK' "$OUT/run.log"; then
  [[ -n "${KEEP_HARNESS:-}" ]] && cp "$prog" "$KEEP_HARNESS"
  echo "FAIL: rc=$rc"
  exit 1
fi
echo "PASS: helpers agree with math::pure::exp"

# --- Lint: no new hand-rolled exp series ------------------------------------
# A stdlib fn whose name contains "exp", of shape fn NAME(<p>: f64) -> f64, that
# runs a Taylor term recurrence `term = term * <id> / ...` is a hand-rolled exp.
# The ones that remain were swept against math::pure::exp on 2026-10-12 and are
# range-reduced (halving, ln2 reduction or reciprocal), so they do not cancel;
# they are pinned in stdlib_exp_series_baseline.txt. A new one must delegate to
# math::pure::exp instead. The baseline may only shrink: an entry that no longer
# matches must be removed in the same change.
BASELINE="$ROOT/scripts/ci/stdlib_exp_series_baseline.txt"
[[ -f "$BASELINE" ]] || { echo "FAIL: missing $BASELINE"; exit 1; }
found="$OUT/series_found.txt"
grep -rlE "term = term \* [A-Za-z_]+ / " stdlib --include=*.sio | LC_ALL=C sort | while read -r f; do
  awk -v F="$f" '
    /^(pub )?fn [A-Za-z0-9_]+\(/ { name=$0; sub(/^(pub )?fn /,"",name); sub(/\(.*/,"",name); sig=$0; body=""; inb=1 }
    inb { body = body "\n" $0 }
    inb && /^}/ {
      inb=0
      if (name ~ /[Ee]xp/ && sig ~ /^(pub )?fn [A-Za-z0-9_]+\([A-Za-z_][A-Za-z0-9_]*: f64\) -> f64/ && body ~ /term = term \* [A-Za-z_]+ \/ /) print F " " name
    }' "$f"
done | LC_ALL=C sort > "$found"
grep -vE '^[[:space:]]*(#|$)' "$BASELINE" | LC_ALL=C sort > "$OUT/series_base.txt"
new_series="$(comm -23 "$found" "$OUT/series_base.txt")"
gone="$(comm -13 "$found" "$OUT/series_base.txt")"
echo "exp series lint: $(wc -l < "$found") hand-rolled series found, $(wc -l < "$OUT/series_base.txt") pinned"
if [[ "$(wc -l < "$found")" -lt 50 ]]; then
  echo "FAIL: the series lint found fewer than 50 helpers; its pattern no longer matches the tree"; exit 1
fi
if [[ -n "$new_series" ]]; then
  echo "FAIL: new hand-rolled exp series (delegate to math::pure::exp instead, #846):"
  printf '  %s\n' $new_series | paste - - ; exit 1
fi
if [[ -n "$gone" ]]; then
  echo "FAIL: pinned series no longer present; remove them from $BASELINE:"
  printf '%s\n' "$gone"; exit 1
fi
echo "STDLIB_EXP_HELPERS_GATE_OK"
