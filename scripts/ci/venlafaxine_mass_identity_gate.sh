#!/usr/bin/env bash
# scripts/ci/venlafaxine_mass_identity_gate.sh
#
# Two-sided mass identity for the venlafaxine XR scenario, on both engines
# that implement it: the stdlib scenario (via the PBPK28 parity reference) and
# the website's Node engine (#2718).
#
# The PBPK28 parity gate's conservation check is one-sided (parent + ODV <=
# released), so a formation step that hands ODV more than the parent lost
# passes whenever the total stays below the release. That is how the #2718
# fabrication (79-99.7 % of every transfer created) stayed green. This gate
# checks the per-step identity both ways, for the steady-state interval at
# dt = 0.5 h and 0.25 h, from the residuals each engine reports.
#
# Standalone it produces both logs itself (Sounio ref on lean_single, which
# runs it in seconds; the parity gate runs it on Madaros and calls this
# script with its own logs via VFX_SIO_LOG / VFX_NODE_LOG / OUT_DIR).
#
# GATE_CONTRACT: v0
# GATE_ID: venlafaxine_mass_identity
# GATE_CLAIMS: the venlafaxine parent and ODV per-step mass identities hold
#              to 1e-9 mg on the Sounio scenario and the Node engine, and a
#              log with created or destroyed mass is rejected
# GATE_ENGINE: lean_single (standalone), node
# GATE_RESULT_ON_SKIP: fail
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
command -v node >/dev/null 2>&1 || { echo "VENLAFAXINE_MASS_IDENTITY_FAIL: node absent" >&2; exit 1; }

if [[ -z "${OUT_DIR:-}" ]]; then
  OUT_DIR="$(mktemp -d)"; trap 'rm -rf "$OUT_DIR"' EXIT
fi
if [[ -z "${VFX_SIO_LOG:-}" ]]; then
  export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
  ulimit -s unlimited 2>/dev/null || true
  LEAN="${SOUNIO_LEAN_SINGLE_BIN:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
  VFX_SIO_LOG="$OUT_DIR/vfx_sounio.txt"
  timeout 600 "$LEAN" tests/run-pass/dissertation_pbpk28_parity_ref_venlafaxine.sio "$OUT_DIR/vfx_ref.elf" \
    > "$OUT_DIR/vfx_ref_compile.txt" 2>&1 || true
  if [[ ! -s "$OUT_DIR/vfx_ref.elf" ]] || ! grep -q '^compile: fns=' "$OUT_DIR/vfx_ref_compile.txt"; then
    echo "VENLAFAXINE_MASS_IDENTITY_FAIL: the Sounio reference did not compile" >&2
    tail -n 5 "$OUT_DIR/vfx_ref_compile.txt" >&2; exit 1
  fi
  chmod +x "$OUT_DIR/vfx_ref.elf"
  timeout 600 "$OUT_DIR/vfx_ref.elf" > "$VFX_SIO_LOG" 2>&1 || {
    echo "VENLAFAXINE_MASS_IDENTITY_FAIL: the Sounio reference returned non-zero" >&2; tail -n 5 "$VFX_SIO_LOG" >&2; exit 1; }
fi
if [[ -z "${VFX_NODE_LOG:-}" ]]; then
  VFX_NODE_LOG="$OUT_DIR/vfx_node.txt"
  node scripts/dissertation/run_venlafaxine_node.mjs > "$VFX_NODE_LOG" 2>&1 || {
    echo "VENLAFAXINE_MASS_IDENTITY_FAIL: the Node runner returned non-zero" >&2; exit 1; }
fi

# Two-sided mass identity (#2718). The check above is one-sided: it fails only
# when parent + ODV exceeds the released mass, so a formation step that hands
# ODV more than the parent lost passed whenever the total stayed below the
# release (the fabrication fixed in 9b8cf1de3 did exactly that). Each engine now
# reports, for the steady-state interval at each dt, the summed |per-step mass
# identity residual| of parent and ODV (x1e12 mg):
#   parent: m1 - (m0 + absorbed - CL*AUC_step - formed)
#   ODV:    m1 - (m0 + formed   - CL*AUC_step)
# Mass created or destroyed in either direction makes it grow. Measured
# 2026-10-12: <= 1.06e-12 mg on both engines; the bound is 1e-9 mg.
VFX_RESID_MAX_E12="${VFX_RESID_MAX_E12:-1000}"
vfx_mass_identity() {  # $1=log $2=label
  awk -F= -v MAX="$VFX_RESID_MAX_E12" -v L="$2" '
    /^VSS\|dt=/ { dt=$2; next }
    /^VSS\|residual_mg_parent_e12=/ { rp[dt]=$2; seen[dt]=1; next }
    /^VSS\|residual_mg_odv_e12=/    { ro[dt]=$2; next }
    END {
      n=0; bad=0
      for (d in seen) {
        n++
        if (!(d in ro) || rp[d]=="" || ro[d]=="") { bad++; printf "  FAIL: %s dt=%s residual record incomplete\n", L, d; continue }
        if (!(rp[d]+0 <= MAX+0)) { bad++; printf "  FAIL: %s dt=%s parent mass identity off by %s e-12 mg (bound %s)\n", L, d, rp[d], MAX }
        if (!(ro[d]+0 <= MAX+0)) { bad++; printf "  FAIL: %s dt=%s ODV mass identity off by %s e-12 mg (bound %s)\n", L, d, ro[d], MAX }
      }
      if (n != 2) { printf "  FAIL: %s expected 2 dt rows with residuals, got %d\n", L, n; exit 1 }
      if (bad > 0) exit 1
      printf "  %s: parent and ODV mass identities within %s e-12 mg at both dt\n", L, MAX
    }' "$1"
}
echo "[pbpk28-parity] venlafaxine two-sided mass identity (per-step residual, both engines)"
# Self-check first: the 0.433479 mg of ODV per 0.5 h step that the pre-fix
# formula created (#2718, PM) must be caught, and so must mass destroyed.
printf 'VSS|dt=0.500000\nVSS|residual_mg_parent_e12=433479000000.0\nVSS|residual_mg_odv_e12=0.5\nVSS|dt=0.250000\nVSS|residual_mg_parent_e12=0.8\nVSS|residual_mg_odv_e12=0.8\n' > "$OUT_DIR/vfx_mass_sab_excess.txt"
printf 'VSS|dt=0.500000\nVSS|residual_mg_parent_e12=1.0\nVSS|residual_mg_odv_e12=55357000000.0\nVSS|dt=0.250000\nVSS|residual_mg_parent_e12=0.8\nVSS|residual_mg_odv_e12=0.8\n' > "$OUT_DIR/vfx_mass_sab_loss.txt"
for sab in excess loss; do
  if vfx_mass_identity "$OUT_DIR/vfx_mass_sab_$sab.txt" "self-check:$sab" > "$OUT_DIR/vfx_mass_sab_$sab.out" 2>&1; then
    echo "[pbpk28-parity] FAIL: the two-sided mass identity accepted the sabotaged '$sab' log" >&2; exit 1; fi
  echo "  self-check ok: '$sab' log rejected ($(grep -m1 FAIL "$OUT_DIR/vfx_mass_sab_$sab.out" | sed 's/^ *//'))"
done
vfx_mass_identity "$VFX_SIO_LOG" sounio || { echo "VENLAFAXINE_MASS_IDENTITY_FAIL" >&2; exit 1; }
vfx_mass_identity "$VFX_NODE_LOG" node || { echo "VENLAFAXINE_MASS_IDENTITY_FAIL" >&2; exit 1; }
echo "VENLAFAXINE_MASS_IDENTITY_PASS"
