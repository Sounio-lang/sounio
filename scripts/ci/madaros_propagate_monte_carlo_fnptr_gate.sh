#!/usr/bin/env bash
# Promotion gate: imported epistemic::propagate::monte_carlo with a named
# fn(f64)->f64 matches lean_single on mean/variance bands under default Madaros.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
SOUC="${SOUC:-$ROOT/bin/souc}"

# SIO-207: epistemic/propagate.sio calls wrapping_add/wrapping_mul, which
# postdate the committed Madaros prebuilt (E137 there). The Witness job
# places the fresh build at /tmp/madaros-ci.elf; prefer it when present so
# this gate measures the PR's compiler, like the neighbouring fresh-build
# steps. Locally the file is absent and the default resolution stands.
if [[ -z "${MADAROS_RAW_BIN:-}" && -x /tmp/madaros-ci.elf ]]; then
  export MADAROS_RAW_BIN=/tmp/madaros-ci.elf
  echo "monte_carlo_fnptr: using fresh Madaros at $MADAROS_RAW_BIN"
fi
PROBE="tests/epistemic_trust/madaros_propagate_monte_carlo_fnptr_probe.sio"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

echo "== madaros_propagate_monte_carlo_fnptr_gate =="

unset SOUNIO_SOUC_ENGINE || true
set +e
"$SOUC" run "$PROBE" >"$OUT/madaros.log" 2>&1
MRC=$?
set -e
[[ "$MRC" -eq 0 ]] || {
    echo "FAIL: Madaros rc=$MRC"
    cat "$OUT/madaros.log" || true
    exit 1
}
grep -Fq 'MONTE_CARLO_FNPTR PASS' "$OUT/madaros.log" || {
    echo "FAIL: Madaros missing MONTE_CARLO_FNPTR PASS"
    cat "$OUT/madaros.log" || true
    exit 1
}

# SIO-207: the lean_single oracle must be built from this tree (stage2):
# the committed lean seed predates wrapping_add/wrapping_mul and silently
# miscompiles propagate's LCG, so it cannot judge this probe. Same
# construction as the selfhost gate's stage2. Set SOUNIO_LEAN_STAGE2 to
# reuse a prebuilt stage2 ELF.
LEAN_STAGE2="${SOUNIO_LEAN_STAGE2:-$OUT/lean-stage2.elf}"
if [[ ! -x "$LEAN_STAGE2" ]]; then
  bash "$ROOT/scripts/ci/build_native_souc.sh" "$OUT/souc-host.elf" >"$OUT/lean-build.log" 2>&1 || {
    echo "FAIL: lean stage2 host build failed"; tail -20 "$OUT/lean-build.log" || true; exit 1; }
  chmod +x "$OUT/souc-host.elf"
  "$OUT/souc-host.elf" self-hosted/compiler/lean_single.sio "$LEAN_STAGE2" --target x86_64-linux >>"$OUT/lean-build.log" 2>&1 || {
    echo "FAIL: lean stage2 build failed"; tail -20 "$OUT/lean-build.log" || true; exit 1; }
  chmod +x "$LEAN_STAGE2"
fi
set +e
"$LEAN_STAGE2" "$PROBE" "$OUT/probe-lean.elf" --target x86_64-linux >"$OUT/lean.log" 2>&1
LRC=$?
if [[ "$LRC" -eq 0 ]]; then
  chmod +x "$OUT/probe-lean.elf"
  "$OUT/probe-lean.elf" >>"$OUT/lean.log" 2>&1
  LRC=$?
fi
set -e
[[ "$LRC" -eq 0 ]] || {
    echo "FAIL: lean_single oracle rc=$LRC"
    cat "$OUT/lean.log" || true
    exit 1
}
grep -Fq 'MONTE_CARLO_FNPTR PASS' "$OUT/lean.log" || {
    echo "FAIL: lean_single missing MONTE_CARLO_FNPTR PASS"
    cat "$OUT/lean.log" || true
    exit 1
}

echo "MADAROS_PROPAGATE_MONTE_CARLO_FNPTR_GATE_OK"
