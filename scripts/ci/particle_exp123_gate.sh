#!/usr/bin/env bash
# Gate for examples/particle_physics/exp123_z_metrology_nonunitary_ew.sio
#
# Three executable verticals on the particle_physics stdlib:
#   EXP1 — Z metrology (GUM Γ(Z→ee) + budget + confidence gate)
#   EXP2 — NonUnitary at Z pole (deficit vs √s curve + JSON receipt)
#   EXP3 — EW tension (M_W tree / Δρ / G_F-Δr pull ladder, S/T/U, Δρ)
#
# Prefer lean_single for full run (Madaros typecheck of particle_physics still
# has E008 residuals). Science-boundary allowlist is verified under Madaros
# advisory preflight: E-SRB-002 must be absent after science-rings carve-out.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
export SOUNIO_SOUC_ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}"

SRC=examples/particle_physics/exp123_z_metrology_nonunitary_ew.sio
OUT=/tmp/particle_exp123_out.txt
ERR=/tmp/particle_exp123_err.txt
JSON_OUT="${PARTICLE_EXP123_JSON:-$ROOT/examples/particle_physics/results/exp123_deficit_curve.json}"

echo "== particle exp123 run (engine=$SOUNIO_SOUC_ENGINE) =="
./bin/souc run "$SRC" >"$OUT" 2>"$ERR" || {
  echo "run failed; stderr:" >&2
  tail -40 "$ERR" >&2
  exit 1
}

grep -q 'PARTICLE_EXP123_OK' "$OUT"
grep -q 'PARTICLE_EXP123_PASS 62' "$OUT"
grep -q 'EXP1_DOMINANT_SOURCE' "$OUT"
grep -q 'EXP2_DEFICIT_POLE 1.000000' "$OUT"
grep -q 'EXP2_SCAN_DEFICIT_POLE' "$OUT"
grep -q 'EXP2_DEFICIT_JSON' "$OUT"
grep -q 'EXP3_MW_PULL_TREE' "$OUT"
grep -q 'EXP3_MW_PULL_RAD' "$OUT"
grep -q 'EXP3_MW_PULL_GF' "$OUT"
grep -q 'EXP3_MW_PRED_GF' "$OUT"
grep -q 'EXP3_TENSION_JSON' "$OUT"
grep -q 'EXP3_TENSION_CONSTRUCTION_GF' "$OUT"
grep -q 'EXP1_DONE' "$OUT"
grep -q 'EXP2_DONE' "$OUT"
grep -q 'EXP3_DONE' "$OUT"

# no unexpected FAIL lines
if grep -q '^FAIL ' "$OUT"; then
  echo "unexpected FAIL lines in output" >&2
  grep '^FAIL ' "$OUT" >&2
  exit 1
fi

# Materialise deficit-curve JSON receipt from the emitted line
mkdir -p "$(dirname "$JSON_OUT")"
python3 - "$OUT" "$JSON_OUT" <<'PY'
import json, re, sys
out_path, json_path = sys.argv[1], sys.argv[2]
text = open(out_path, encoding="utf-8", errors="replace").read()
m = re.search(r"^EXP2_DEFICIT_JSON\s+(\{.*\})\s*$", text, re.M)
if not m:
    raise SystemExit("EXP2_DEFICIT_JSON line missing")
payload = json.loads(m.group(1))
assert payload.get("schema") == "particle.exp123.deficit_curve.v1"
assert payload.get("particle") == "Z"
pts = payload.get("points")
assert isinstance(pts, list) and len(pts) >= 5
pole = next(p for p in pts if p.get("label") == "pole")
assert abs(float(pole["deficit"]) - 1.0) < 1e-6
# Monotone high-side: pole > mid > hi5 > thr
by = {p["label"]: float(p["deficit"]) for p in pts}
assert by["pole"] > by["mid"] > by["hi5"] > by["thr1pct"] > 0.0
with open(json_path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
    fh.write("\n")
print(f"EXP2_DEFICIT_JSON_WRITTEN {json_path}")
PY

# Madaros check path: science-boundary allowlist + typecheck.
# Full Madaros *run* still SEGV in imported lowering (compiler residual) —
# not claimed green here. Typecheck must be clean.
echo "== particle exp123 Madaros check (boundary + typecheck) =="
BOUND_ERR=/tmp/particle_exp123_boundary_err.txt
set +e
SOUNIO_SOUC_ENGINE=madaros ./bin/souc check "$SRC" >"$BOUND_ERR" 2>&1
MAD_RC=$?
set -e
if grep -q 'E-SRB-002' "$BOUND_ERR"; then
  echo "science-boundary allowlist failed: E-SRB-002 still present" >&2
  grep 'E-SRB-002' "$BOUND_ERR" >&2
  exit 1
fi
if ! grep -q 'check: OK' "$BOUND_ERR" && ! grep -q 'verdict=0' "$BOUND_ERR"; then
  # Prefer explicit OK; fall back to exit code + no error[
  if [[ "$MAD_RC" -ne 0 ]] || grep -q 'error\[' "$BOUND_ERR"; then
    echo "Madaros typecheck failed:" >&2
    tail -40 "$BOUND_ERR" >&2
    exit 1
  fi
fi
echo "PARTICLE_EXP123_BOUNDARY_OK (no E-SRB-002)"
echo "PARTICLE_EXP123_MADAROS_CHECK_OK"

# Madaros native *run* of the reduced core vertical (N4 partial)
# Full EXP123 still SEGVs at lower_array — see docs/handoff/particle_exp123_madaros_lower_array_segv_2026-07-25.md
echo "== particle exp123 Madaros core run =="
CORE=examples/particle_physics/exp123_madaros_core.sio
CORE_OUT=/tmp/particle_exp123_madaros_core_out.txt
CORE_ERR=/tmp/particle_exp123_madaros_core_err.txt
set +e
SOUNIO_SOUC_ENGINE=madaros ./bin/souc run "$CORE" >"$CORE_OUT" 2>"$CORE_ERR"
CORE_RC=$?
set -e
if ! grep -q 'PARTICLE_MADAROS_CORE_OK' "$CORE_OUT"; then
  echo "Madaros core run failed (rc=$CORE_RC):" >&2
  tail -40 "$CORE_ERR" >&2
  tail -20 "$CORE_OUT" >&2
  exit 1
fi
echo "PARTICLE_EXP123_MADAROS_RUN_OK (core)"

echo "PARTICLE_EXP123_GATE_OK"
