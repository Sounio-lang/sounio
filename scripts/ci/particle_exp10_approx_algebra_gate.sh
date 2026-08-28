#!/usr/bin/env bash
# Gate for examples/particle_physics/exp10_approx_effect_algebra.sio
#
# lean_single: full algebra + effect core + physics (30/30)
# Madaros: algebra + effect core (27/30); physics imports residual (PARTIAL_OK)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"

SRC=examples/particle_physics/exp10_approx_effect_algebra.sio
JSON_OUT="${PARTICLE_EXP10_JSON:-$ROOT/examples/particle_physics/results/exp10_approx_algebra.json}"

echo "== particle exp10 lean_single full =="
LEAN_OUT=/tmp/particle_exp10_lean_out.txt
LEAN_ERR=/tmp/particle_exp10_lean_err.txt
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run "$SRC" >"$LEAN_OUT" 2>"$LEAN_ERR" || {
  echo "lean run failed" >&2
  tail -40 "$LEAN_ERR" >&2
  exit 1
}
grep -q 'PARTICLE_EXP10_OK' "$LEAN_OUT"
grep -q 'PARTICLE_EXP10_PASS 30' "$LEAN_OUT"
grep -q 'PARTICLE_EXP10_ALGEBRA_OK' "$LEAN_OUT"
grep -q 'PARTICLE_EXP10_EFFECT_CORE_OK' "$LEAN_OUT"
grep -q 'PARTICLE_EXP10_EFFECT_PHYSICS_OK' "$LEAN_OUT"
grep -q 'L4_nu_nwa_tension' "$LEAN_OUT"
if grep -q '^FAIL ' "$LEAN_OUT"; then
  echo "unexpected FAIL under lean" >&2
  grep '^FAIL ' "$LEAN_OUT" >&2
  exit 1
fi

mkdir -p "$(dirname "$JSON_OUT")"
python3 - "$LEAN_OUT" "$JSON_OUT" <<'PY'
import json, re, sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
m = re.search(r"EXP10_ALGEBRA_JSON\s+(\{.*?\})", text, re.S)
if not m:
    raise SystemExit("EXP10_ALGEBRA_JSON missing")
raw = re.sub(r"\s+", " ", m.group(1)).strip()
payload = json.loads(raw)
assert payload.get("schema") == "particle.exp10.approx_algebra.v1"
assert int(payload.get("triple_tension", 0)) == 1
assert float(payload["nwa_h"]) < float(payload["nwa_z"])
assert float(payload["triple_combined"]) > 1.0
open(sys.argv[2], "w", encoding="utf-8").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(f"EXP10_ALGEBRA_JSON_WRITTEN {sys.argv[2]}")
PY

echo "== particle exp10 Madaros algebra+core =="
MAD_OUT=/tmp/particle_exp10_madaros_out.txt
MAD_ERR=/tmp/particle_exp10_madaros_err.txt
set +e
SOUNIO_SOUC_ENGINE=madaros ./bin/souc run "$SRC" >"$MAD_OUT" 2>"$MAD_ERR"
MAD_RC=$?
set -e
if ! grep -q 'PARTICLE_EXP10_ALGEBRA_OK' "$MAD_OUT"; then
  echo "Madaros algebra failed rc=$MAD_RC" >&2
  tail -40 "$MAD_ERR" >&2
  tail -30 "$MAD_OUT" >&2
  exit 1
fi
if ! grep -q 'PARTICLE_EXP10_EFFECT_CORE_OK' "$MAD_OUT"; then
  echo "Madaros effect core failed" >&2
  tail -30 "$MAD_OUT" >&2
  exit 1
fi
# Full OK or PARTIAL_OK (physics import residual)
if ! grep -qE 'PARTICLE_EXP10_OK|PARTICLE_EXP10_PARTIAL_OK' "$MAD_OUT"; then
  echo "Madaros missing OK/PARTIAL_OK" >&2
  exit 1
fi
echo "PARTICLE_EXP10_MADAROS_CORE_OK"

# Residual #1 closeout surface: free-fn approx_effects physics (thin IR)
echo "== particle exp10 Madaros physics thin vertical =="
PHYS=examples/particle_physics/exp10_approx_physics.sio
PHYS_OUT=/tmp/particle_exp10_physics_out.txt
PHYS_ERR=/tmp/particle_exp10_physics_err.txt
set +e
SOUNIO_SOUC_ENGINE=madaros ./bin/souc run "$PHYS" >"$PHYS_OUT" 2>"$PHYS_ERR"
PHYS_RC=$?
set -e
if ! grep -q 'PARTICLE_EXP10_PHYSICS_OK' "$PHYS_OUT"; then
  echo "Madaros physics thin failed rc=$PHYS_RC" >&2
  tail -40 "$PHYS_ERR" >&2
  tail -20 "$PHYS_OUT" >&2
  exit 1
fi
echo "PARTICLE_EXP10_MADAROS_PHYSICS_OK"
echo "PARTICLE_EXP10_GATE_OK"
