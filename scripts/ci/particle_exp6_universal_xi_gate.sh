#!/usr/bin/env bash
# Gate for examples/particle_physics/exp6_universal_deficit_xi.sio
#
# Universal reduced-variable NonUnitary deficit: d(ξ)=1/(1+ξ²) shared by Z,W,H,t.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
export SOUNIO_SOUC_ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}"

SRC=examples/particle_physics/exp6_universal_deficit_xi.sio
OUT=/tmp/particle_exp6_out.txt
ERR=/tmp/particle_exp6_err.txt
JSON_OUT="${PARTICLE_EXP6_JSON:-$ROOT/examples/particle_physics/results/exp6_cross_species_xi.json}"

echo "== particle exp6 universal xi (engine=$SOUNIO_SOUC_ENGINE) =="
./bin/souc run "$SRC" >"$OUT" 2>"$ERR" || {
  echo "run failed; stderr:" >&2
  tail -40 "$ERR" >&2
  exit 1
}

grep -q 'PARTICLE_EXP6_OK' "$OUT"
grep -q 'PARTICLE_EXP6_UNIVERSAL_XI_OK' "$OUT"
grep -q 'PARTICLE_EXP6_PASS 41' "$OUT"
grep -q 'EXP6_CLAIM NonUnitary_deficit_is_universal_under_reduced_xi' "$OUT"
grep -q 'EXP6_CROSS_JSON' "$OUT"
grep -q 'EXP6_SPECIES_JSON' "$OUT"
if grep -q '^FAIL ' "$OUT"; then
  echo "unexpected FAIL lines" >&2
  grep '^FAIL ' "$OUT" >&2
  exit 1
fi

mkdir -p "$(dirname "$JSON_OUT")"
python3 - "$OUT" "$JSON_OUT" <<'PY'
import json, re, sys
out_path, json_path = sys.argv[1], sys.argv[2]
text = open(out_path, encoding="utf-8", errors="replace").read()
m = re.search(r"^EXP6_CROSS_JSON\s+(\{.*\})\s*$", text, re.M)
if not m:
    raise SystemExit("EXP6_CROSS_JSON missing")
payload = json.loads(m.group(1))
assert payload.get("schema") == "particle.exp6.cross_species_xi.v1"
assert payload.get("claim") == "NonUnitary_deficit_universal_under_reduced_xi"
pts = payload.get("points")
assert isinstance(pts, list) and len(pts) >= 2
for p in pts:
    assert abs(float(p["residual"])) < 1e-9
    assert abs(float(p["d_z"]) - float(p["d_w"])) < 1e-9
with open(json_path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
    fh.write("\n")
print(f"EXP6_CROSS_JSON_WRITTEN {json_path}")
PY

# Madaros check + full run (proven green 2026-07-26)
echo "== particle exp6 Madaros check + full run =="
BOUND_ERR=/tmp/particle_exp6_madaros_err.txt
set +e
SOUNIO_SOUC_ENGINE=madaros ./bin/souc check "$SRC" >"$BOUND_ERR" 2>&1
MAD_RC=$?
set -e
if grep -q 'E-SRB-002' "$BOUND_ERR"; then
  echo "E-SRB-002 present" >&2
  exit 1
fi
if [[ "$MAD_RC" -ne 0 ]] && grep -q 'error\[' "$BOUND_ERR"; then
  tail -40 "$BOUND_ERR" >&2
  exit 1
fi
FULL_OUT=/tmp/particle_exp6_madaros_out.txt
FULL_ERR=/tmp/particle_exp6_madaros_full_err.txt
set +e
SOUNIO_SOUC_ENGINE=madaros ./bin/souc run "$SRC" >"$FULL_OUT" 2>"$FULL_ERR"
FULL_RC=$?
set -e
if ! grep -q 'PARTICLE_EXP6_OK' "$FULL_OUT"; then
  echo "Madaros full EXP6 failed rc=$FULL_RC" >&2
  tail -40 "$FULL_ERR" >&2
  tail -20 "$FULL_OUT" >&2
  exit 1
fi
echo "PARTICLE_EXP6_MADAROS_RUN_OK"
echo "PARTICLE_EXP6_GATE_OK"
