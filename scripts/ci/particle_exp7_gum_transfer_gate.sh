#!/usr/bin/env bash
# Gate for examples/particle_physics/exp7_gum_xi_tension_transfer.sio
#
# EXP7: (A) GUM-propagated ξ / d(ξ) / thr  (B) EXP3↔EXP6 transfer receipt
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
export SOUNIO_SOUC_ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}"

SRC=examples/particle_physics/exp7_gum_xi_tension_transfer.sio
OUT=/tmp/particle_exp7_out.txt
ERR=/tmp/particle_exp7_err.txt
JSON_OUT="${PARTICLE_EXP7_JSON:-$ROOT/examples/particle_physics/results/exp7_transfer_tension_xi.json}"

echo "== particle exp7 gum+transfer (engine=$SOUNIO_SOUC_ENGINE) =="
./bin/souc run "$SRC" >"$OUT" 2>"$ERR" || {
  echo "run failed:" >&2
  tail -40 "$ERR" >&2
  exit 1
}

grep -q 'PARTICLE_EXP7_OK' "$OUT"
grep -q 'PARTICLE_EXP7_GUM_XI_OK' "$OUT"
grep -q 'PARTICLE_EXP7_TRANSFER_OK' "$OUT"
grep -q 'PARTICLE_EXP7_PASS 30' "$OUT"
grep -q 'EXP7_GUM_JSON' "$OUT"
grep -q 'EXP7_TRANSFER_JSON' "$OUT"
grep -q 'tension_ladder_improves_without_breaking_xi_universality' "$OUT"
if grep -q '^FAIL ' "$OUT"; then
  echo "unexpected FAIL" >&2
  grep '^FAIL ' "$OUT" >&2
  exit 1
fi

mkdir -p "$(dirname "$JSON_OUT")"
python3 - "$OUT" "$JSON_OUT" <<'PY'
import json, re, sys
out_path, json_path = sys.argv[1], sys.argv[2]
text = open(out_path, encoding="utf-8", errors="replace").read()
m = re.search(r"^EXP7_TRANSFER_JSON\s+(\{.*\})\s*$", text, re.M)
if not m:
    raise SystemExit("EXP7_TRANSFER_JSON missing")
payload = json.loads(m.group(1))
assert payload.get("schema") == "particle.exp7.transfer_tension_xi.v1"
assert "pull_tree" in payload and "pull_gf" in payload
assert abs(float(payload["pull_tree"])) > abs(float(payload["pull_gf"]))
res = payload.get("residual_zw")
assert isinstance(res, list) and all(abs(float(x)) < 1e-9 for x in res)
with open(json_path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
    fh.write("\n")
print(f"EXP7_TRANSFER_JSON_WRITTEN {json_path}")
PY

echo "== particle exp7 Madaros full run =="
FULL_OUT=/tmp/particle_exp7_madaros_out.txt
FULL_ERR=/tmp/particle_exp7_madaros_err.txt
set +e
SOUNIO_SOUC_ENGINE=madaros ./bin/souc run "$SRC" >"$FULL_OUT" 2>"$FULL_ERR"
FULL_RC=$?
set -e
if ! grep -q 'PARTICLE_EXP7_OK' "$FULL_OUT"; then
  echo "Madaros EXP7 failed rc=$FULL_RC" >&2
  tail -40 "$FULL_ERR" >&2
  tail -20 "$FULL_OUT" >&2
  exit 1
fi
echo "PARTICLE_EXP7_MADAROS_RUN_OK"
echo "PARTICLE_EXP7_GATE_OK"
