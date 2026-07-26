#!/usr/bin/env bash
# Dual broken-structure receipt gate (N3)
#
# Validates EXP5 emits two sounio.broken_structure.v1 receipts
# (qft_unstable + sedenion_zd) with mandatory non-isomorphism disclaimer.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
export SOUNIO_SOUC_ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}"

SRC=examples/particle_physics/exp5_broken_structure_dual.sio
OUT=/tmp/particle_exp5_out.txt
ERR=/tmp/particle_exp5_err.txt

echo "== particle exp5 dual broken-structure (engine=$SOUNIO_SOUC_ENGINE) =="
./bin/souc run "$SRC" >"$OUT" 2>"$ERR" || {
  echo "run failed; stderr:" >&2
  tail -40 "$ERR" >&2
  exit 1
}

grep -q 'PARTICLE_EXP5_OK' "$OUT"
grep -q 'PARTICLE_EXP5_DUAL_RECEIPT_OK' "$OUT"
grep -q 'PARTICLE_EXP5_PASS 8' "$OUT"
grep -q 'EXP5_NON_ISOMORPHISM NonUnitary_deficit_is_not_sedenion_zero_divisor' "$OUT"
grep -q 'EXP5_ANALOGY_LEVEL receipt_geometry_and_typed_honesty_only' "$OUT"
grep -q 'EXP5_QFT_JSON' "$OUT"
grep -q 'EXP5_SED_JSON' "$OUT"
grep -q '"domain":"qft_unstable"' "$OUT"
grep -q '"domain":"sedenion_zd"' "$OUT"
grep -q 'sounio.broken_structure.v1' "$OUT"

if grep -q '^FAIL ' "$OUT"; then
  echo "unexpected FAIL lines" >&2
  grep '^FAIL ' "$OUT" >&2
  exit 1
fi

# Python schema validation of both JSON lines
python3 - "$OUT" <<'PY'
import json, re, sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
domains = {}
for key in ("EXP5_QFT_JSON", "EXP5_SED_JSON"):
    m = re.search(rf"^{key}\s+(\{{.*\}})\s*$", text, re.M)
    if not m:
        raise SystemExit(f"missing {key}")
    obj = json.loads(m.group(1))
    assert obj.get("schema") == "sounio.broken_structure.v1", obj
    assert "domain" in obj
    assert obj.get("non_isomorphism") == "NonUnitary_deficit_is_not_sedenion_zero_divisor"
    pts = obj.get("points")
    assert isinstance(pts, list) and len(pts) >= 2
    for p in pts:
        pr = float(p["proximity"])
        assert 0.0 <= pr <= 1.0 + 1e-9, p
    domains[obj["domain"]] = obj

assert "qft_unstable" in domains
assert "sedenion_zd" in domains

q = domains["qft_unstable"]
by = {p["label"]: float(p["proximity"]) for p in q["points"]}
assert abs(by["pole"] - 1.0) < 1e-2
assert by["pole"] > by["mid"] > by["hi5"] > by["thr1pct"] > 0.0

s = domains["sedenion_zd"]
sb = {p["label"]: float(p["proximity"]) for p in s["points"]}
assert sb["t1"] > 0.9
assert sb["t0"] < sb["t1"]
print("EXP5_SCHEMA_OK domains=" + ",".join(sorted(domains)))
PY

# Schema doc must exist
test -f docs/research/receipt_broken_structure.v1.md
grep -q 'Explicit non-isomorphism' docs/research/receipt_broken_structure.v1.md

echo "PARTICLE_EXP5_GATE_OK"
