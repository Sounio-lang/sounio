#!/usr/bin/env bash
# Gate for examples/particle_physics/exp8_deficit_collapse_failure.sio
#
# EXP8: DeficitCollapse failure surface + SchemeTension
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
export SOUNIO_SOUC_ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}"

SRC=examples/particle_physics/exp8_deficit_collapse_failure.sio
OUT=/tmp/particle_exp8_out.txt
ERR=/tmp/particle_exp8_err.txt
JSON_OUT="${PARTICLE_EXP8_JSON:-$ROOT/examples/particle_physics/results/exp8_deficit_collapse.json}"

echo "== particle exp8 collapse (engine=$SOUNIO_SOUC_ENGINE) =="
./bin/souc run "$SRC" >"$OUT" 2>"$ERR" || {
  echo "run failed:" >&2
  tail -40 "$ERR" >&2
  exit 1
}

grep -q 'PARTICLE_EXP8_OK' "$OUT"
grep -q 'PARTICLE_EXP8_COLLAPSE_OK' "$OUT"
grep -q 'PARTICLE_EXP8_SCHEME_OK' "$OUT"
grep -q 'PARTICLE_EXP8_PASS 20' "$OUT"
grep -q 'EXP8_COLLAPSE_JSON' "$OUT"
grep -q 'EXP8_SCHEME_JSON' "$OUT"
grep -q 'pure_BW_xi_collapse_has_failure_surface' "$OUT"
grep -q 'honest_schemes_can_disagree_beyond_2sigma' "$OUT"
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
# print_int may insert newlines inside the JSON object — flatten from marker to closing brace
m = re.search(r"EXP8_COLLAPSE_JSON\s+(\{.*?\})", text, re.S)
if not m:
    raise SystemExit("EXP8_COLLAPSE_JSON missing")
raw = re.sub(r"\s+", " ", m.group(1)).strip()
payload = json.loads(raw)
assert payload.get("schema") == "particle.exp8.deficit_collapse.v1"
assert int(payload["running_fail_status"]) == 1
assert int(payload["interference_fail_status"]) == 2
assert float(payload["running_fail_xi2_residual"]) > 1e-3
assert float(payload["interference_fail_xi1_residual"]) > 1e-3
assert int(payload["interference_alpha0_status"]) == 0
with open(json_path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
    fh.write("\n")
print(f"EXP8_COLLAPSE_JSON_WRITTEN {json_path}")
PY

echo "== particle exp8 Madaros full run =="
FULL_OUT=/tmp/particle_exp8_madaros_out.txt
FULL_ERR=/tmp/particle_exp8_madaros_err.txt
set +e
SOUNIO_SOUC_ENGINE=madaros ./bin/souc run "$SRC" >"$FULL_OUT" 2>"$FULL_ERR"
FULL_RC=$?
set -e
if ! grep -q 'PARTICLE_EXP8_OK' "$FULL_OUT"; then
  echo "Madaros EXP8 failed rc=$FULL_RC" >&2
  tail -40 "$FULL_ERR" >&2
  tail -30 "$FULL_OUT" >&2
  exit 1
fi
echo "PARTICLE_EXP8_MADAROS_RUN_OK"
echo "PARTICLE_EXP8_GATE_OK"
