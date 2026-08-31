#!/usr/bin/env bash
# Gate for examples/particle_physics/exp11_scheme_approx_product.sio
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"

SRC=examples/particle_physics/exp11_scheme_approx_product.sio
JSON_OUT="${PARTICLE_EXP11_JSON:-$ROOT/examples/particle_physics/results/exp11_scheme_approx_product.json}"

run_eng() {
  local eng="$1" out="$2" err="$3"
  echo "== particle exp11 engine=$eng =="
  set +e
  SOUNIO_SOUC_ENGINE="$eng" ./bin/souc run "$SRC" >"$out" 2>"$err"
  local rc=$?
  set -e
  if ! grep -q 'PARTICLE_EXP11_OK' "$out"; then
    echo "engine=$eng failed rc=$rc" >&2
    tail -40 "$err" >&2
    tail -20 "$out" >&2
    exit 1
  fi
  grep -q 'PARTICLE_EXP11_PASS 16' "$out"
  grep -q 'PARTICLE_EXP11_PRODUCT_OK' "$out"
  if grep -q '^FAIL ' "$out"; then
    echo "FAIL under $eng" >&2
    grep '^FAIL ' "$out" >&2
    exit 1
  fi
}

run_eng lean_single /tmp/particle_exp11_lean_out.txt /tmp/particle_exp11_lean_err.txt
run_eng madaros /tmp/particle_exp11_mad_out.txt /tmp/particle_exp11_mad_err.txt

mkdir -p "$(dirname "$JSON_OUT")"
python3 - /tmp/particle_exp11_lean_out.txt "$JSON_OUT" <<'PY'
import json, re, sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
m = re.search(r"EXP11_PRODUCT_JSON\s+(\{.*?\})", text, re.S)
if not m:
    raise SystemExit("EXP11_PRODUCT_JSON missing")
raw = re.sub(r"\s+", " ", m.group(1)).strip()
payload = json.loads(raw)
assert payload.get("schema") == "particle.exp11.scheme_approx_product.v1"
assert int(payload["nfail_cells"]) == 4
assert int(payload["l4_on_fixed_tension"]) == 1
assert int(payload["l4_on_fixed_fails"]) == 0
assert float(payload["running_triple_cell_res"]) > float(payload["fixed_empty_cell_res"])
assert int(payload["interf_alpha0_fails"]) == 0
assert int(payload["interf_alpha_fails"]) == 1
open(sys.argv[2], "w", encoding="utf-8").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(f"EXP11_PRODUCT_JSON_WRITTEN {sys.argv[2]}")
PY

echo "PARTICLE_EXP11_GATE_OK"
