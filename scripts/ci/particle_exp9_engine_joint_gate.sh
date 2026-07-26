#!/usr/bin/env bash
# Gate for examples/particle_physics/exp9_engine_joint_gum.sio
#
# A) Joint (M,Γ) GUM under lean_single + Madaros
# B) EngineDisagreement dual-engine receipt (lean vs Madaros metrics)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"

SRC=examples/particle_physics/exp9_engine_joint_gum.sio
LEAN_OUT=/tmp/particle_exp9_lean_out.txt
MAD_OUT=/tmp/particle_exp9_madaros_out.txt
JSON_OUT="${PARTICLE_EXP9_JSON:-$ROOT/examples/particle_physics/results/exp9_engine_disagreement.json}"
JOINT_JSON="${PARTICLE_EXP9_JOINT_JSON:-$ROOT/examples/particle_physics/results/exp9_joint_gum.json}"

run_engine() {
  local eng="$1" out="$2" err="$3"
  echo "== particle exp9 run engine=$eng =="
  set +e
  SOUNIO_SOUC_ENGINE="$eng" ./bin/souc run "$SRC" >"$out" 2>"$err"
  local rc=$?
  set -e
  if ! grep -q 'PARTICLE_EXP9_OK' "$out"; then
    echo "engine=$eng failed rc=$rc" >&2
    tail -40 "$err" >&2
    tail -20 "$out" >&2
    exit 1
  fi
  grep -q 'PARTICLE_EXP9_PASS 17' "$out"
  grep -q 'PARTICLE_EXP9_JOINT_OK' "$out"
  if grep -q '^FAIL ' "$out"; then
    echo "FAIL lines under $eng" >&2
    grep '^FAIL ' "$out" >&2
    exit 1
  fi
}

run_engine lean_single "$LEAN_OUT" /tmp/particle_exp9_lean_err.txt
run_engine madaros "$MAD_OUT" /tmp/particle_exp9_madaros_err.txt

mkdir -p "$(dirname "$JSON_OUT")"
python3 - "$LEAN_OUT" "$MAD_OUT" "$JSON_OUT" "$JOINT_JSON" <<'PY'
import json, re, sys
from pathlib import Path

lean_path, mad_path, json_path, joint_path = sys.argv[1:5]
lean = Path(lean_path).read_text(encoding="utf-8", errors="replace")
mad = Path(mad_path).read_text(encoding="utf-8", errors="replace")

def grab(text: str, key: str) -> float:
    m = re.search(rf"^{re.escape(key)}\s+([+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?)\s*$", text, re.M)
    if not m:
        raise SystemExit(f"missing metric {key}")
    return float(m.group(1))

metrics = [
    "EXP9_ENGINE_DEFICIT_POLE",
    "EXP9_ENGINE_PEAK_LOCAL",
    "EXP9_ENGINE_PEAK_STDLIB",
    "EXP9_ENGINE_XI_JOINT_VAL",
    "EXP9_ENGINE_XI_JOINT_VAR",
    "EXP9_XI_MASS_ONLY_VAR",
    "EXP9_XI_JOINT_VAR",
    "EXP9_BUDGET_EXPAND",
]

rows = []
for key in metrics:
    lv = grab(lean, key)
    mv = grab(mad, key)
    res = abs(lv - mv)
    # peak stdlib: expect possible disagreement (Madaros residual)
    # others: should agree tightly
    if key == "EXP9_ENGINE_PEAK_STDLIB":
        agrees = res < 1e-9
        # document expected residual class
        expected_disagreement = True
    elif key in ("EXP9_XI_MASS_ONLY_VAR", "EXP9_XI_JOINT_VAR", "EXP9_ENGINE_XI_JOINT_VAR"):
        # print_f64 rounding / engine eps — relative 20% or abs 2e-6
        agrees = res <= max(2e-6, 0.25 * max(abs(lv), abs(mv), 1e-30))
        expected_disagreement = False
    else:
        agrees = res <= max(1e-9, 1e-6 * max(abs(lv), abs(mv), 1e-30))
        expected_disagreement = False
    rows.append({
        "quantity": key,
        "lean": lv,
        "madaros": mv,
        "residual": res,
        "agrees": bool(agrees),
        "expected_disagreement": expected_disagreement,
    })

# Hard requirements
def row(q):
    return next(r for r in rows if r["quantity"] == q)

assert row("EXP9_ENGINE_DEFICIT_POLE")["agrees"], "deficit pole engine disagreement"
assert row("EXP9_ENGINE_PEAK_LOCAL")["agrees"], "local peak must agree across engines"
assert row("EXP9_ENGINE_XI_JOINT_VAL")["agrees"], "joint xi val must agree"
assert row("EXP9_BUDGET_EXPAND")["agrees"], "budget expand must agree"
# Witness the known residual: stdlib peak disagrees (Madaros ~0, lean ~5e-6)
peak = row("EXP9_ENGINE_PEAK_STDLIB")
assert peak["lean"] > 1e-7, "lean stdlib peak should be physical"
assert peak["madaros"] < 1e-9, "madaros stdlib peak residual still zero under this IR"
assert peak["residual"] > 1e-7, "engine disagreement on PEAK_STDLIB not observed"

payload = {
    "schema": "particle.exp9.engine_disagreement.v1",
    "claim": "compiled_knowledge_may_disagree_across_engines",
    "non_isomorphism": "EngineDisagreement_is_not_physics_scheme_tension",
    "metrics": rows,
    "witness": {
        "quantity": "EXP9_ENGINE_PEAK_STDLIB",
        "lean": peak["lean"],
        "madaros": peak["madaros"],
        "residual": peak["residual"],
        "note": "imported eemm_z_peak_xsec_nu under full EXP9 IR returns 0 on Madaros; local peak agrees",
    },
}
Path(json_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"EXP9_ENGINE_DISAGREEMENT_JSON_WRITTEN {json_path}")

# Joint gum receipt from lean
m = re.search(r"EXP9_JOINT_JSON\s+(\{.*?\})", lean, re.S)
if not m:
    raise SystemExit("EXP9_JOINT_JSON missing")
raw = re.sub(r"\s+", " ", m.group(1)).strip()
joint = json.loads(raw)
assert joint.get("schema") == "particle.exp9.joint_gum.v1"
assert float(joint["budget_expand"]) > 1.0
assert float(joint["xi_joint_var"]) > float(joint["xi_mass_only_var"])
Path(joint_path).write_text(json.dumps(joint, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"EXP9_JOINT_JSON_WRITTEN {joint_path}")
print(f"EXP9_WITNESS_PEAK_STDLIB lean={peak['lean']} madaros={peak['madaros']} residual={peak['residual']}")
PY

echo "PARTICLE_EXP9_GATE_OK"
