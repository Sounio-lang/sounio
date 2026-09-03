#!/usr/bin/env bash
# Native CDCL + DIMACS vertical-slice gate.
# Runs only tiny SAT/UNSAT fixtures; scale and stress belong on Slurm.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SOUC="${SOUC:-$ROOT/bin/souc}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"

if [[ ! -x "$SOUC" ]]; then
  echo "cdcl_check_gate: missing executable: $SOUC" >&2
  exit 1
fi

# `stdlib/theorem/cdcl.sio` has no entry point, so it is typechecked transitively.
FILES=(
  "tests/stdlib/theorem/test_cdcl_core.sio"
)

PASS=0
FAIL=0
for f in "${FILES[@]}"; do
  if "$SOUC" check "$f" >/dev/null 2>&1; then
    echo "CHECK OK  $f"
    PASS=$((PASS + 1))
  else
    echo "CHECK FAIL $f" >&2
    "$SOUC" check "$f" 2>&1 | tail -20 >&2 || true
    FAIL=$((FAIL + 1))
  fi
done

echo "cdcl_check_gate: $PASS passed, $FAIL failed"
[[ "$FAIL" -eq 0 ]] || exit 1

python3 -m py_compile scripts/research/embed_dimacs_cdcl.py bin/sounio-sat

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT INT TERM

bin/sounio-sat tests/fixtures/solver/native_cdcl_sat.cnf \
  --out-dir "$WORK/sat" >"$WORK/sat.out" 2>"$WORK/sat.err"
python3 - "$WORK/sat/receipt.json" <<'PY'
import json, sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
assert receipt["result"] == "SAT"
assert receipt["model_verified"] is True
PY
test -s "$WORK/sat/model.txt"

bin/sounio-sat tests/fixtures/solver/native_cdcl_unsat.cnf \
  --out-dir "$WORK/unsat" >"$WORK/unsat.out" 2>"$WORK/unsat.err"
python3 - "$WORK/unsat/receipt.json" <<'PY'
import json, sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
assert receipt["result"] == "UNSAT"
assert receipt["native_rup_verified"] is True
assert receipt["host_rup_verified"] is True
assert receipt["proof_format"] == "DRUP_SUBSET_OF_DRAT"
assert receipt["lrat_sha256"]
PY
test -s "$WORK/unsat/proof.drat"
test -s "$WORK/unsat/proof.lrat"

bin/sounio-sat tests/fixtures/solver/native_cdcl_unsat_unannotated.cnf \
  --out-dir "$WORK/unsat-unannotated" >"$WORK/unsat-unannotated.out" 2>"$WORK/unsat-unannotated.err"
rg -q '"result": "UNSAT"' "$WORK/unsat-unannotated/receipt.json"
rg -q '"host_rup_verified": true' "$WORK/unsat-unannotated/receipt.json"

bin/sounio-sat tests/fixtures/solver/native_cdcl_sat_unused_var.cnf \
  --out-dir "$WORK/sat-unused" >"$WORK/sat-unused.out" 2>"$WORK/sat-unused.err"
rg -q '"model_verified": true' "$WORK/sat-unused/receipt.json"
rg -q 'v 1 -2 -3 0' "$WORK/sat-unused/model.txt"

if bin/sounio-sat tests/fixtures/solver/native_cdcl_bad_literal.cnf \
  --out-dir "$WORK/bad" >"$WORK/bad.out" 2>"$WORK/bad.err"; then
  echo "cdcl_check_gate: invalid DIMACS literal was accepted" >&2
  exit 1
fi
rg -q 'literal 2 exceeds declared variable count 1' "$WORK/bad.err"
test ! -e "$WORK/bad"

printf '2 0\n' >"$WORK/forged.drat"
if python3 examples/erdos/drup_to_lrat_rup.py \
  tests/fixtures/solver/native_cdcl_unsat.cnf "$WORK/forged.drat" "$WORK/forged.lrat" \
  >"$WORK/forged.out" 2>"$WORK/forged.err"; then
  echo "cdcl_check_gate: forged proof was accepted by host RUP checker" >&2
  exit 1
fi

python3 - "$ROOT" "$WORK" <<'PY'
import importlib.util
import pathlib
import runpy
import sys

root = pathlib.Path(sys.argv[1])
work = pathlib.Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("embed", root / "scripts/research/embed_dimacs_cdcl.py")
embed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(embed)

try:
    embed.generate_sio(
        1024, [], instance_id="too-wide", cnf_path=root / "tests/fixtures/solver/native_cdcl_sat.cnf",
        expected="", emit_verify=True,
    )
except ValueError as exc:
    assert "n_vars=1024" in str(exc)
else:
    raise AssertionError("1024-variable boundary was accepted")

try:
    embed.generate_sio(
        1, [(1,) * 1024 for _ in range(33)], instance_id="too-many-lits",
        cnf_path=root / "tests/fixtures/solver/native_cdcl_sat.cnf", expected="", emit_verify=True,
    )
except ValueError as exc:
    assert "input literals exceeds" in str(exc)
else:
    raise AssertionError("input literal reserve boundary was accepted")

evil = work / "evil\nfn injected() -> i32 { return 1 }.cnf"
evil.write_text("p cnf 1 1\n1 0\n", encoding="ascii")
generated = work / "evil.sio"
embed.embed(evil, generated, instance_id="safe-name", emit_unsat_proof=True)
text = generated.read_text(encoding="utf-8")
assert "fn injected" not in text
assert "generated_from_sha256" in text

oversize = work / "oversize.cnf"
with oversize.open("wb") as handle:
    handle.truncate(embed.CDCL_INPUT_MAX_BYTES + 1)
try:
    embed.parse_dimacs(oversize)
except ValueError as exc:
    assert "input exceeds" in str(exc)
else:
    raise AssertionError("oversized DIMACS input was accepted")

driver = runpy.run_path(str(root / "bin/sounio-sat"))
parse_native_output = driver["parse_native_output"]
complete_sat_model = driver["complete_sat_model"]
run_bounded = driver["run_bounded"]

for malformed in (
    "SOUNIO_CDCL_RESULT id 1\nSOUNIO_CDCL_RESULT id 1\n",
    "SOUNIO_CDCL_RESULT wrong 1\n",
    "SOUNIO_CDCL_UNKNOWN id 1\n",
):
    try:
        parse_native_output(malformed, "id")
    except RuntimeError:
        pass
    else:
        raise AssertionError(f"malformed native protocol was accepted: {malformed!r}")

try:
    complete_sat_model({1: 1, 4: -1}, 3)
except RuntimeError as exc:
    assert "out-of-range" in str(exc)
else:
    raise AssertionError("out-of-range native model variable was accepted")

try:
    run_bounded(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        timeout=1,
        memory_mb=128,
    )
except RuntimeError as exc:
    assert "timed out" in str(exc)
else:
    raise AssertionError("subprocess timeout was not normalized")
PY

echo "cdcl_check_gate: PASS (SAT replay + dual RUP/DRAT receipt + adversarial mutations)"
