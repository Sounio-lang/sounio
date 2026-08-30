#!/usr/bin/env bash
# scripts/clinical_pgx_elplus_gate.sh
#
# Gate for the role-aware EL+ PGx (pharmacogenomic) diplotype ->
# phenotype -> safety demo (examples/clinical/pgx_elplus_demo.sio):
# gene-drug loss-of-function flags are DERIVED from an EL+ TBox
# (allele hierarchy *4 sqsubseteq NullAllele sqsubseteq NoFunction,
# phenotype definitions with a conjunction filler, and ONE role chain
# metabolizedBy o hasVariant sqsubseteq contraindicated) closed per
# patient diplotype by the verified engine of
# stdlib/ontology/elplus.sio, and the boolean
# elplus_subsumes_dense(drug, exists contraindicated.NoFunction) gates
# the existing numeric CL-scaling lane
# (darwin_pbpk::pgx::cyp2d6_haloperidol).
#
# Engine: default (Madaros).  Unlike
# scripts/clinical_ddi_elplus_gate.sh this demo imports no
# lean_single-only module and passes under BOTH engines (verified
# 2026-08-05); the gate exercises the default lane.
#
# Exit 0 = demo prints ALL PASS.  Exit 1 otherwise.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
SOUC="${SOUC:-$ROOT/bin/souc}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT

SRC="examples/clinical/pgx_elplus_demo.sio"
echo "== clinical_pgx_elplus_gate: engine=${SOUNIO_SOUC_ENGINE:-default} src=$SRC =="

if ! "$SOUC" compile "$SRC" -o "$OUT/pgx.elf" >"$OUT/pgxc.log" 2>&1; then
  echo "FAIL: compile"; tail -30 "$OUT/pgxc.log" || true
  exit 1
fi
chmod +x "$OUT/pgx.elf"
if ! "$OUT/pgx.elf" >"$OUT/pgx.log" 2>&1; then
  echo "FAIL: run"; cat "$OUT/pgx.log" || true
  exit 1
fi
if ! grep -qx "ALL PASS" "$OUT/pgx.log"; then
  echo "FAIL: missing exact ALL PASS line"; cat "$OUT/pgx.log" || true
  exit 1
fi

echo "CLINICAL_PGX_ELPLUS_GATE_OK"
exit 0
