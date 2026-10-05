#!/usr/bin/env bash
# scripts/clinical_pathway_temporal_gate.sh
#
# Gate for the qualitative-temporal chemotherapy pathway demo
# (examples/clinical/pathway_temporal_demo.sio): regimen phase
# orderings and guideline constraints are stated as pointisable Allen
# interval relations, translated to endpoint point constraints by
# stdlib/ontology/temporal.sio, and closed by the verified EL+ engine
# of stdlib/ontology/elplus.sio (roleComp = path consistency over the
# point algebra; inconsistency surfaces as a strict self-loop
# before(c, c)).  The numeric sequencing lane
# (stdlib/clinical/mercyful.sio) validates the same pathway backbone
# and the two layers are asserted consistent.
#
# Engine: lean_single.  The demo imports stdlib/clinical/mercyful.sio,
# and any program importing another module segfaults at runtime on the
# current Madaros native lane (multi-module lowering bug, see
# docs/audit/MADAROS_MULTIMODULE_FALLBACK_SEGFAULT_2026-06-30.md); like
# scripts/clinical_ddi_elplus_gate.sh this gate therefore pins
# lean_single.  The demo still COMPILES under the default engine.
#
# Exit 0 = demo prints ALL PASS.  Exit 1 otherwise.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
export SOUNIO_SOUC_ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}"
SOUC="${SOUC:-$ROOT/bin/souc}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT

SRC="examples/clinical/pathway_temporal_demo.sio"
echo "== clinical_pathway_temporal_gate: engine=$SOUNIO_SOUC_ENGINE src=$SRC =="

if ! "$SOUC" compile "$SRC" -o "$OUT/pathway.elf" >"$OUT/pathwayc.log" 2>&1; then
  echo "FAIL: compile"; tail -30 "$OUT/pathwayc.log" || true
  exit 1
fi
chmod +x "$OUT/pathway.elf"
if ! "$OUT/pathway.elf" >"$OUT/pathway.log" 2>&1; then
  echo "FAIL: run"; cat "$OUT/pathway.log" || true
  exit 1
fi
if ! grep -qx "ALL PASS" "$OUT/pathway.log"; then
  echo "FAIL: missing exact ALL PASS line"; cat "$OUT/pathway.log" || true
  exit 1
fi

echo "CLINICAL_PATHWAY_TEMPORAL_GATE_OK"
exit 0
