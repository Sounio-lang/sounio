#!/usr/bin/env bash
# scripts/clinical_ddi_elplus_gate.sh
#
# Gate for the role-aware EL+ DDI screen
# (examples/clinical/ddi_elplus_demo.sio): ChEBI-grounded drug classes
# from stdlib/chemistry/ontology.sio are closed under the verified EL+
# engine (stdlib/ontology/elplus.sio) with a role hierarchy
# (inhibits sqsubseteq alters_activity_of) and composition chains
# (metabolized_by o part_of sqsubseteq metabolized_by), and the
# patient-level interaction pair is DERIVED via elplus_derive_conflicts.
#
# Engine: lean_single.  stdlib/chemistry/ontology.sio is a
# lean_single-lane module (its &str/&string workaround predicates do not
# type-check on the current Madaros lane); the elplus engine runs under
# both.  The demo therefore pins lean_single like
# scripts/clinical_midazolam_ddi_e2e_gate.sh does.
#
# Exit 0 = demo prints ALL PASS.  Exit 1 otherwise.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
export SOUNIO_SOUC_ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}"
SOUC="${SOUC:-$ROOT/bin/souc}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT

SRC="examples/clinical/ddi_elplus_demo.sio"
echo "== clinical_ddi_elplus_gate: engine=$SOUNIO_SOUC_ENGINE src=$SRC =="

if ! "$SOUC" compile "$SRC" -o "$OUT/ddi.elf" >"$OUT/ddic.log" 2>&1; then
  echo "FAIL: compile"; tail -30 "$OUT/ddic.log" || true
  exit 1
fi
chmod +x "$OUT/ddi.elf"
if ! "$OUT/ddi.elf" >"$OUT/ddi.log" 2>&1; then
  echo "FAIL: run"; cat "$OUT/ddi.log" || true
  exit 1
fi
if ! grep -qx "ALL PASS" "$OUT/ddi.log"; then
  echo "FAIL: missing exact ALL PASS line"; cat "$OUT/ddi.log" || true
  exit 1
fi

echo "CLINICAL_DDI_ELPLUS_GATE_OK"
exit 0
