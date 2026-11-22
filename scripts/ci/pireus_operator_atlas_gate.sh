#!/usr/bin/env bash
# eisa_h.pireus_operator_atlas.v1: the Sounio novelty classifier for Pireus
# kind=2 operators. The atlas is the frozen V13 class set (62) plus the frozen
# V1 corpus (3); canonical forms come from the unchanged V13 canonicalizer.
# Compiles the Sounio controls and the admission engine, rejects a stub, then
# runs the transport test against both (~4 min).
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9
. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "pireus_operator_atlas"

CONTROLS="tools/pireus/continuity/atlas/test_operator_atlas.sio"
GATE="tools/pireus/continuity/atlas/test_operator_atlas.py"
ADMISSION="tools/pireus/continuity/admission.sio"
require_file "stdlib/hardware/pireus/operator_atlas.sio"
require_file "$CONTROLS"
require_file "$GATE"
require_file "$ADMISSION"
require_tool python3

SOUC="${SOUC_BIN:-$ROOT_DIR/bin/souc}"
require_executable "$SOUC"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
for pair in "$CONTROLS:$WORK/atlas-test.elf" "$ADMISSION:$WORK/admission.elf"; do
  src="${pair%%:*}"; out="${pair#*:}"
  if ! timeout 600 "$SOUC" "$src" "$out" > "$WORK/compile.log" 2>&1; then
    tail -20 "$WORK/compile.log" >&2
    gate_fail "$src does not compile"
  fi
  require_elf "$out" "$src compiled to something that is not an ELF"
  chmod +x "$out"
done

# Control: a stub that claims PASS without the atlas table must be rejected.
STUB="$WORK/stub.sh"
printf '#!/usr/bin/env bash\necho "ATLAS_RESULT PASS"\n' > "$STUB"
chmod +x "$STUB"
if timeout 120 python3 "$GATE" "$STUB" "$WORK/admission.elf" > "$WORK/stub.log" 2>&1; then
  gate_fail "control failed: a stub atlas passed the transport test"
fi
echo "  control: a stub atlas is rejected"

gate_capture_rc "$WORK/rc" -- timeout 1800 python3 "$GATE" "$WORK/atlas-test.elf" "$WORK/admission.elf" > "$WORK/gate.log" 2>&1
RC="$(cat "$WORK/rc")"
if [[ "$RC" != "0" ]]; then
  tail -30 "$WORK/gate.log" >&2
  gate_fail "operator atlas controls failed (rc=$RC)"
fi
require_text "ATLAS_TABLE_MATCHES_FROZEN_V13 classes=62" "$WORK/gate.log"
require_text "PIREUS_OPERATOR_ATLAS_GATE_PASS" "$WORK/gate.log"
gate_pass "operator atlas reproduces 62 frozen V13 classes; invariance, negatives and receipt contract pass"
