#!/usr/bin/env bash
# The PIREUS admission engine is compiled and asked, not merely hashed.
#
# tools/pireus/continuity/admission.sio is the non-delegable semantic authority
# of the PIREUS cycle: the LLM proposes, this executable decides. Measured
# 2026-09-11, nothing in this repository compiled it. cycle.py records
# `admission_source_sha256` -- a hash of the source text -- and no workflow,
# Makefile target or script ever built it or ran
# tools/pireus/continuity/test_admission.py, which takes the engine as its
# only argument and had therefore never been run against one here.
#
# What that cost: #2476 taught the engine kind=2 (operator proposals, phase
# key, 130623 mask). A v1-shaped document carrying kind=2 stopped refusing
# with KIND and started refusing with KIND_MASK -- both correct, but
# test_admission.py still asserted KIND. The gate was red from the moment the
# patch merged and nothing could notice, because nothing ran it.
#
# Compiling costs three seconds and the gate under a second.
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9
. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "pireus_admission_engine"

SOURCE="tools/pireus/continuity/admission.sio"
GATE="tools/pireus/continuity/test_admission.py"
require_file "$SOURCE"
require_file "$GATE"
require_tool python3

SOUC="${SOUC_BIN:-$ROOT_DIR/bin/souc}"
require_executable "$SOUC"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
ENGINE="$WORK/admission.elf"

echo "  compiling $SOURCE with $SOUC"
if ! timeout 600 "$SOUC" "$SOURCE" "$ENGINE" > "$WORK/compile.log" 2>&1; then
  tail -20 "$WORK/compile.log" >&2
  gate_fail "the admission engine does not compile -- PIREUS has no authority to run"
fi
require_elf "$ENGINE" "admission.sio compiled to something that is not an ELF"
chmod +x "$ENGINE"

# Control. test_admission.py must be reading the engine's answers, not printing
# a marker on its own. A stub that admits everything has to make it fail; if it
# does not, a green run below says nothing about the real engine.
STUB="$WORK/stub.sh"
cat > "$STUB" <<'STUBEOF'
#!/usr/bin/env bash
echo '{"schema":1,"authority":"Sounio","decision":"ADMIT","kind":"lowering","plan_id":0,"tensor_components":4096,"tensor_encoding":"cd16-abk-offset1-v1","tensor_sha256":"0","context_sha256":"0","proposal_sha256":"0","fp_parity":"UNMEASURED","claim_ready":false,"formal_v13_v14":"OPEN"}'
STUBEOF
chmod +x "$STUB"
if timeout 120 python3 "$GATE" "$STUB" > "$WORK/stub.log" 2>&1; then
  gate_fail "control failed: an engine that admits everything passed the gate"
fi
echo "  control: an always-admit stub is rejected"

gate_capture_rc "$WORK/rc" -- timeout 300 python3 "$GATE" "$ENGINE" > "$WORK/gate.log" 2>&1
RC="$(cat "$WORK/rc")"
if [[ "$RC" != "0" ]]; then
  tail -20 "$WORK/gate.log" >&2
  gate_fail "the admission engine failed its own adversarial gate (rc=$RC)"
fi
require_text "PIREUS_EXTERNAL_ADMISSION_GATE_PASS" "$WORK/gate.log"

# The sedenion structure tensor under cd16-abk-offset1-v1 is a fixed
# mathematical object: 4096 coefficients, one byte each, from cd_sigma alone.
# Both the kind=1 lowering receipt and the kind=2 phase-0 operator receipt must
# carry this digest. A change here is a change to the algebra, never a refactor.
CD_TENSOR="6afd9bdb6593d7283fb6cc01bc7d2c58d35e6de843c011cdca3d07233c63be29"
require_text "$CD_TENSOR" "$WORK/gate.log"
OPERATORS="$(grep -c '"kind": "operator"' "$WORK/gate.log")"
require_min_count "$OPERATORS" 9 "admitted operator receipts"

gate_pass "admission engine compiled and answered $(grep -c '^' "$WORK/gate.log") probes, $OPERATORS of them operators"
