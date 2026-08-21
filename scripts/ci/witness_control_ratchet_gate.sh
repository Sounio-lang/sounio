#!/usr/bin/env bash
# Witness control ratchet.
#
# A witness exists to fail when a rule stops working. If nothing can make it
# fall, it is not a witness -- it is a test that happens to pass, and it will
# keep passing after the rule it was written for has been deleted.
#
# Measured 2026-08-20, and the reason this gate exists: the witness gate's own
# positive control was hardcoded to SOUNIO_WIDE_MUL_SABOTAGE, which reaches
# nothing outside 256/512-bit arithmetic. Applied to a Mod witness it left the
# ELF byte-identical and reported CONTROL_FAIL -- blaming the witness for the
# sabotage being inapplicable. Witnesses now declare their own knob.
#
# Founder ruling 2026-08-20: freeze the count of listed witnesses with no
# `//@ sabotage:` knob, and fail the next one. No existing witness turns red;
# no new witness enters without a control. The frozen number may only SHRINK.
set -uo pipefail
cd "$(dirname "$0")/../.."

LIST=scripts/ci/witnesses.list
FROZEN_FILE=scripts/ci/witness_control.frozen
ART_DIR=artifacts/gates
mkdir -p "$ART_DIR"

frozen=$(sed -n 's/^total=//p' "$FROZEN_FILE" 2>/dev/null | head -1)
: "${frozen:=0}"

missing=0
listed=0
declared=0
while IFS= read -r pat; do
  case "$pat" in ''|'#'*) continue ;; esac
  for f in $pat; do
    [ -f "$f" ] || { echo "WITNESS_CONTROL_FAIL listed but absent: $f"; exit 1; }
    listed=$((listed + 1))
    if grep -qE '^//@[[:space:]]*sabotage:' "$f"; then
      declared=$((declared + 1))
      knob=$(sed -n 's|^//@[[:space:]]*sabotage:[[:space:]]*||p' "$f" | head -1)
      echo "WITNESS_CONTROL_OK $f knob=$knob"
    else
      missing=$((missing + 1))
      echo "WITNESS_CONTROL_ABSENT $f -- no //@ sabotage: knob"
    fi
  done
done < "$LIST"

st=pass
if [ "$missing" -gt "$frozen" ]; then
  echo "WITNESS_CONTROL_FAIL rose frozen=$frozen measured=$missing"
  echo "A new witness entered without a control. Declare '//@ sabotage: <KNOB>'"
  echo "naming the environment knob that must make it fall."
  st=fail
elif [ "$missing" -lt "$frozen" ]; then
  echo "WITNESS_CONTROL_OK fell frozen=$frozen measured=$missing -- lower the frozen total"
else
  echo "WITNESS_CONTROL_OK held=$missing of $listed listed ($declared declared)"
fi

cat > "$ART_DIR/witness_control_ratchet.json" <<JSON
{"status":"$st","metrics":{"total":$listed,"passed":$declared,"failed":$missing,"not_run":0}}
JSON
echo "status=$st"
echo "metrics {total=$listed, passed=$declared, failed=$missing, not_run=0}"
echo "artifact=$ART_DIR/witness_control_ratchet.json"
[ "$st" = pass ]
