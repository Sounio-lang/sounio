#!/usr/bin/env bash
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/gate_artifact.sh"
# RelocationTable capacity coherence.
#
# self-hosted/native/reloc.sio declares `entries: [Relocation; N]` and guards
# every add_* with `count < reloc_table_capacity()`. Those two numbers are
# written in different places and nothing in the language ties them together.
#
# They had already drifted. The array was declared at 4096 and all four guards
# tested 256, so from the 257th relocation the entry was DROPPED and the
# function returned normally -- no flag, no error, nothing downstream. An
# unpatched rip-relative site keeps the displacement it was emitted with, which
# in this backend is zero, so the program reads the start of .rodata instead of
# its constant. It assembles, links and runs.
#
# Same family as float_slot_capacity_coherence_gate.sh,
# irfunction_instr_capacity_coherence_gate.sh and
# mir_instr_capacity_coherence_gate.sh: a declaration and its bound check,
# compared against each other rather than against a number in this script.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="${RELOC_CAPACITY_SOURCE:-$ROOT/self-hosted/native/reloc.sio}"
ART="$ROOT/artifacts/gates/reloc_table_capacity_coherence.v1.json"
mkdir -p "$(dirname "$ART")"

fail() {
  printf 'RELOC_CAPACITY_FAIL reason=%s\n' "$1" >&2
  printf '{"status":"fail","reason":"%s","metrics":{"total":0,"passed":0,"failed":1,"not_run":0}}\n' "$1" | gate_write_artifact "$ART"
  exit 1
}
[[ -f "$SOURCE" ]] || fail source_missing
command -v python3 >/dev/null 2>&1 || fail python3_missing

check() { python3 "$ROOT/scripts/ci/lib/reloc_capacity_coherence.py" "$1"; }

# Positive control FIRST. A checker that has never failed has measured nothing,
# and the defect this gate exists for is precisely a guard that disagrees with
# its declaration -- so the control reintroduces exactly that.
SAB=$(mktemp); trap 'rm -f "$SAB"' EXIT
sed 's/if t\.count < reloc_table_capacity() {/if t.count < 256 {/' "$SOURCE" | gate_write_artifact "$SAB"
if check "$SAB" >/dev/null 2>&1; then
  echo "CONTROL_FAIL: the sabotaged source passed. This gate inspects nothing." >&2
  printf '{"status":"fail","reason":"positive control did not fire","metrics":{"total":0,"passed":0,"failed":1,"not_run":0}}\n' | gate_write_artifact "$ART"
  exit 1
fi
echo "control: a literal 256 guard is rejected, as required"

# The flag is worthless if nothing reads it. Checked in the emitter, not here,
# because that is where the refusal has to live -- the same place code_overflow
# and reloc_overflow are already refused on.
CONSUMER="$ROOT/self-hosted/native/codegen_x86_linux.sio"
if ! grep -q 'nc\.relocs\.overflow' "$CONSUMER"; then
  echo "RELOC_CAPACITY_FAIL reason=nothing_refuses_on_overflow" >&2
  echo "  reloc.sio records a dropped relocation but $CONSUMER never checks it," >&2
  echo "  so the binary ships with an unpatched site instead of being refused." >&2
  printf '{"status":"fail","reason":"nothing refuses on overflow","metrics":{"total":0,"passed":0,"failed":1,"not_run":0}}\n' | gate_write_artifact "$ART"
  exit 1
fi
echo "control: the emitter refuses on a dropped relocation"

if out=$(check "$SOURCE" 2>&1); then
  echo "$out"
  printf '{"status":"pass","metrics":{"total":1,"passed":1,"failed":0,"not_run":0}}\n' | gate_write_artifact "$ART"
  exit 0
fi
echo "$out" >&2
printf '{"status":"fail","metrics":{"total":1,"passed":0,"failed":1,"not_run":0}}\n' | gate_write_artifact "$ART"
exit 1
