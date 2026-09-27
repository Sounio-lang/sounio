#!/usr/bin/env bash
# native_reloc_shape_gate.sh -- a recorded relocation is PATCHED or REFUSED,
# never left at its placeholder 0.
#
# History (2026-09-26): apply_relocations_into patched a kind 2/3/4 RIP-relative
# relocation only when native_v2_reloc_is_rip_disp_patch recognised the bytes
# before the disp32 -- a four-entry whitelist (48 8d 05, 48 8b 1d, 48 89 05,
# f2 0f 10 05/0d). Any other encoding was skipped with no diagnostic: `mov
# rcx,[rip+d]` (48 8b 0d) or `mov r8,[rip+d]` (4c 8b 05) stayed [rip+0] and read
# the next instruction's bytes, rc=0, wrong answer. The profiling-counter dump's
# own `mov rax,[rip+d]` (48 8b 05) was outside the whitelist too. Kind 1 (call)
# had the same silent skip on a non-e8 site or an out-of-range target.
#
# Now the acceptor takes any REX.W[+R] 8B/89/8D or F2 [REX] 0F 10/11 with
# modrm mod=00 rm=101 -- every [rip+disp32] form whose disp32 is the last field
# -- and every other unpatched relocation makes the ELF writers refuse (rc=25).
#
# Witness (--native-v2-emit-reloc-shape <shape> <out>, main stores 37 in a .data
# slot, reads it back through the shape under test with a kind-3 relocation):
#   shape 0  mov rcx,[rip+d]            must run and exit 37
#   shape 1  mov r8,[rip+d]  (REX.R)    must run and exit 37
#   shape 2  mov dword [rip+d],imm32    disp32 is NOT last -> must refuse, rc=25,
#                                       no ELF written
# Unpatched, shapes 0/1 exit with the low byte of the next instruction (0x48=72
# for shape 0, 0x4c=76 for shape 1); shape 2 would write an ELF.
#
# Usage: MADAROS_RAW_BIN=<current-source Madaros ELF> bash scripts/ci/native_reloc_shape_gate.sh
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9
. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "native_reloc_shape_gate"

RAW="${MADAROS_RAW_BIN:-}"
[[ -n "$RAW" ]] || gate_fail "MADAROS_RAW_BIN must name a Madaros built from this tree (a prebuilt lags source)"
require_executable "$RAW"
WRAPPER="$ROOT_DIR/bin/madaros"
require_executable "$WRAPPER"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-reloc-shape.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

emit() {
  local shape="$1" out="$2" log="$3"
  MADAROS_RAW_BIN="$RAW" "$WRAPPER" --native-v2-emit-reloc-shape "$shape" "$out" >"$log" 2>&1
  return 0
}

for shape in 0 1; do
  elf="$WORK/reloc_shape_${shape}.bin"
  log="$WORK/reloc_shape_${shape}.log"
  emit "$shape" "$elf" "$log"
  if grep -Fq "label=reloc-shape rc=25" "$log"; then
    # Refusing is fail-closed, but these two ARE patchable forms: a refusal here
    # means the acceptor regressed to a whitelist.
    cat "$log" >&2
    gate_fail "shape $shape (a last-field [rip+disp32] form) was refused instead of patched"
  fi
  require_text "emitted label=reloc-shape" "$log"
  require_elf "$elf"
  chmod +x "$elf"
  "$elf"
  got=$?
  [[ -n "$got" ]] || gate_fail "shape $shape: no exit status captured"
  if [[ "$got" -ne 37 ]]; then
    gate_fail "shape $shape: exit $got, expected 37 -- the relocation was left at [rip+0] (unpatched shapes exit 72/76)"
  fi
  echo "[reloc-shape] PASS shape $shape: patched, exit 37"
done

elf="$WORK/reloc_shape_2.bin"
log="$WORK/reloc_shape_2.log"
emit 2 "$elf" "$log"
if [[ -e "$elf" ]]; then
  cat "$log" >&2
  gate_fail "shape 2 (disp32 followed by imm32) produced an ELF -- an unpatchable relocation was not refused"
fi
require_text "label=reloc-shape rc=25" "$log"
require_text "Error: native relocation left unpatched" "$log"
require_text "reason=unrecognised-instruction-shape" "$log"
require_text_regex "bytes_before=([0-9a-f]{2} )*c7 05" "$log"
echo "[reloc-shape] PASS shape 2: refused rc=25, no ELF, message names the bytes"

gate_pass "RIP-relative relocations are patched (mov rcx / mov r8 via [rip+d]) or refused (rc=25), never left at [rip+0]"
