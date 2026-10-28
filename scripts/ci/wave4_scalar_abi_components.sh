#!/usr/bin/env bash
# Component qualification; the whole rebuilt compiler has a separate gate.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd -P)
[[ $# -ge 1 && $# -le 2 && "$1" == /* && ! -e "$1" ]] || exit 64
[[ "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]] || exit 64
: "${SEED_RAW_BIN:?absolute legacy seed ELF required}"
[[ "$SEED_RAW_BIN" == /* && -x "$SEED_RAW_BIN" ]] || exit 64
mode=${2:-after}
[[ "$mode" == after || "$mode" == before ]] || exit 64
OUT="$1"
mkdir -p "$OUT/source"
cp -a "$ROOT/self-hosted" "$ROOT/stdlib" "$OUT/source/"
cat "$ROOT/tests/fixtures/wave4_scalar_abi/lower_probe.sio.in" >> "$OUT/source/self-hosted/ir/lower.sio"
cat "$ROOT/tests/fixtures/wave4_scalar_abi/exit_probe.sio.in" >> "$OUT/source/self-hosted/native/codegen_x86_linux.sio"
cp "$ROOT/tests/fixtures/wave4_scalar_abi/lower_driver.sio.in" "$OUT/source/lower_driver.sio"
cp "$ROOT/tests/fixtures/wave4_scalar_abi/exit_driver.sio.in" "$OUT/source/exit_driver.sio"
cd "$OUT/source"
if [[ "$mode" == before ]];then
 grep -vF '    if ir_name_eq(name, make_name("read_f64")) { fn_slot.returns_float = 1 }' self-hosted/ir/lower.sio > "$OUT/lower.before"
 cp "$OUT/lower.before" self-hosted/ir/lower.sio
 sed '/^fn nc_emit_exit_code/,/^}/s/nc_emit_mov_rax_imm(nc, 231)/nc_emit_mov_rax_imm(nc, 60)/' self-hosted/native/codegen_x86_linux.sio > "$OUT/exit.before"
 cp "$OUT/exit.before" self-hosted/native/codegen_x86_linux.sio
fi
find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "$OUT/INPUTS.sha256"
sha256sum "$SEED_RAW_BIN" > "$OUT/SEED.sha256"
export SOUNIO_STDLIB_PATH="$OUT/source/self-hosted"
ulimit -s 524288
ulimit -c 0
fail=0
for k in lower exit;do
 c=0;n=999
 timeout 900 "$SEED_RAW_BIN" "$k"_driver.sio "$OUT/$k.elf" > "$OUT/$k.compile" 2>&1 || c=$?
 if [[ "$c" == 0 ]];then
  chmod +x "$OUT/$k.elf"
  sha256sum "$OUT/$k.elf" >> "$OUT/ELFS.sha256"
  n=0;timeout 60 "$OUT/$k.elf" > "$OUT/$k.native" 2>&1 || n=$?
  if [[ "$mode" == after && "$n" == 0 ]];then
   marker=RAW_READ_FLOAT_METADATA_OK
   [[ "$k" != exit ]] || marker=FATAL_EXIT_GROUP_EMISSION_OK
   grep -qx "$marker" "$OUT/$k.native" || fail=1
  fi
 fi
 expected=0
 if [[ "$mode" == before ]];then
  expected=31
  [[ "$k" != exit ]] || expected=42
 fi
 printf '%s\t%s\t%s\t%s\n' "$k" "$c" "$n" "$expected" >> "$OUT/RESULTS.tsv"
 [[ "$c:$n" == "0:$expected" ]] || fail=1
done
sha256sum -c "$OUT/INPUTS.sha256" > "$OUT/INPUTS.verify" || fail=1
sha256sum -c "$OUT/SEED.sha256" > "$OUT/SEED.verify" || fail=1
[[ ! -s "$OUT/ELFS.sha256" ]] || sha256sum -c "$OUT/ELFS.sha256" > "$OUT/ELFS.verify" || fail=1
printf '%s\n' "$fail" > "$OUT/STATUS"
exit "$fail"
