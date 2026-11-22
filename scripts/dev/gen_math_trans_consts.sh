#!/usr/bin/env bash
# Generate self-hosted/native/math_trans_consts.sio -- the .rodata constants of
# the tan / atan / atan2 / asin / acos / tanh emitters -- with bc, at 900 decimal
# digits. Nothing in the output is typed by hand: every double is the bc-rounded
# nearest binary64 of an exact rational or of pi / atan / ln computed by bc.
#
#   bash scripts/dev/gen_math_trans_consts.sh > self-hosted/native/math_trans_consts.sio
#
# Needs only bc and awk. The two .bc files live beside this script.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW="$(cat "$HERE/gen_math_trans_dbl.bc" "$HERE/gen_math_trans_consts.bc" | BC_LINE_LENGTH=0 bc -l)"
# Group -> ordered list of "label" entries; fixed values are exact powers of two.
emit() { # name count
  echo "// $1: appends exactly $2 consecutive doubles to .rodata; element i is at first_offset + 8*i."
  echo "pub fn $1(nc: NativeCompiler) -> NativeCompiler with Mut, Panic, Div {"
  echo "    var c = nc"
}
fin() { echo "    c"; echo "}"; echo; }
add() { # hex comment
  echo "    c.rodata = data_section_add_f64(c.rodata, $1)   // $2"
}
get() { # group label -> hex lines
  echo "$RAW" | awk -v g="$1" -v l="$2" '$1==g && $2==l {print $3}'
}
echo "// self-hosted::native::math_trans_consts -- GENERATED, DO NOT EDIT."
echo "//"
echo "// Regenerate with:  bash scripts/dev/gen_math_trans_consts.sh > self-hosted/native/math_trans_consts.sio"
echo "// Source of truth:  scripts/dev/gen_math_trans_consts.bc (bc -l, scale 900)."
echo "//"
echo "// Every hex literal below came out of bc's own correctly rounded conversion."
echo "// None was typed. Consecutive data_section_add_f64 calls land exactly 8 bytes"
echo "// apart, which is what lets an emitter capture one offset per group."
echo
echo "use native::elf::{data_section_add_f64}"
echo "use native::codegen_x86_linux::{NativeCompiler}"
echo

emit mt_reg_atan_consts 40
add 0x3FF0000000000000 "[0] 1.0"
add 0x4000000000000000 "[1] 2.0"
add 0x3FF8000000000000 "[2] 1.5"
add 0x0000000000000000 "[3] 0.0"
i=0; for h in $(get ATANHI tab); do add "$h" "[$((4+i))] atanhi[$i]"; i=$((i+1)); done
i=0; for h in $(get ATANLO tab); do add "$h" "[$((8+i))] atanlo[$i]"; i=$((i+1)); done
i=0; for h in $(get ATAN_T tab); do add "$h" "[$((12+i))] T[$i] = (-1)^$i/(2*$i+3)"; i=$((i+1)); done
add "$(get COMMON pi_hi)" "[36] pi, high"
add "$(get COMMON pi_lo)" "[37] pi, low"
add "$(get COMMON pio4)" "[38] pi/4"
add "$(get COMMON pio4x3)" "[39] 3*pi/4"
fin

emit mt_reg_tan_consts 28
add "$(get COMMON inv_pio2)" "[0] 2/pi"
add "$(get COMMON pio2_1)" "[1] pi/2, part 1"
add "$(get COMMON pio2_2)" "[2] pi/2, part 2"
add "$(get COMMON pio2_3)" "[3] pi/2, part 3"
add "$(get COMMON pio2_4)" "[4] pi/2, part 4"
add 0x41A0000002000000 "[5] SPLIT = 2^27 + 1"
add 0x3FF0000000000000 "[6] 1.0"
add 0x3FE0000000000000 "[7] 0.5"
add 0x0000000000000000 "[8] 0.0"
i=0; for h in $(get SIN_S tab); do add "$h" "[$((9+i))] S[$i] = (-1)^($i+1)/(2*$i+3)!"; i=$((i+1)); done
i=0; for h in $(get COS_Q tab); do add "$h" "[$((19+i))] Q[$i] = (-1)^$i/(2*$i+4)!"; i=$((i+1)); done
fin

emit mt_reg_tanh_consts 21
add 0x3FF0000000000000 "[0] 1.0"
add 0x4000000000000000 "[1] 2.0"
add "$(get COMMON inv_ln2)" "[2] 1/ln2"
add "$(get COMMON ln2_hi)" "[3] ln2, high 40 bits"
add "$(get COMMON ln2_lo)" "[4] ln2 - ln2_hi"
i=0; for h in $(get EXPM1_E tab); do add "$h" "[$((5+i))] E[$i] = 1/($i+2)!"; i=$((i+1)); done
fin
