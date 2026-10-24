#!/bin/bash
set -euo pipefail
r=/scratch/sounio-stdlib-advanced-20261023/lr_fixed_cpp01
d=/scratch/sounio-stdlib-advanced-20261023/lr_diagnostic02
ulimit -s 524288
for case in 3 11 13; do [[ $(cat "$d/case-$case.status") == 0 ]]; grep -qx LR_DIAGNOSTIC_COMPLETE "$d/case-$case.stdout"; done
sha256sum "$r/source/"*.cpp "$d"/case-{3,11,13}.stdout > "$r/INPUTS.sha256"
rc=0
/usr/bin/time -v -o "$r/compile.resource" g++ -std=c++23 -O2 "$r/source/count_profile_lr_fixed_reference.cpp" -lquadmath -o "$r/lr_fixed_reference" > "$r/compile.stdout" 2> "$r/compile.stderr" || rc=$?
printf '%s\n' "$rc" > "$r/compile.status"; [[ $rc == 0 ]]
rc=0
/usr/bin/time -v -o "$r/native.resource" "$r/lr_fixed_reference" "$d"/case-{3,11,13}.stdout > "$r/native.stdout" 2> "$r/native.stderr" || rc=$?
printf '%s\n' "$rc" > "$r/native.status"
[[ $rc == 0 ]]
rc=0
"$r/lr_fixed_reference" --mutant-sign "$d"/case-{3,11,13}.stdout > "$r/mutant-sign.stdout" 2> "$r/mutant-sign.stderr" || rc=$?
printf '%s\n' "$rc" > "$r/mutant-sign.status"; [[ $rc == 2 ]]
sha256sum "$r/lr_fixed_reference" "$r/native.stdout" "$r/mutant-sign.stdout" "$r/mutant-sign.stderr" > "$r/OUTPUTS.sha256"
printf '0\n' > "$r/STATUS"
