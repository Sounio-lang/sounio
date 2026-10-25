#!/usr/bin/env bash
# Test-only ISA witness, never a Sounio backend or interval certification.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
out=${1:?fresh absolute output required}
[[ $(uname -m) == x86_64 && "$out" == /* ]] || exit 64
mkdir -p "$(dirname "$out")";mkdir "$out" || exit 65
trap 'rc=$?;printf "%s\n" "$rc" > "$out/STATUS"' EXIT
mpfr_inc=${ER_MPFR_INCLUDE:-/usr/include}
gmp_inc=${ER_GMP_INCLUDE:-/usr/include/x86_64-linux-gnu}
[[ -f "$mpfr_inc/mpfr.h" && -f "$gmp_inc/gmp.h" ]] || { echo 'identified MPFR/GMP headers required' >&2;exit 64; }
cd "$root"
printf 'phase\texit\n' > "$out/RESULTS.tsv"
printf 'control\texpected\tactual\n' > "$out/CONTROLS.tsv"
step(){ local name=$1;shift;local rc=0;"$@" > "$out/$name.stdout" 2> "$out/$name.stderr" || rc=$?;printf '%s\t%s\n' "$name" "$rc" >> "$out/RESULTS.tsv";return "$rc"; }
expected(){ local name=$1 want=$2;shift 2;local rc=0;step "$name" "$@" || rc=$?;printf '%s\t%s\t%s\n' "$name" "$want" "$rc" >> "$out/CONTROLS.tsv";[[ $rc == "$want" ]]; }
printf '%s\n' 'scope=test-only C++23 finite-input directed primitives; no Sounio/interval/decorations/backends certificate' > "$out/SCOPE.txt"
printf '%s\n' 'no_YMM_embedded_rounding;scalar_XMM_and_ZMM_only;strict_MXCSR_0x1F80_per_kernel' >> "$out/SCOPE.txt"
sha256sum tests/reference/interval_x86_er_reference.cpp scripts/stdlib/interval_x86_er_gate.sh "$mpfr_inc/mpfr.h" "$gmp_inc/gmp.h" > "$out/INPUTS.sha256"
step compiler-version g++ --version
sha256sum "$(readlink -f "$(command -v g++)")" > "$out/COMPILER.sha256"
step cpu lscpu
step build g++ -std=c++23 -O2 -Wall -Wextra -Werror -fno-fast-math -ffp-contract=off -frounding-math -fno-tree-vectorize -fno-tree-slp-vectorize -I"$mpfr_inc" -I"$gmp_inc" tests/reference/interval_x86_er_reference.cpp -l:libmpfr.so.6 -l:libgmp.so.10 -o "$out/reference"
sha256sum "$out/reference" > "$out/BINARY.sha256"
step libraries ldd "$out/reference"
mapfile -t oracle_libraries < <(awk '$1~/^lib(mpfr|gmp)\.so/{print $3}' "$out/libraries.stdout")
[[ ${#oracle_libraries[@]} == 2 ]]
printf '%s\n' "${oracle_libraries[@]}" | grep -Fq libmpfr.so.6
printf '%s\n' "${oracle_libraries[@]}" | grep -Fq libgmp.so.10
sha256sum "${oracle_libraries[@]}" > "$out/ORACLE_LIBRARIES.sha256"
step disassembly objdump -d -C -Mintel "$out/reference"
# Every operation/format/width has both directed instruction identities.
: > "$out/DIRECTED_INSTRUCTIONS.txt"
for op in add sub mul div;do
 for shape in ss sd ps pd;do
  for round in rd ru;do
   grep -E "v${op}${shape}[[:space:]].*\\{${round}-sae\\}" "$out/disassembly.stdout" >> "$out/DIRECTED_INSTRUCTIONS.txt"
  done
 done
done
if grep -Ei 'ymm.*\{(rd|ru)-sae\}|v(fmadd|fmsub|fnmadd|fnmsub)' "$out/disassembly.stdout" > "$out/FORBIDDEN_INSTRUCTIONS.txt";then echo 'forbidden rounding width or implicit FMA' >&2;exit 8;fi
step million-cases /usr/bin/time -v -o "$out/million-cases.resource" timeout 900 "$out/reference" --cases 1000000
[[ $(grep -Ec '^ER_CASES format=(32|64) op=[0-3] cases=1000000 scalar=PASS ZMM=PASS bit_identity=PASS digest_fnv64=' "$out/million-cases.stdout") == 8 ]]
grep -Fxq 'ER_TOTAL indexed_cases=8000000 scalar_cases=8000000 zmm_lanes=8000000' "$out/million-cases.stdout"
grep -Fxq 'EVEX_ER_MPFR_PASS formats=2 operations=4 variants=scalar,ZMM environment_controls=5_per_format' "$out/million-cases.stdout"
for mode in rn swapped flush;do
 expected "mutant-$mode" 8 timeout 30 "$out/reference" --cases 512 "--mutant-$mode"
 grep -Fq 'rounding mismatch' "$out/mutant-$mode.stderr"
 grep -Fq 'directed endpoint disagreement' "$out/mutant-$mode.stderr"
done
expected mutant-env-bypass 8 timeout 30 "$out/reference" --cases 512 --mutant-env-bypass
grep -Fq 'environment refusal missing (scalar)' "$out/mutant-env-bypass.stderr"
step unchanged-inputs sha256sum -c "$out/INPUTS.sha256"
step unchanged-libraries sha256sum -c "$out/ORACLE_LIBRARIES.sha256"
step unchanged-compiler sha256sum -c "$out/COMPILER.sha256"
echo EVEX_ER_WITNESS_GATE_PASS_TEST_ONLY
