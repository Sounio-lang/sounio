#!/bin/bash
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:15:00
#SBATCH --export=NIL
set -euo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
ulimit -s 524288
cd /scratch/sounio-stdlib-inference-scale-20261024/SIO168/source01
out=../phase01
export MADAROS_RAW_BIN=/scratch/sounio-stdlib-inference-scale-20261024/compiler_ci_c21/madaros.elf.ready
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
printf 'a292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3  %s\n' "$MADAROS_RAW_BIN" | sha256sum -c -
sha256sum "$MADAROS_RAW_BIN" bin/souc-lean-single-x86_64 > "$out/compiler.sha256"
sha256sum stdlib/collections/numeric_index_sort.sio tests/reference/numeric_index_sort_reference.cpp > "$out/inputs.sha256"
g++ -std=c++23 -O2 -Wall -Wextra tests/reference/numeric_index_sort_reference.cpp -o "$out/reference" > "$out/reference.compile.stdout" 2> "$out/reference.compile.stderr"
"$out/reference" --generate > "$out/generated.sio"
SOUNIO_SOUC_ENGINE=lean_single bin/souc check "$out/generated.sio" > "$out/seed.stdout" 2> "$out/seed.stderr"
if grep -Eq '^error:|error\[E' "$out/seed.stdout" "$out/seed.stderr";then exit 1;fi
/usr/bin/time -v bin/souc compile "$out/generated.sio" -o "$out/native.elf" > "$out/compile.stdout" 2> "$out/compile.stderr"
if grep -Eq '^error:|error\[E' "$out/compile.stdout" "$out/compile.stderr";then exit 1;fi
chmod +x "$out/native.elf";sha256sum "$out/native.elf" > "$out/elf.sha256"
/usr/bin/time -v timeout 240 "$out/native.elf" > "$out/native.stdout" 2> "$out/native.stderr"
"$out/reference" "$out/native.stdout" > "$out/verify.stdout" 2> "$out/verify.stderr"
echo INDEX_SORT_GATE_PASS > "$out/gate.result"
