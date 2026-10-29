#!/bin/bash
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027/compiler-error-site05
D=$B/results;S=$B/source
RAW=/scratch/sounio-stdlib-wave3-20261027/root-integrated02/results/madaros
mkdir "$D"
cp -a /scratch/sounio-stdlib-wave3-20261027/whole-integrated01/source "$S"
export PATH=/usr/bin:/bin SOUNIO_STDLIB_PATH="$S/stdlib" SOUNIO_LOOP_REGIONS=1 SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
ulimit -s 524288;ulimit -c 0
trap 'r=$?;echo "$r" >"$D/STATUS";exit "$r"' EXIT
cd "$S"
printf '8abe891a66c71bdbeb784a9200d1ea395d4d5a55f0036a0daa51ad3cf0614932  %s\n' "$RAW" >"$D/COMPILER.sha256"
sha256sum -c "$D/COMPILER.sha256" >"$D/COMPILER.before.verify"
sha256sum "$B/trace.cpp" "$B/trace.original.cpp" "$B/signal-control.cpp" >"$D/TOOLS.sha256"
find stdlib tests/stdlib/chemistry -type f -print0|sort -z|xargs -0 sha256sum >"$D/SOURCE.sha256"
g++ -std=c++23 -O2 -Wall -Wextra -Werror "$B/trace.cpp" -o "$D/trace"
g++ -std=c++23 -O2 -Wall -Wextra -Werror "$B/signal-control.cpp" -o "$D/signal-control"
rc=0
timeout 15 "$D/trace" "$D/signal-control" >"$D/signal-control.stdout" 2>&1 ||rc=$?
[[ "$rc" == 139 ]]
grep -q 'fault_signal=11' "$D/signal-control.stdout"
for name in test_chem_network_ode_sys5 test_kinetics_gri_mech;do
 rc=0
 timeout 300 "$D/trace" "$RAW" compile "tests/stdlib/chemistry/$name.sio" -o "$D/$name.elf" >"$D/$name.stdout" 2>&1 ||rc=$?
 printf '%s\t%s\n' "$name" "$rc" >>"$D/RESULTS.tsv"
 [[ "$rc" == 139 ]]
 grep -q 'fault_signal=11' "$D/$name.stdout"
done
rc=0
timeout 300 "$D/trace" "$RAW" compile tests/stdlib/chemistry/test_catalysis_stdlib.sio -o "$D/catalysis.elf" >"$D/catalysis.compile.stdout" 2>&1 ||rc=$?
[[ "$rc" == 0 && -s "$D/catalysis.elf" ]]
grep -Fxq 'exit=0' "$D/catalysis.compile.stdout"
chmod +x "$D/catalysis.elf"
timeout 60 "$D/catalysis.elf" >"$D/catalysis.native.stdout" 2>&1
grep -Fxq CATALYSIS_STDLIB_OK "$D/catalysis.native.stdout"
sha256sum "$D/trace" "$D/signal-control" "$D/catalysis.elf" >"$D/ELFS.sha256"
readelf -l "$RAW" >"$D/RAW_SEGMENTS"
sha256sum -c "$D/COMPILER.sha256" >"$D/COMPILER.final.verify"
sha256sum -c "$D/TOOLS.sha256" >"$D/TOOLS.final.verify"
sha256sum -c "$D/SOURCE.sha256" >"$D/SOURCE.final.verify"
objdump -D -b binary -m i386:x86-64 --adjust-vma=0x400000 --start-address=0x5884a04 --stop-address=0x5884b40 "$RAW" >"$D/REPORTER_DISASSEMBLY.txt"
echo COMPILER_CRASH_DIAGNOSTIC_CAPTURED_NOT_A_FIX
