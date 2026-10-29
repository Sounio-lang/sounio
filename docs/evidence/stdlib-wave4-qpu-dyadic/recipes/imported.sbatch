#!/bin/bash
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027
R=$B/root-qpu-dyadic01
D=$R/imported-results
S=$R/library
RAW=$B/root-integrated02/results/madaros
mkdir "$D"
finish(){ r=$?;trap - EXIT;for m in SOURCE RAW EXECUTABLES;do if [[ -s "$D/$m.sha256" ]];then sha256sum -c "$D/$m.sha256" >"$D/$m.final.verify" 2>&1||r=1;fi;done;echo "$r" >"$D/STATUS";exit "$r"; }
trap finish EXIT
cd "$S"
find . -type f -print0|sort -z|xargs -0 sha256sum >"$D/SOURCE.sha256"
cp "$B/root-integrated02/RAW_READY" "$D/RAW.sha256"
sha256sum -c "$D/RAW.sha256" >"$D/RAW.before.verify"
export SOUNIO_STDLIB_PATH="$S/stdlib" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
ulimit -s 524288;ulimit -c 0
for mode in 0 1;do
 export SOUNIO_LOOP_REGIONS=$mode
 for file in tests/stdlib/qpu/test_qpu_chsh.sio tests/stdlib/quantum/test_born_shot_law.sio tests/stdlib/quantum/test_qasm_tape.sio;do
  tag=$(basename "$file" .sio)
  timeout 1200 "$RAW" check "$file" >"$D/$tag.$mode.check.stdout" 2>&1
  timeout 1200 "$RAW" compile "$file" -o "$D/$tag.$mode.elf" >"$D/$tag.$mode.compile.stdout" 2>&1
  chmod +x "$D/$tag.$mode.elf"
  sha256sum "$D/$tag.$mode.elf" >>"$D/EXECUTABLES.sha256"
  timeout 30 "$D/$tag.$mode.elf" >"$D/$tag.$mode.native.stdout" 2>&1
  grep -Fxq ALL_PASS "$D/$tag.$mode.native.stdout"
  printf '%s\t%s\tcheck0_compile0_native0\n' "$tag" "$mode" >>"$D/PHASES.tsv"
 done
done
echo QPU_DYADIC_IMPORTED_ORIGINALS_OK
