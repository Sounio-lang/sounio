#!/bin/bash
#SBATCH --job-name=quantum-pair-library-originals
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380p-dl380p-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/quantum-pair-library25/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/quantum-pair-library25/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027;R=$B/services/quantum-pair-library25;D=$R/results;U=$B/root-integrated01
mkdir "$D"
finish(){ r=$?;trap - EXIT;for n in INPUTS COMPILER SOURCES WITNESSES;do if [[ -s "$D/$n.sha256" ]];then sha256sum -c "$D/$n.sha256" > "$D/$n.final.verify" 2>&1||r=1;fi;done;echo "$r" > "$D/COLLECTOR_STATUS";exit "$r";};trap finish EXIT
find "$R/input" -type f -print0|sort -z|xargs -0 sha256sum > "$D/INPUTS.sha256"
cp "$U/RAW_READY" "$D/COMPILER.sha256";sha256sum -c "$D/COMPILER.sha256" > "$D/COMPILER.before.verify"
find "$R/input/source/tests" -type f -print0|sort -z|xargs -0 sha256sum > "$D/WITNESSES.sha256"
export SOUNIO_LOOP_REGIONS=1 SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
ulimit -s 524288;ulimit -c 0
phase(){ local label=$1 seconds=$2;shift 2;local r=0;timeout "$seconds" "$@" > "$D/$label.stdout" 2>&1||r=$?;printf '%s\t%s\n' "$label" "$r" >> "$D/PHASES.tsv";return "$r"; }
fail=0
for variant in before after;do
 cp -a "$R/input/source" "$R/$variant"
 cp "$R/input/$variant/stdlib/quantum/"*.sio "$R/$variant/stdlib/quantum/"
 export SOUNIO_STDLIB_PATH="$R/$variant/stdlib"
 find "$R/$variant" -type f -print0|sort -z|xargs -0 sha256sum >> "$D/SOURCES.sha256"
 for lib in qasm_tape born_shot_law;do phase "$variant.$lib.library-check" 1200 "$U/results/madaros" --check "$R/$variant/stdlib/quantum/$lib.sio"||true;done
 while IFS=$'\t' read -r name path deadline;do
  source="$R/$variant/$path"
  rc=0;phase "$variant.$name.check" 1200 "$U/results/madaros" --check "$source"||rc=$?
  if [[ "$variant" == after && "$rc" != 0 ]];then fail=1;fi
  rc=0;phase "$variant.$name.compile" 1200 "$U/results/madaros" compile "$source" -o "$D/$variant.$name.elf"||rc=$?
  if [[ "$rc" == 0 && -s "$D/$variant.$name.elf" ]];then
   chmod +x "$D/$variant.$name.elf";sha256sum "$D/$variant.$name.elf" >> "$D/EXECUTABLES.sha256"
   rc=0;phase "$variant.$name.native" "$deadline" "$D/$variant.$name.elf"||rc=$?
   if [[ "$variant" == after && "$rc" != 0 ]];then fail=1;fi
  elif [[ "$variant" == after ]];then fail=1;fi
 done <<'CASES'
born_shot_law	tests/stdlib/quantum/test_born_shot_law.sio	30
qasm_tape	tests/stdlib/quantum/test_qasm_tape.sio	30
qpu_chsh	tests/stdlib/qpu/test_qpu_chsh.sio	30
overtake_slow	tests/stdlib/qpu/test_qpu_chsh_overtake_slow.sio	3000
CASES
done
echo 'Same whole19392 mode1/compiler deadlines1200, native30, explicit original slow3000; original programs untouched. Scientific C++ observables qualified separately.' > "$D/SCOPE"
[[ "$fail" == 0 ]]
echo QUANTUM_ORIGINAL_PROGRAMS_RECORDED
