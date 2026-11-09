#!/usr/bin/env bash
#SBATCH --job-name=SIO172-SI-location
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:07:00
#SBATCH --export=NIL
set -uo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
export MADAROS_RAW_BIN=/scratch/sounio-stdlib-inference-scale-20261024/compiler_ci_c21/madaros.elf.ready
export SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
ulimit -s 524288
ulimit -c 0
p=/scratch/sounio-stdlib-inference-scale-20261024/SIO172/probe19
cd "$p/source";export SOUNIO_STDLIB_PATH="$PWD/stdlib";out="$p/artifacts"
src=docs/evidence/stdlib-SIO-172/probe19/si_trace.sio
sha256sum "$src" stdlib/physics/phonon_factor.sio stdlib/physics/phonon.sio "$MADAROS_RAW_BIN">"$out/INPUTS.sha256"
rc=0;timeout 120 bin/souc check "$src" > "$out/check.stdout" 2> "$out/check.stderr"||rc=$?;echo "$rc">"$out/check.status"
if [[ $rc == 0 ]];then
 rc=0;timeout 180 bin/souc compile "$src" -o "$out/trace.elf" > "$out/compile.stdout" 2> "$out/compile.stderr"||rc=$?;echo "$rc">"$out/compile.status"
 if [[ $rc == 0 ]];then chmod +x "$out/trace.elf";timeout 30 "$out/trace.elf" > "$out/native.stdout" 2> "$out/native.stderr";echo "$?">"$out/native.status";sha256sum "$out/trace.elf">"$out/ELF.sha256";fi
fi
rc=0;g++ -std=c++23 -O2 -Wall -Wextra -Werror docs/evidence/stdlib-SIO-172/probe19/trace.cpp -o "$out/tracer" > "$out/tracer.compile.stdout" 2> "$out/tracer.compile.stderr"||rc=$?;echo "$rc">"$out/tracer.compile.status"
if [[ $rc == 0 ]];then timeout 30 "$out/tracer" /scratch/sounio-stdlib-inference-scale-20261024/SIO172/qual18/artifacts/si.elf > "$out/tracer.stdout" 2> "$out/tracer.stderr";echo "$?">"$out/tracer.status";fi
