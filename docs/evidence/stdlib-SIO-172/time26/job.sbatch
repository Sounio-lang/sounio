#!/usr/bin/env bash
#SBATCH --job-name=SIO172-time26
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:15:00
#SBATCH --export=NIL
set -uo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
p=/scratch/sounio-stdlib-inference-scale-20261024/SIO172/time26
export MADAROS_RAW_BIN=/scratch/sounio-stdlib-inference-scale-20261024/compiler_ci_c21/madaros.elf.ready
export PHONON_GATE_OUTPUT="$p/artifacts"
cd "$p/source"
rc=0;bash scripts/stdlib/phonon_factor_gate.sh > "$p/gate.stdout" 2> "$p/gate.stderr"||rc=$?
echo "$rc">"$p/gate.status"
cp -a "$p/source" "$p/before_source"
cp "$p/before_source/docs/evidence/stdlib-SIO-172/time_before/phonon_factor_original.sio" "$p/before_source/stdlib/physics/phonon_factor.sio"
mkdir "$p/before_artifacts";cd "$p/before_source";export SOUNIO_STDLIB_PATH="$PWD/stdlib";out="$p/before_artifacts";src=docs/evidence/stdlib-SIO-172/time_before/time_fixture.sio
find stdlib tests docs/evidence/stdlib-SIO-172/time_before -type f -print0|sort -z|xargs -0 sha256sum > "$out/SOURCE.sha256"
rc=0;timeout 120 bin/souc check "$src" >"$out/check.stdout" 2>"$out/check.stderr"||rc=$?;echo "$rc">"$out/check.status"
if [[ $rc == 0 ]];then
 rc=0;timeout 180 bin/souc compile "$src" -o "$out/before.elf" >"$out/compile.stdout" 2>"$out/compile.stderr"||rc=$?;echo "$rc">"$out/compile.status"
 if [[ $rc == 0 ]];then chmod +x "$out/before.elf";timeout 30 "$out/before.elf">"$out/native.stdout" 2>"$out/native.stderr";echo "$?">"$out/native.status";sha256sum "$out/before.elf">"$out/ELF.sha256";fi
fi
if [[ -x $p/artifacts/reference ]];then "$p/artifacts/reference" time_before "$out/native.stdout">"$out/reference.stdout" 2>"$out/reference.stderr";echo "$?">"$out/reference.status";fi
sha256sum -c "$p/artifacts/COMPILER.sha256">"$p/compiler.postverify.txt"
