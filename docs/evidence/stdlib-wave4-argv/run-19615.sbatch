#!/bin/bash
#SBATCH --job-name=stdlib-argv-same-elf-controls
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/argv-same-elf47/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/argv-same-elf47/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027;R=$B/services/argv-same-elf47;P=$B/services/argv-final42;D=$R/results
mkdir "$D"
finish(){ r=$?;trap - EXIT;sha256sum -c "$D/ELF.sha256" >"$D/ELF.final.verify" 2>&1||r=1;sha256sum -c "$P/results/INPUTS.sha256" >"$D/INPUTS.final.verify" 2>&1||r=1;echo "$r" >"$D/STATUS";exit "$r";};trap finish EXIT
sha256sum -c "$P/results/EXECUTABLES.sha256" >"$D/ELF.before.verify"
cp "$P/results/EXECUTABLES.sha256" "$D/ELF.sha256"
cd "$P/input"
mapfile -t frozen <tests/stdlib/hardware/test_pireus_material_engine_admission.sio.args
timeout 30 "$P/results/material.elf" "${frozen[@]}" >"$D/frozen.stdout" 2>&1
echo 0 >"$D/frozen.status";grep -Fq 'pireus material engine admission test passed' "$D/frozen.stdout"
r=0;timeout 30 "$P/results/material.elf" >"$D/noargv.stdout" 2>&1||r=$?
[[ $r == 2 ]];echo "$r" >"$D/noargv.status"
mapfile -t current <"$P/control.original.args"
r=0;timeout 30 "$P/results/material.elf" "${current[@]}" >"$D/current-docs.stdout" 2>&1||r=$?
[[ $r == 1 ]];echo "$r" >"$D/current-docs.status";grep -Fq 'pireus material engine admission test failed: 9' "$D/current-docs.stdout"
echo SAME_ELF_FROZEN0_NOARGV2_CURRENTDOCS1_OK
