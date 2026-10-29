#!/bin/bash
#SBATCH --job-name=stdlib-frozen-argv-material
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/argv-source-guards44/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/argv-source-guards44/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027;R=$B/services/argv-source-guards44;S=$R/input;D=$R/results;U=$B/root-integrated02
export PATH=$B/services-tools-recovered/tools:/usr/bin:/bin
export SOUNIO_REPO_HARD_NO_RUST=1 SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1 SOUNIO_STDLIB_PATH="$S/stdlib" SOUC_BIN=/scratch/sounio-stdlib-wave3-20261027/services/argv-contract39/raw-wrapper
ulimit -s 524288;ulimit -c 0
finish(){ r=$?;trap - EXIT;for n in INPUTS COMPILER EXECUTABLES;do if [[ -s "$D/$n.sha256" ]];then sha256sum -c "$D/$n.sha256" >"$D/$n.final.verify" 2>&1||r=1;fi;done;echo "$r" >"$D/STATUS";exit "$r";};trap finish EXIT
cd "$S"
find "$S" -type f -print0|sort -z|xargs -0 sha256sum >"$D/INPUTS.sha256"
cp "$U/RAW_READY" "$D/COMPILER.sha256";sha256sum "$SOUC_BIN" >>"$D/COMPILER.sha256"
for t in argv_contract_gate.mjs test_stdlib_compile_fail.mjs test_stdlib_runtime_refusal.mjs test_stdlib_silent_refusal.mjs test_stdlib_e2e_diagnostics.mjs test_run_full_suite.mjs test_emit_stdlib_e2e_result.mjs;do node "scripts/stdlib/$t" >"$D/$t.stdout" 2>&1;echo "$t PASS";done
echo FINAL_SOURCE_GUARDS_AND_ARGV_CONTROLS_OK
