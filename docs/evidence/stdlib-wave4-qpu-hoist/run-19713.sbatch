#!/bin/bash
#SBATCH --job-name=qpu-historical-dyadic-hoist-contract
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/qpu-dyadic-compat60/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/qpu-dyadic-compat60/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
R=$B/root-integrated04
source "$R/tools-env-v3.sh"
D=$B/services/qpu-dyadic-compat60
S=$B/services/qpu-dyadic-compat58/input
RAW=$R/results/madaros
NODE=$B/services-tools-recovered/tools/node
mkdir "$D/receipts"
finish(){ r=$?;trap - EXIT;for m in INPUTS TOOLS;do sha256sum -c "$D/receipts/$m.sha256" >"$D/receipts/$m.final.verify" 2>&1||r=1;done;echo "$r" >"$D/STATUS";exit "$r";};trap finish EXIT
find "$S" -type f -print0|sort -z|xargs -0 sha256sum >"$D/receipts/INPUTS.sha256"
(find "$R/private-tools-v3" "$R/private-runtime" "$G" -type f -print0;printf '%s\0' "$R/tools-env-v3.sh" "$RAW" "$NODE" "$0")|sort -zu|xargs -0 sha256sum >"$D/receipts/TOOLS.sha256"
"$NODE" - "$S" "$D" <<'JS'
const fs=require('fs'),cp=require('child_process'),s=process.argv[2],d=process.argv[3];
const old=fs.readFileSync(s+'/docs/evidence/stdlib-wave4-qpu-hoist/historical-dyadic-gate.preimage.sh','utf8');
const code=old.split("<<'JS'\n")[1].split('\nJS')[0];
const p=cp.spawnSync(process.execPath,['-',s,d,s+'/stdlib'],{input:code,encoding:'utf8'});
fs.writeFileSync(d+'/receipts/before-source-validation.stdout',p.stdout);
fs.writeFileSync(d+'/receipts/before-source-validation.stderr',p.stderr);
fs.writeFileSync(d+'/receipts/before-source-validation.status',String(p.status)+'\n');
if(p.status!==1||!p.stderr.includes('source body mismatch'))throw Error('original source refusal not demonstrated');
JS
mkdir "$D/native"
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror -shared -fPIC -pthread "$S/runtime/stdlib_platform.cpp" -o "$D/native/libstdlib_platform.so" >"$D/receipts/runtime.build.stdout" 2>"$D/receipts/runtime.build.stderr"
sha256sum "$D/native/libstdlib_platform.so" >"$D/receipts/RUNTIME.sha256"
export LD_LIBRARY_PATH="$D/native:$LD_LIBRARY_PATH"
export NODE_BIN="$NODE" SOUNIO_LOOP_CHECKPOINT_ELISION=0
bash "$S/scripts/stdlib/wave4_qpu_dyadic_gate.sh" "$D/results" "$RAW" >"$D/gate.stdout" 2>"$D/gate.stderr"
