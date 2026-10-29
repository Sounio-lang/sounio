#!/bin/bash
#SBATCH --job-name=legacy-flow-final
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-producer-custody01/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-producer-custody01/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=$B/compiler-resumed/root5-producer-custody01
S=$B/compiler-resumed/generic-module-native/source
RAW=$B/root-integrated04/results/madaros
R=$B/root-integrated04
source "$R/tools-env-v3.sh"
ulimit -s 524288;ulimit -c 0
cd "$D"
F=$B/compiler-resumed/root5-final-transport04
test "$(cat "$F/GATE_STATUS")" = 0
test "$(cat "$F/results/after/ORACLE_STATUS")" = 0
test "$(cat "$F/results/generation2/ORACLE_STATUS")" = 0
test "$(cat "$F/results/SELFCOMPILE_STATUS")" = 0
for sub in root5-final-profiles05 root5-final-metadata02 root5-final-reductions03 root5-final-controls02;do test "$(cat "$B/compiler-resumed/$sub/STATUS")" = PASS;done
for status in after native oracle;do test "$(cat "$B/compiler-resumed/root5-exact-consumers02/results/$status.STATUS")" = 0;done
test "$(cat "$B/compiler-resumed/root5-exact-consumers02/results/before.STATUS")" = 1
test "$(cat "$B/compiler-resumed/root5-proof-component03/results/native.STATUS")" = 0
test "$(cat "$B/compiler-resumed/root5-proof-component03/results/oracle.STATUS")" = 0
sha256sum "$F/source/self-hosted/compiler/lean_single.sio" "$F/results/candidate-producer.elf" "$F/results/generation2-producer.elf" >QUALIFIED.sha256
printf '%s  %s\n' 815661329ab8a1e65c851982b1a4b4b4ae504d63c6dcea911c95602f5a8863b0 "$F/source/self-hosted/compiler/lean_single.sio" f2ab364356061eb9f092741e0c0e2ecea0c02018745a5b83e8b9926e547a57f8 "$F/results/generation2-producer.elf" |sha256sum -c >PIN.verify
out=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/root5-producer-flat-final
mkdir -p "$out"
test ! -e "$out/producer-matrices-sdk-receipts.tar.gz"
tar czf "$out/producer-matrices-sdk-receipts.tar.gz" -C "$B/compiler-resumed" root5-final-transport04 root5-final-profiles05 root5-final-metadata02 root5-final-reductions03 root5-exact-consumers02 root5-proof-component03 root5-final-controls02
sha256sum "$out/producer-matrices-sdk-receipts.tar.gz" >CUSTODY.sha256
sha256sum -c CUSTODY.sha256 >CUSTODY.verify
cat QUALIFIED.sha256 CUSTODY.sha256
echo PASS >STATUS

