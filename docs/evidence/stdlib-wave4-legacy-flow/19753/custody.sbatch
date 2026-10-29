#!/bin/bash
#SBATCH --job-name=legacy-flow-custody
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/legacy-flow-final01/custody-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/legacy-flow-final01/custody-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=$B/compiler-resumed/legacy-flow-final01
S=$B/compiler-resumed/generic-module-native/source
A=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/legacy-flow-final01
cd "$D"
sha256sum -c SOURCE.before.sha256 > SOURCE.custody.verify
sha256sum -c SDK.before.sha256 > SDK.custody.verify
mkdir -p "$A" compact
for f in worker.sbatch GATE_STATUS gate.stdout gate.stderr SOURCE.before.sha256 SOURCE.final.verify SDK.before.sha256 SDK.final.verify aarch64.guard.stdout; do cp "$f" compact/; done
for f in INPUTS.before.sha256 INPUTS.final.verify baseline.BUILD_STATUS candidate.BUILD_STATUS SELFCOMPILE_STATUS wrong_value.ORACLE_STATUS duplicate_output.ORACLE_STATUS missing_output.ORACLE_STATUS wrong_value.oracle.stdout duplicate_output.oracle.stdout missing_output.oracle.stderr; do cp "results/$f" compact/; done
for a in before after generation2; do
 mkdir -p "compact/$a"
 cp "results/$a/oracle.stdout" "results/$a/oracle.stderr" "results/$a/ORACLE_STATUS" "compact/$a/"
done
sha256sum results/baseline-producer.elf results/candidate-producer.elf results/generation2-producer.elf results/reference source/self-hosted/compiler/lean_single.sio > compact/QUALIFIED.sha256
cp "$B/compiler-resumed/legacy-flow-matrix01/after/oracle.stdout" compact/19747.after.oracle.stdout
cp "$B/compiler-resumed/legacy-flow-matrix01/before/oracle.stdout" compact/19747.before.oracle.stdout
cp "$B/compiler-resumed/legacy-flow-matrix01/SELFCOMPILE_STATUS" compact/19747.SELFCOMPILE_STATUS
cp "$B/compiler-resumed/legacy-flow-matrix01/selfcompile.stdout" compact/19747.selfcompile.stdout
tar czf "$A/producer-sdk-matrix-receipts.tar.gz" -C "$B" compiler-resumed/legacy-flow-final01/source compiler-resumed/legacy-flow-final01/results compiler-resumed/legacy-flow-final01/compact compiler-resumed/legacy-flow-final01/worker.sbatch compiler-resumed/legacy-flow-final01/SOURCE.before.sha256 compiler-resumed/legacy-flow-final01/SDK.before.sha256 compiler-resumed/legacy-flow-baseline01/baseline.sio -C "$S" self-hosted bin/souc-lean-single-x86_64
sha256sum "$A/producer-sdk-matrix-receipts.tar.gz" > compact/CUSTODY.sha256
cp compact/CUSTODY.sha256 "$A/"
sha256sum -c SOURCE.before.sha256 > SOURCE.custody.final.verify
sha256sum -c SDK.before.sha256 > SDK.custody.final.verify
find compact -type f -print0 | sort -z | xargs -0 sha256sum > COMPACT.sha256
