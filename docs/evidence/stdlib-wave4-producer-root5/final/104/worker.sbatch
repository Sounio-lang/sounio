#!/bin/bash
#SBATCH --job-name=legacy-flow-final
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-final-transport04/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-final-transport04/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=$B/compiler-resumed/root5-final-transport04
R=$B/root-integrated04
S=$B/compiler-resumed/generic-module-native/source
source "$R/tools-env-v3.sh"
cd "$D"
find source -type f -print0 | sort -z | xargs -0 sha256sum > SOURCE.before.sha256
find "$S/self-hosted" -type f -name '*.sio' -print0 | sort -z | xargs -0 sha256sum > SDK.before.sha256
perl source/tests/fixtures/wave4_legacy_flow/aarch64_source_guard.pl "$B/compiler-resumed/legacy-flow-baseline01/baseline.sio" source/self-hosted/compiler/lean_single.sio > aarch64.guard.stdout
set +e
bash source/scripts/compiler/legacy_flow_gate.sh "$B/compiler-resumed/legacy-flow-baseline01/baseline.sio" "$D/source/self-hosted/compiler/lean_single.sio" "$S/bin/souc-lean-single-x86_64" "$S/self-hosted" "$D/results" > gate.stdout 2>gate.stderr
rc=$?
set -e
echo "$rc" > GATE_STATUS
sha256sum -c SOURCE.before.sha256 > SOURCE.final.verify
sha256sum -c SDK.before.sha256 > SDK.final.verify
exit "$rc"

