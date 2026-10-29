#!/bin/bash
#SBATCH --job-name=compiler-oracle-custody
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/sig139-supplement01/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/sig139-supplement01/slurm-%j.err
set -euo pipefail
D=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/sig139-supplement01
B=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed
trap 'r=$?;echo "$r" > "$D/WORKER_STATUS";exit "$r"' EXIT
O=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/sig139-census01
A="$O/packet-oracle-and-wide-cfg-supplement.tar.gz"
test ! -e "$A"
tar czf "$A" -C "$B" sig139-oracle01 wide-mustdef01
sha256sum "$A" > "$D/ARCHIVE.sha256"
sha256sum -c "$D/ARCHIVE.sha256" > "$D/ARCHIVE.readback.verify"
tar tzf "$A" > "$D/ARCHIVE.LIST"
cp "$D/ARCHIVE.sha256" "$O/supplement-archive.sha256"
cp "$D/ARCHIVE.readback.verify" "$O/supplement-archive.readback.verify"
echo SUPPLEMENT_CUSTODY_READBACK_OK > "$D/STATUS"
