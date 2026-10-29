#!/bin/bash
#SBATCH --job-name=bidi-canonical-compact-receipt
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/bidi-canonical-receipt68/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/bidi-canonical-receipt68/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=$B/services/bidi-canonical-receipt68
O=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services/bidi-canonical-readback-export13
NODE=$B/services-tools-recovered/tools/node
mkdir "$D/metadata"
finish(){ r=$?;trap - EXIT;echo "$r" >"$D/metadata/STATUS";exit "$r";};trap finish EXIT
test "$(cat "$B/services/bidi-canonical-readback66/STATUS")" = 0
mkdir "$D/extra"
A=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/root/bidi-canonical-export07/bidi-complete-sources-data-elfs-tooling-attempts.tar.gz
tar -xzf "$A" -C "$D/extra" --no-same-owner root-bidi-inputs01/slurm-19616.log root-bidi-inputs03/slurm-19654.log bidi-canonical-19668.log bidi-canonical-19670.log bidi-canonical-19672.log root-bidi-inputs07.slurm.out
sha256sum "$NODE" "$D/input/collect.cjs" "$0" >"$D/metadata/TOOLS.sha256"
"$NODE" "$D/input/collect.cjs" "$D" >"$D/metadata/collect.stdout" 2>"$D/metadata/collect.stderr"
cp "$0" "$D/input/compact.sbatch"
find "$D/input" "$D/compact" "$D/extra" -type f -print0|sort -z|xargs -0 sha256sum >"$D/metadata/CONTENTS.sha256"
tar -czf "$D/bidi-canonical-readback.tar.gz" -C "$D" input compact metadata extra
gzip -t "$D/bidi-canonical-readback.tar.gz"
sha256sum -c "$D/metadata/CONTENTS.sha256" >"$D/metadata/CONTENTS.final.verify"
sha256sum -c "$D/metadata/TOOLS.sha256" >"$D/metadata/TOOLS.final.verify"
mkdir "$O"
cp "$D/bidi-canonical-readback.tar.gz" "$O/"
sha256sum "$O/bidi-canonical-readback.tar.gz" >"$O/ARCHIVE.sha256"
sha256sum -c "$O/ARCHIVE.sha256" >"$O/READBACK.verify"
stat -c '%s' "$O/bidi-canonical-readback.tar.gz" >"$O/ARCHIVE.bytes"
echo 0 >"$O/COPY_STATUS"
cp -a "$D/metadata" "$O/metadata"
echo BIDI_CANONICAL_COMPACT_CUSTODY_OK
