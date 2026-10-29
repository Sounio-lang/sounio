#!/bin/bash
#SBATCH --job-name=bidi-canonical-compact-receipt
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/bidi-canonical-receipt67/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/bidi-canonical-receipt67/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=$B/services/bidi-canonical-receipt67
O=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services/bidi-canonical-readback-export13
NODE=$B/services-tools-recovered/tools/node
mkdir "$D/metadata"
finish(){ r=$?;trap - EXIT;echo "$r" >"$D/metadata/STATUS";exit "$r";};trap finish EXIT
test "$(cat "$B/services/bidi-canonical-readback66/STATUS")" = 0
sha256sum "$NODE" "$D/input/collect.cjs" "$0" >"$D/metadata/TOOLS.sha256"
"$NODE" "$D/input/collect.cjs" "$D" >"$D/metadata/collect.stdout" 2>"$D/metadata/collect.stderr"
cp "$0" "$D/input/compact.sbatch"
find "$D/input" "$D/compact" -type f -print0|sort -z|xargs -0 sha256sum >"$D/metadata/CONTENTS.sha256"
tar -czf "$D/bidi-canonical-readback.tar.gz" -C "$D" input compact metadata
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
