#!/bin/bash
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=$B/compiler-resumed/storage-closed-audit05
E=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/compiler/storage-closed-audit-export02
mountpoint -q /orangefs/training
mkdir "$E" "$E/inputs" "$E/source-variants"
for variant in original candidate; do
 mkdir "$E/source-variants/$variant"
 cp -a "$D/source/." "$E/source-variants/$variant/"
 cp "$D/results/$variant.injected.storage_types.sio" "$E/source-variants/$variant/self-hosted/check/storage_types.sio"
 (cd "$E/source-variants/$variant"; sha256sum -c "$D/results/$variant.SOURCE.sha256") > "$E/$variant.reconstructed-source.verify"
done
cp "$B/root-integrated04/results/madaros" "$E/inputs/madaros.raw04"
cp "$B/root-integrated04/source/bin/souc-lean-single-x86_64" "$E/inputs/seed.9cf"
cp "$B/root-integrated04/tools-env-v3.sh" "$E/inputs/tools-env-v3.sh"
cp /orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/compiler/storage-closed-audit-export01/ARCHIVE.sha256 "$E/inputs/PRIOR_ARCHIVE.sha256"
sha256sum -c "$D/TOOLS.sha256" > "$E/tools.verify"
sha256sum -c "$D/INPUTS.sha256" > "$E/inputs.verify"
cd "$B/compiler-resumed"
find storage-closed-audit05 -type f -print0 | sort -z | xargs -0 sha256sum > "$E/INPUTS.before.sha256"
tar -czf "$E/storage-closed-audit-ref.tar.gz" storage-closed-audit05 -C "$E" inputs source-variants
sha256sum "$E/storage-closed-audit-ref.tar.gz" > "$E/ARCHIVE.sha256"
tar -tzf "$E/storage-closed-audit-ref.tar.gz" > "$E/ARCHIVE_CONTENTS.txt"
sha256sum -c "$E/ARCHIVE.sha256"
find storage-closed-audit05 -type f -print0 | sort -z | xargs -0 sha256sum > "$E/INPUTS.after.sha256"
cmp "$E/INPUTS.before.sha256" "$E/INPUTS.after.sha256"
sacct -X -j19786 --format=JobID,State,ExitCode,Elapsed,MaxRSS -P > "$E/SLURM.inputs.txt"
printf 'STORAGE_CLOSED_AUDIT_REF_EXPORT_OK\n' > "$E/STATUS"
