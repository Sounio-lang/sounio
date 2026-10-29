#!/bin/bash
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
E=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/compiler/pbpk-binding-stamp-export01
mountpoint -q /orangefs/training
mkdir "$E" "$E/inputs"
R=$B/root-integrated04
cp "$R/results/madaros" "$E/inputs/madaros.raw04"
cp "$R/source/bin/souc-lean-single-x86_64" "$E/inputs/seed.9cf"
cp "$R/tools-env-v3.sh" "$E/inputs/tools-env-v3.sh"
sha256sum "$R/results/madaros" "$R/source/bin/souc-lean-single-x86_64" "$R/tools-env-v3.sh" > "$E/EXECUTED_TOOLS.sha256"
cmp "$R/results/madaros" "$E/inputs/madaros.raw04"
cmp "$R/source/bin/souc-lean-single-x86_64" "$E/inputs/seed.9cf"
cd "$B/compiler-resumed"
for d in pbpk-binding-stamp01 pbpk-binding-stamp02 pbpk-binding-stamp03 pbpk-binding-stamp04 pbpk-binding-stamp05; do
  (cd "$d"; sha256sum -c SOURCE.sha256) > "$E/$d.source.verify"
done
(cd pbpk-binding-stamp06/gate; sha256sum -c SOURCE.sha256; sha256sum -c TOOLS.sha256; sha256sum -c ELF.sha256) > "$E/published.verify"
find pbpk-binding-stamp01 pbpk-binding-stamp02 pbpk-binding-stamp03 pbpk-binding-stamp04 pbpk-binding-stamp05 pbpk-binding-stamp06 -type f -print0 | sort -z | xargs -0 sha256sum > "$E/INPUTS.before.sha256"
tar -czf "$E/pbpk-binding-stamp.tar.gz" pbpk-binding-stamp01 pbpk-binding-stamp02 pbpk-binding-stamp03 pbpk-binding-stamp04 pbpk-binding-stamp05 pbpk-binding-stamp06 -C "$E" inputs
tar -tzf "$E/pbpk-binding-stamp.tar.gz" > "$E/ARCHIVE_CONTENTS.txt"
sha256sum "$E/pbpk-binding-stamp.tar.gz" > "$E/ARCHIVE.sha256"
sha256sum -c "$E/ARCHIVE.sha256"
sha256sum -c "$E/EXECUTED_TOOLS.sha256" > "$E/tools.verify"
find pbpk-binding-stamp01 pbpk-binding-stamp02 pbpk-binding-stamp03 pbpk-binding-stamp04 pbpk-binding-stamp05 pbpk-binding-stamp06 -type f -print0 | sort -z | xargs -0 sha256sum > "$E/INPUTS.after.sha256"
cmp "$E/INPUTS.before.sha256" "$E/INPUTS.after.sha256"
sacct -X -j19763,19765,19766,19767,19768,19771,19772 --format=JobID,State,ExitCode,Elapsed,MaxRSS -P > "$E/SLURM.inputs.txt"
printf 'PBPK_BINDING_STAMP_EXPORT_OK\n' > "$E/STATUS"
