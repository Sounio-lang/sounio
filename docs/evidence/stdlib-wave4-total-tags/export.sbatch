#!/bin/bash
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=$B/compiler-resumed
E=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/compiler/total-tags-export01
mountpoint -q /orangefs/training
mkdir "$E" "$E/inputs"
sha256sum -c "$D/total-tags01/INPUTS.sha256" > "$E/prototype.inputs.verify"
sha256sum -c "$D/total-tags01/TOOLS.sha256" > "$E/prototype.tools.verify"
sha256sum -c "$D/total-tags02/gate/INPUTS.sha256" > "$E/published.inputs.verify"
sha256sum -c "$D/total-tags02/gate/TOOLS.sha256" > "$E/published.tools.verify"
cp "$B/root-integrated04/results/madaros" "$E/inputs/accepted-raw04-check-control.elf"
cp "$B/root-integrated04/source/bin/souc-lean-single-x86_64" "$E/inputs/old-seed9cf.elf"
cp "$B/root-integrated05/seed-private/generation2-producer.elf" "$E/inputs/final-seedccfa.elf"
cp "$B/root-integrated04/tools-env-v3.sh" "$E/inputs/tools-env-v3.sh"
cp /orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/compiler/raw05-failed-export01/ARCHIVE.sha256 "$E/inputs/FAILED_RAW05_ARCHIVE.sha256"
cd "$D"
find total-tags01 total-tags02 -type f -print0 | sort -z | xargs -0 sha256sum > "$E/BEFORE.sha256"
tar -czf "$E/total-tags-components.tar.gz" total-tags01 total-tags02 -C "$E" inputs
sha256sum "$E/total-tags-components.tar.gz" > "$E/ARCHIVE.sha256"
tar -tzf "$E/total-tags-components.tar.gz" > "$E/ARCHIVE_CONTENTS.txt"
sha256sum -c "$E/ARCHIVE.sha256"
find total-tags01 total-tags02 -type f -print0 | sort -z | xargs -0 sha256sum > "$E/AFTER.sha256"
cmp "$E/BEFORE.sha256" "$E/AFTER.sha256"
sacct -X -j19802,19808 --format=JobID,State,ExitCode,Elapsed,MaxRSS -P > "$E/SLURM.txt"
printf 'TOTAL_TAGS_EXPORT_OK_NO_NEW_MODERN_COMPILER\n' > "$E/STATUS"
