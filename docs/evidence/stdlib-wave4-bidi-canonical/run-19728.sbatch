#!/bin/bash
#SBATCH --job-name=bidi-canonical-independent-readback
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/bidi-canonical-readback66/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/bidi-canonical-readback66/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=$B/services/bidi-canonical-readback66
O=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/root/bidi-canonical-export07
A=$O/bidi-complete-sources-data-elfs-tooling-attempts.tar.gz
NODE=$B/services-tools-recovered/tools/node
mkdir "$D/extracted"
finish(){ r=$?;trap - EXIT;sha256sum -c "$D/TOOLS.sha256" >"$D/TOOLS.final.verify" 2>&1||r=1;echo "$r" >"$D/STATUS";exit "$r";};trap finish EXIT
sha256sum "$NODE" "$D/input/readback.cjs" "$0" >"$D/TOOLS.sha256"
printf '1f0d5d5f1b8e1a72b5bb083e554cf29e0765d614a84d47e76e53fb10b28cd14d  %s\n' "$A" >"$D/ARCHIVE.expected.sha256"
test "$(stat -c '%s' "$A")" = 1547845754
sha256sum -c "$D/ARCHIVE.expected.sha256" >"$D/ARCHIVE.before.verify"
tar -tzf "$A" >"$D/ARCHIVE_CONTENTS.readback.txt"
cmp "$D/ARCHIVE_CONTENTS.readback.txt" "$O/ARCHIVE_CONTENTS.txt"
echo 0 >"$D/INVENTORY_STATUS"
grep -E '^root-bidi-inputs(01|03|04|05|06|07)/results/' "$D/ARCHIVE_CONTENTS.readback.txt" >"$D/EXTRACT.list"
grep -E '^root-bidi-inputs(01|03)/(slurm-[0-9]+\\.log)$|^bidi-canonical-[0-9]+\\.log$|^root-bidi-inputs07.slurm.out$|^root-bidi-inputs05/source/(tests/fixtures/wave4_bidi_runtime/official-corpus.json|tests/stdlib/wave3-bidi/conformance.sio|scripts/stdlib/runtime_fixtures.mjs)$|^root-bidi-recipe-copies/|^root-bidi-inputs07.sh$' "$D/ARCHIVE_CONTENTS.readback.txt" >>"$D/EXTRACT.list"
LC_ALL=C sort -u "$D/EXTRACT.list" -o "$D/EXTRACT.list"
tar -xzf "$A" -C "$D/extracted" --no-same-owner --no-recursion --verbatim-files-from -T "$D/EXTRACT.list"
"$NODE" "$D/input/readback.cjs" "$D" >"$D/metadata.stdout" 2>"$D/metadata.stderr"
sha256sum -c "$D/ARCHIVE.expected.sha256" >"$D/ARCHIVE.final.verify"
find "$D/extracted" -type f -print0|sort -z|xargs -0 sha256sum >"$D/EXTRACTED.sha256"
sacct -X -j 19616,19618,19654,19668,19670,19672,19686,19708 --format=JobID,JobName,State,ExitCode,Elapsed,AllocCPUS,ReqMem,NodeList -P >"$D/ACCOUNTING"
echo BIDI_CANONICAL_INDEPENDENT_READBACK_OK
