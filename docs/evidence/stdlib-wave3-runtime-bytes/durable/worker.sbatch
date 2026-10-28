#!/bin/bash
#SBATCH --job-name=stdlib-runtime-bytes-custody
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/core/runtime-bytes-custody01/results/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/core/runtime-bytes-custody01/results/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
R="$B/core/runtime-bytes-custody01"
D=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/runtime-bytes01
mkdir -p "$R/stage" "$D"
trap 'r=$?;echo "$r" > "$R/results/WORKER_STATUS";date -u > "$R/results/ENDED_UTC";exit "$r"' EXIT
test "$(cat "$B/core/runtime-bytes03/results/WORKER_STATUS")" = 0
grep -Fx RUNTIME_BYTES_TWO_RAW_CPP23_ONLY_OK "$B/core/runtime-bytes03/results/STATUS"
for name in runtime-bytes01 runtime-bytes02 runtime-bytes03 runtime-bounds01 runtime-bounds02 runtime-bounds03 runtime-bounds04 runtime-bounds05 runtime-linear03 runtime-noreclaim01;do
 mkdir -p "$R/stage/$name"
 cp -a "$B/core/$name/input" "$R/stage/$name/"
 cp -a "$B/core/$name/results" "$R/stage/$name/"
 cp "$B/core/$name/worker.sbatch" "$R/stage/$name/"
done
mkdir -p "$R/stage/binaries" "$R/stage/original-data"
cp "$B/root-integrated01/results/madaros" "$R/stage/binaries/predecessor-madaros"
cp "$B/root-integrated02/results/madaros" "$R/stage/binaries/converged-madaros"
cp "$B/services-tools-recovered/platform-abi-112/libstdlib_platform.so" "$R/stage/binaries/"
cp "$B/core/runtime-bytes03/source/tests/fixtures/fmri/tiny_real_slice_nifti1.nii" "$R/stage/original-data/"
cp "$B/core/runtime-bytes03/source/tests/fixtures/fmri/tiny_real_motion.tsv" "$R/stage/original-data/"
sacct -j 19509,19547,19549,19550,19551,19552,19553,19554,19556,19557 --format=JobID,State,ExitCode,Elapsed,MaxRSS -P > "$R/stage/SACCT.tsv"
cd "$B/core/runtime-bytes03/source"
sha256sum -c "$B/core/runtime-bytes03/input/STDLIB.sha256" > "$R/results/QUALIFIED_STDLIB.final.verify"
cd "$R/stage"
find . -type f -print0|sort -z|xargs -0 sha256sum > "$R/results/BUNDLE_FILES.sha256"
cp "$R/results/BUNDLE_FILES.sha256" "$R/stage/BUNDLE_FILES.sha256"
cp "$R/worker.sbatch" "$R/stage/CUSTODY_WORKER.sbatch"
tar -czf "$D/runtime-bytes-source-raw-receipts-v1.tar.gz" .
sha256sum "$D/runtime-bytes-source-raw-receipts-v1.tar.gz" > "$R/results/ARCHIVE.sha256"
mkdir "$R/readback"
tar -xzf "$D/runtime-bytes-source-raw-receipts-v1.tar.gz" -C "$R/readback"
cd "$R/readback"
sha256sum -c BUNDLE_FILES.sha256 > "$R/results/READBACK.verify"
sha256sum -c "$R/results/ARCHIVE.sha256" > "$R/results/ARCHIVE.final.verify"
cp "$R/results/ARCHIVE.sha256" "$D/"
echo RUNTIME_BYTES_SOURCE_RAW_CUSTODY_ONLY_OK > "$R/results/STATUS"
