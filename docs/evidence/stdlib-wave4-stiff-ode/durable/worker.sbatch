#!/bin/bash
#SBATCH --job-name=stdlib-stiff-custody
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/core/stiff-custody01/results/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/core/stiff-custody01/results/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
R="$B/core/stiff-custody01"
D=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/stiff-ode01
mkdir -p "$R/stage" "$D"
trap 'r=$?;echo "$r" > "$R/results/WORKER_STATUS";date -u > "$R/results/ENDED_UTC";exit "$r"' EXIT
test "$(cat "$B/core/stiff-contract04/results/WORKER_STATUS")" = 0
grep -Fx STIFF_ODE_FOCUSED_REAL_MC_CPP23_OK "$B/core/stiff-contract04/results/STATUS"
for name in stiff-reference01 stiff-reference02 stiff-native01 stiff-contract03 stiff-contract04;do
 mkdir -p "$R/stage/$name"
 cp -a "$B/core/$name/input" "$R/stage/$name/"
 cp -a "$B/core/$name/results" "$R/stage/$name/"
 cp "$B/core/$name/worker.sbatch" "$R/stage/$name/"
done
mkdir -p "$R/stage/binaries" "$R/stage/root-counterexample"
cp "$B/root-integrated02/results/madaros" "$R/stage/binaries/madaros"
cp -a "$B/stiff-contract02/." "$R/stage/root-counterexample/"
sacct -j 19546,19560,19561,19562,19564,19566 --format=JobID,State,ExitCode,Elapsed,MaxRSS -P > "$R/stage/SACCT.tsv"
cd "$B/core/stiff-contract04/source"
sha256sum -c "$B/core/stiff-contract04/input/STDLIB.sha256" > "$R/results/QUALIFIED_STDLIB.final.verify"
cd "$R/stage"
find . -type f -print0|sort -z|xargs -0 sha256sum > "$R/results/BUNDLE_FILES.sha256"
cp "$R/results/BUNDLE_FILES.sha256" "$R/stage/BUNDLE_FILES.sha256"
cp "$R/worker.sbatch" "$R/stage/CUSTODY_WORKER.sbatch"
tar -czf "$D/stiff-ode-real-mc-source-raw-receipts-v1.tar.gz" .
sha256sum "$D/stiff-ode-real-mc-source-raw-receipts-v1.tar.gz" > "$R/results/ARCHIVE.sha256"
mkdir "$R/readback"
tar -xzf "$D/stiff-ode-real-mc-source-raw-receipts-v1.tar.gz" -C "$R/readback"
cd "$R/readback"
sha256sum -c BUNDLE_FILES.sha256 > "$R/results/READBACK.verify"
sha256sum -c "$R/results/ARCHIVE.sha256" > "$R/results/ARCHIVE.final.verify"
cp "$R/results/ARCHIVE.sha256" "$D/"
echo STIFF_ODE_SOURCE_RAW_CUSTODY_ONLY_OK > "$R/results/STATUS"
