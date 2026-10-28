#!/bin/bash
#SBATCH --job-name=stdlib-bidi-candidate-durable
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/core/bidi-icu-complete-archive01/results/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/core/bidi-icu-complete-archive01/results/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
R=/scratch/sounio-stdlib-wave3-20261027/core/bidi-icu-complete-archive01
B=/scratch/sounio-stdlib-wave3-20261027/core
D=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/bidi-icu-complete01
trap 'r=$?;echo "$r" > "$R/results/WORKER_STATUS";date -u > "$R/results/ENDED_UTC";exit "$r"' EXIT
test ! -e "$D"
mkdir -p "$D"
test "$(cat "$B/bidi-icu-complete03/results/WORKER_STATUS")" = 0
test "$(cat "$B/bidi-icu-complete03/candidate/STATUS")" = 0
sha256sum -c "$B/bidi-icu-complete03/results/SOURCE.sha256" > "$R/results/SOURCE.final.verify"
for kind in INPUTS PRISTINE BEFORE SOURCE_TREE CANDIDATE EXECUTABLES;do
 sha256sum -c "$B/bidi-icu-complete03/candidate/$kind.sha256" > "$R/results/$kind.final.verify"
done
sacct -j19235,19247,19253,19261 -X --format=JobID,JobName,State,ExitCode,Elapsed -P > "$R/results/JOBS.tsv"
tar -C "$B" --exclude=bidi-icu-complete03/candidate/build -czf "$D/final-sdk-source-receipts-19261.tar.gz" bidi-icu-complete03
tar -C "$B" --exclude=bidi-icu-complete01/candidate/build --exclude=bidi-icu-complete02/candidate/build -czf "$D/before-failed-history-19235-19247-19253.tar.gz" bidi-icu-collect01 bidi-icu-complete01 bidi-icu-complete02
cp "$R/worker.sbatch" "$R/results/JOBS.tsv" "$D/"
cp "$B/bidi-icu-complete03/candidate/RESULTS.tsv" "$D/QUALIFIED_RESULTS.tsv"
cp "$B/bidi-icu-complete03/candidate/override-current-type.patch" "$D/"
cp "$B/bidi-icu-complete03/candidate/PATCH_CUSTODY.json" "$B/bidi-icu-complete03/candidate/HEADER_CUSTODY.json" "$D/"
printf '%s\n' 'Candidate full861948 concordance only. Pristine strict still FAIL; no primary oracle promotion, Sounio production edit, upstream publication, or whole-stdlib acceptance.' > "$D/SCOPE"
sha256sum "$D/final-sdk-source-receipts-19261.tar.gz" "$D/before-failed-history-19235-19247-19253.tar.gz" "$D/worker.sbatch" "$D/JOBS.tsv" "$D/QUALIFIED_RESULTS.tsv" "$D/override-current-type.patch" "$D/PATCH_CUSTODY.json" "$D/HEADER_CUSTODY.json" "$D/SCOPE" > "$D/ARCHIVES.sha256"
sha256sum -c "$D/ARCHIVES.sha256" > "$D/ARCHIVES.final.verify"
tar -tzf "$D/final-sdk-source-receipts-19261.tar.gz" > "$R/results/final.inventory"
tar -tzf "$D/before-failed-history-19235-19247-19253.tar.gz" > "$R/results/history.inventory"
for item in input/scripts/stdlib/wave4_bidi_icu_complete_candidate.sh input/tests/reference/wave4_bidi_icu_collect.cpp input/tests/reference/wave4_bidi_sdk_controls.cpp candidate/source/icu/source/common/ubidi.cpp candidate/source/icu/source/common/ubidiimp.h candidate/install/lib/libicuuc.so.79.1 candidate/install/lib/libicui18n.so.79.1 candidate/install/lib/libicudata.so.79.1;do
 tar -xOzf "$D/final-sdk-source-receipts-19261.tar.gz" "bidi-icu-complete03/$item" | sha256sum >> "$R/results/ARCHIVED_CONTENT.sha256"
 sha256sum "$B/bidi-icu-complete03/$item" >> "$R/results/LIVE_CONTENT.sha256"
done
cut -d' ' -f1 "$R/results/ARCHIVED_CONTENT.sha256" > "$R/results/ARCHIVED_DIGESTS"
cut -d' ' -f1 "$R/results/LIVE_CONTENT.sha256" > "$R/results/LIVE_DIGESTS"
cmp "$R/results/ARCHIVED_DIGESTS" "$R/results/LIVE_DIGESTS"
cp "$R/results/ARCHIVED_CONTENT.sha256" "$R/results/LIVE_CONTENT.sha256" "$D/"
echo BIDI_CANDIDATE_DURABLE_OK
