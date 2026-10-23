#!/usr/bin/env bash
set -euo pipefail
umask 077
export PATH=/tmp/sounio-stable-api-bin:$PATH
run_root=/scratch/sounio-nb2-residual-bars-20261023/run1
custody=/home/devsounio/sounio-artifact-archive/nb2-residual-bars-20261023
mkdir -- "$custody"
# Preserve compressed executables, logs and receipts; raw originals remain scratch.
tar -C "$run_root" --exclude='*.elf' -cf - artifacts logs | tar -C "$custody" -xf -
cp "$run_root/"{SOURCE.txt,SOURCE_FILES.sha256,SUBMISSION.txt,SNAPSHOT.sha256,source.tar.gz,job.sh,dispatch.sh} "$custody/"
cmp --silent "$run_root/source.tar.gz" "$custody/source.tar.gz"
cd "$run_root/source"
find bin scripts self-hosted stdlib examples -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "$custody/SOURCE_FILES.after.sha256"
sha256sum --check --status "$run_root/SOURCE_FILES.sha256"
if [[ -f bin/madaros-linux-x86_64 ]]; then
 sha256sum bin/madaros-linux-x86_64 bin/madaros-linux-x86_64.gz > "$custody/MATERIALIZED_PREBUILT.sha256"
 gzip -dc bin/madaros-linux-x86_64.gz | sha256sum > "$custody/MATERIALIZED_PREBUILT.readback.sha256"
 actual=$(sha256sum bin/madaros-linux-x86_64 | cut -d ' ' -f1)
 readback=$(gzip -dc bin/madaros-linux-x86_64.gz | sha256sum | cut -d ' ' -f1)
 [[ "$actual" == "$readback" ]]
fi
cd "$custody"
for archive in artifacts/madaros.elf.gz artifacts/native/*/program.elf.gz; do
 raw=${archive%.gz}
 actual=$(sha256sum "$run_root/$raw" | cut -d ' ' -f1)
 readback=$(gzip -dc "$archive" | sha256sum | cut -d ' ' -f1)
 [[ "$actual" == "$readback" ]]
 printf '%s  %s\n' "$readback" "$raw" >> EXECUTABLE_READBACK.sha256
done
kubectl -n slurm-pilot exec deploy/slurm-pilot-login-slinky -- sacct -j 16017 --format=JobID,State,Elapsed,ExitCode,TotalCPU,MaxRSS -P > SLURM_ACCOUNTING.txt
printf 'endpoint_check=manifest_listed_files_only\ncomplete_tree_immutability=false\nraw_override=fresh_source_built_compiler\ntraining_quota_paths=not_used_for_source_or_job_outputs\nscheduler_storage=separate_Ceph_PVC\nraw_elf_copies_in_custody=none\ncompressed_elf_readbacks=matched_scratch_originals\n' > ENDPOINT_AND_STORAGE_SCOPE.txt
find . -type f ! -name CUSTODY_FILES.sha256 -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > CUSTODY_FILES.sha256
sha256sum --check --status CUSTODY_FILES.sha256
