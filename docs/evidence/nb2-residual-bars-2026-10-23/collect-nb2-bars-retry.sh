#!/usr/bin/env bash
set -euo pipefail
umask 077
export PATH=/tmp/sounio-stable-api-bin:$PATH
run_root=/scratch/sounio-nb2-residual-bars-20261023/run2
custody=/home/devsounio/sounio-artifact-archive/nb2-residual-bars-20261023
mkdir "$custody/retry"
tar -C "$run_root" --exclude='*.elf' -cf - . | tar -C "$custody/retry" -xf -
for archive in "$custody/retry/artifacts/native/"*/program.elf.gz; do
 name=$(basename "$(dirname "$archive")")
 actual=$(sha256sum "$run_root/artifacts/native/$name/program.elf" | cut -d ' ' -f1)
 readback=$(gzip -dc "$archive" | sha256sum | cut -d ' ' -f1)
 [[ "$actual" == "$readback" ]]
 printf '%s  %s\n' "$readback" "$name/program.elf" >> "$custody/retry/EXECUTABLE_READBACK.sha256"
done
kubectl -n slurm-pilot exec deploy/slurm-pilot-login-slinky -- sacct -j 16017,16019 --format=JobID,State,Elapsed,ExitCode,TotalCPU,MaxRSS -P > "$custody/SLURM_ACCOUNTING.txt"
cp "$custody/CUSTODY_FILES.sha256" "$custody/CUSTODY_INITIAL.sha256"
(cd "$custody" && find . -type f ! -name CUSTODY_FILES.sha256 -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > CUSTODY_FILES.sha256 && sha256sum --check --status CUSTODY_FILES.sha256)
