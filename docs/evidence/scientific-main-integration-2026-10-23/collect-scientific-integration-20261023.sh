#!/usr/bin/env bash
set -euo pipefail
umask 077
export PATH=/tmp/sounio-stable-api-bin:$PATH
base=/home/devsounio/sounio-artifact-archive/scientific-integration-20261023
out="$base/final"
mkdir -- "$out"
shared=/var/lib/cephfs-slurm/training/sounio-foundry/runs
local_root=/scratch/sounio-scientific-integration-20261023/run2
cp -a "$local_root" "$out/execution"
cp "$base/COMPLETE_SNAPSHOT.sha256" "$base/COMPLETE_SNAPSHOT.bytes" "$out/"
for attempt in scientific-integration-20261023 scientific-integration-20261023-v2; do
 mkdir "$out/$attempt"
 cp -a "$shared/$attempt/logs" "$out/$attempt/"
 for file in summary.json results.tsv run_packet.json runtime-receipt.json submission.json DISPATCH_STATUS.txt RESULT.txt RESULTS.tsv COMPILER.sha256 BUILD_RESOURCE.txt; do
  if [[ -f "$shared/$attempt/$file" ]]; then cp "$shared/$attempt/$file" "$out/$attempt/"; fi
 done
done
kubectl -n slurm-pilot exec deploy/slurm-pilot-login-slinky -- sacct -j 15990,15992 --format=JobID,State,Elapsed,ExitCode,TotalCPU,MaxRSS -P > "$out/SLURM_ACCOUNTING.txt"
cp /tmp/scientific-integration-dispatch-20261023.sh /tmp/scientific-integration-dispatch-20261023-v2.sh /tmp/submit-scientific-integration-20261023.sh /tmp/retry-scientific-integration-20261023.sh "$out/"
(cd "$out" && find . -type f ! -name FILES.sha256 -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > FILES.sha256 && sha256sum --check --status FILES.sha256)
printf 'private_custody=verified_file_hashes\n' > "$out/CUSTODY.txt"
