#!/usr/bin/env bash
set -euo pipefail
export PATH=/tmp/sounio-stable-api-bin:$PATH
packet=/home/devsounio/sounio-artifact-archive/prepared/nb-current-source-typecheck-20261020
repo=/workspace/sounio-sprint-stats-viz-main6-counts-20261020
commit=9285bb2f118983282256035ef1a9df3a13dfb12a
test ! -e "$packet"
mkdir -p "$packet"
kubectl -n beagle exec sounio-workspace-control-0 -c workspace-ssh -- runuser -u openvscode-server -- git -C "$repo" archive "$commit" bin scripts self-hosted stdlib examples/viz_nb_dispersion_diagnostic.sio | tar -xf - -C "$packet"
cp /tmp/nb-current-source-typecheck-dispatch-20261020.sh "$packet/scripts/run_sio_test_suite.sh"
chmod +x "$packet/scripts/run_sio_test_suite.sh"
cd "$packet"
printf '%s\n' "source_commit=$commit" 'purpose=current-source-typecheck-only' 'package_delta=scripts/run_sio_test_suite.sh' 'corpus_acceptance=false' 'numerical_acceptance=false' > SOURCE.txt
find bin scripts self-hosted stdlib examples -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > MANIFEST.sha256
sha256sum --check --status MANIFEST.sha256
sha256sum SOURCE.txt MANIFEST.sha256 scripts/run_sio_test_suite.sh stdlib/stats/regression/nb_dispersion.sio examples/viz_nb_dispersion_diagnostic.sio
df -B1 /var/lib/cephfs-slurm/training
exec /home/devsounio/projects/sounio/sounio-forge submit corpus-smoke --source-path "$packet" --gpu off --cpu-cpus 2 --cpu-memory-mib 16384 --cpu-time 00:15:00 --expected-job-uid 1000 --expected-job-gid 1000 --submit-user sounio --artifact-root /var/lib/cephfs-slurm/training/sounio-foundry/runs --run-id nb-current-source-typecheck-20261020
