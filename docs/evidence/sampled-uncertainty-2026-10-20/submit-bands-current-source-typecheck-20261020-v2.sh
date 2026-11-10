#!/usr/bin/env bash
set -euo pipefail
export PATH=/tmp/sounio-stable-api-bin:$PATH
commit=${1:?frozen source commit required}
[[ "$commit" =~ ^[a-f0-9]{40}$ ]]
packet=/home/devsounio/sounio-artifact-archive/prepared/bands-current-source-typecheck-20261020-v2
repo=/workspace/sounio-sprint-stats-viz-main7-bands-20261020
test ! -e "$packet"
mkdir -p "$packet"
kubectl -n beagle exec sounio-workspace-control-0 -c workspace-ssh -- runuser -u openvscode-server -- git -C "$repo" archive "$commit" bin/souc bin/madaros bin/souc-linux-x86_64 bin/madaros-linux-x86_64.gz bin/madaros-linux-x86_64.sha256 scripts self-hosted stdlib examples/viz_physchem_demo examples/viz_lab examples/viz_lab_window examples/pbpk_dashboard examples/viz_sampled_uncertainty.sio | tar -xf - -C "$packet"
cp /tmp/bands-current-source-typecheck-dispatch-20261020-v2.sh "$packet/scripts/run_sio_test_suite.sh"
chmod +x "$packet/scripts/run_sio_test_suite.sh"
cd "$packet"
test -x bin/souc-linux-x86_64
test "$(head -c 2 bin/souc-linux-x86_64)" != '#!'
printf '%s\n' "source_commit=$commit" 'purpose=current-source-check-and-figure' 'package_delta=scripts/run_sio_test_suite.sh' 'binary_allowlist=canonical_wrappers_primary_bootstrap_committed_madaros_gzip_and_digest' 'corpus_acceptance=false' 'numerical_acceptance=false' > SOURCE.txt
find bin scripts self-hosted stdlib examples -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > MANIFEST.sha256
sha256sum --check --status MANIFEST.sha256
sha256sum SOURCE.txt MANIFEST.sha256 scripts/run_sio_test_suite.sh stdlib/viz/uncertainty.sio stdlib/viz/epiviz.sio stdlib/viz/ir.sio stdlib/viz/viz_canvas.sio stdlib/viz/viz_html.sio stdlib/viz/workbench.sio
df -B1 /var/lib/cephfs-slurm/training
exec /home/devsounio/projects/sounio/sounio-forge submit corpus-smoke --source-path "$packet" --gpu off --cpu-cpus 2 --cpu-memory-mib 16384 --cpu-time 00:20:00 --expected-job-uid 1000 --expected-job-gid 1000 --submit-user sounio --artifact-root /var/lib/cephfs-slurm/training/sounio-foundry/runs --run-id bands-current-source-typecheck-20261020-v2
