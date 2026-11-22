#!/usr/bin/env bash
set -euo pipefail
umask 077
export PATH=/tmp/sounio-stable-api-bin:$PATH
commit=${1:?frozen source commit required}
[[ "$commit" =~ ^[a-f0-9]{40}$ ]]
packet=/home/devsounio/sounio-artifact-archive/prepared/scientific-integration-20261023
repo=/workspace/sounio-sprint-stats-viz-main9-logarithmic-20261021
mkdir -- "$packet"
kubectl -n beagle exec sounio-workspace-control-0 -c workspace-ssh -- runuser -u openvscode-server -- git -C "$repo" archive "$commit" bin/souc bin/madaros bin/souc-linux-x86_64 bin/souc-lean-single-x86_64 bin/souc-lean-single-x86_64.SeedReceipt.json bin/madaros-linux-x86_64.gz bin/madaros-linux-x86_64.sha256 scripts self-hosted stdlib examples/viz_nb2_deviance.sio examples/viz_nb_dispersion_diagnostic.sio examples/viz_logarithmic_primitives.sio examples/viz_genomics_entropy.sio examples/viz_sampled_uncertainty.sio | tar -xf - -C "$packet"
cp /tmp/scientific-integration-dispatch-20261023.sh "$packet/scripts/run_sio_test_suite.sh"
chmod +x "$packet/scripts/run_sio_test_suite.sh"
cd "$packet"
printf '%s\n' "source_commit=$commit" 'main_basis=3ef3dbaa6e' 'purpose=current-source-scientific-integration' 'packet_delta=scripts/run_sio_test_suite.sh' 'tests_run=false' 'numerical_concordance=false' > SOURCE.txt
find bin scripts self-hosted stdlib examples -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > MANIFEST.sha256
sha256sum --check --status MANIFEST.sha256
# Heavy outputs use a private ext4 directory visible in the Slurm worker.
kubectl -n slurm-pilot exec slurm-pilot-worker-cpuops-4pvxd -- install -d -m 700 -o 1000 -g 1000 /scratch/sounio-scientific-integration-20261023
exec /home/devsounio/projects/sounio/sounio-forge submit corpus-smoke --source-path "$packet" --gpu off --cpu-cpus 2 --cpu-memory-mib 16384 --cpu-time 00:25:00 --expected-job-uid 1000 --expected-job-gid 1000 --submit-user sounio --artifact-root /var/lib/cephfs-slurm/training/sounio-foundry/runs --run-id scientific-integration-20261023
