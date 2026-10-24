#!/usr/bin/env bash
set -euo pipefail
export PATH=/tmp/sounio-stable-api-bin:$PATH
commit=${1:?frozen source commit required}
[[ "$commit" =~ ^[a-f0-9]{40}$ ]]
packet=/home/devsounio/sounio-artifact-archive/prepared/logarithmic-current-source-20261021-v2
repo=/workspace/sounio-sprint-stats-viz-main9-logarithmic-20261021
kubectl -n beagle exec sounio-workspace-control-0 -c workspace-ssh -- runuser -u openvscode-server -- git -C "$repo" diff --exit-code 5708a48a22684f242b3fa5d8d376d6389c456b68 "$commit" -- bin scripts self-hosted stdlib
test ! -e "$packet"
mkdir -p "$packet"
kubectl -n beagle exec sounio-workspace-control-0 -c workspace-ssh -- runuser -u openvscode-server -- git -C "$repo" archive "$commit" bin/souc bin/madaros bin/souc-linux-x86_64 bin/madaros-linux-x86_64.gz bin/madaros-linux-x86_64.sha256 scripts self-hosted stdlib examples/viz_nb_dispersion_diagnostic.sio examples/viz_logarithmic_primitives.sio | tar -xf - -C "$packet"
cp /tmp/logarithmic-current-source-dispatch-20261021-v2.sh "$packet/scripts/run_sio_test_suite.sh"
chmod +x "$packet/scripts/run_sio_test_suite.sh"
mkdir -p "$packet/reused-compiler"
origin=/var/lib/cephfs-slurm/training/sounio-foundry/runs/logarithmic-current-source-20261021/compiler
cp "$origin/madaros-logarithmic-current.gz" "$origin/COMPILER.sha256" "$packet/reused-compiler/"
printf '%s\n' 'origin_job=15823' 'origin_source_commit=5708a48a22684f242b3fa5d8d376d6389c456b68' 'compiler_input_trees_identical=bin,scripts,self-hosted,stdlib' > "$packet/reused-compiler/ORIGIN.txt"
cd "$packet"
test -x bin/souc-linux-x86_64
test "$(head -c 2 bin/souc-linux-x86_64)" != '#!'
printf '%s\n' "source_commit=$commit" 'purpose=attested-current-source-compiler-reuse-check-and-two-demonstrations' 'compiler_origin_job=15823' 'compiler_origin_source_commit=5708a48a22684f242b3fa5d8d376d6389c456b68' 'compiler_input_trees_identical=bin,scripts,self-hosted,stdlib' 'comparative_benchmark=false' 'package_delta=scripts/run_sio_test_suite.sh' 'binary_allowlist=canonical_wrappers_primary_bootstrap_committed_madaros_gzip_and_digest' 'corpus_acceptance=false' 'numerical_acceptance=false' > SOURCE.txt
find bin scripts self-hosted stdlib examples reused-compiler -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > MANIFEST.sha256
sha256sum --check --status MANIFEST.sha256
sha256sum SOURCE.txt MANIFEST.sha256 scripts/run_sio_test_suite.sh stdlib/math/logarithmic.sio stdlib/math/channels.sio stdlib/math/probability.sio stdlib/stats/regression/poisson.sio examples/viz_logarithmic_primitives.sio examples/viz_nb_dispersion_diagnostic.sio
df -B1 /var/lib/cephfs-slurm/training
exec /home/devsounio/projects/sounio/sounio-forge submit corpus-smoke --source-path "$packet" --gpu off --cpu-cpus 2 --cpu-memory-mib 16384 --cpu-time 00:20:00 --expected-job-uid 1000 --expected-job-gid 1000 --submit-user sounio --artifact-root /var/lib/cephfs-slurm/training/sounio-foundry/runs --run-id logarithmic-current-source-20261021-v2
