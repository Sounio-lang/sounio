#!/usr/bin/env bash
set -euo pipefail
export PATH=/tmp/sounio-stable-api-bin:$PATH
commit=${1:?frozen source commit required}
[[ "$commit" =~ ^[a-f0-9]{40}$ ]]
packet=/home/devsounio/sounio-artifact-archive/prepared/nb2-native-profile-20261021
repo=/workspace/sounio-sprint-stats-viz-main8-nb2-20261021
test ! -e "$packet"
mkdir -p "$packet"
kubectl -n beagle exec sounio-workspace-control-0 -c workspace-ssh -- runuser -u openvscode-server -- git -C "$repo" archive "$commit" bin/souc bin/madaros bin/souc-linux-x86_64 bin/madaros-linux-x86_64.gz bin/madaros-linux-x86_64.sha256 scripts self-hosted stdlib examples/viz_nb_dispersion_diagnostic.sio | tar -xf - -C "$packet"
cp /tmp/nb2-native-profile-dispatch-20261021.sh "$packet/scripts/run_sio_test_suite.sh"
chmod +x "$packet/scripts/run_sio_test_suite.sh"
mkdir -p "$packet/native-input"
origin=/var/lib/cephfs-slurm/training/sounio-foundry/runs/nb2-current-source-20261021/figure
test "$(cat "$origin/EXECUTION.txt")" = 'exit_code=0'
cp "$origin/nb2-diagnostic.elf.gz" "$origin/PROGRAM_IDENTITIES.sha256" "$packet/native-input/"
cd "$packet"
test -x bin/souc-linux-x86_64
test "$(head -c 2 bin/souc-linux-x86_64)" != '#!'
printf '%s\n' "source_commit=$commit" 'purpose=exact-native-elf-resource-observation' 'native_origin_job=15783' 'compiler_use=orientation-only-no-program-compilation' 'comparative_benchmark=false' 'package_delta=scripts/run_sio_test_suite.sh' 'binary_allowlist=canonical_wrappers_primary_bootstrap_committed_madaros_gzip_and_digest' 'corpus_acceptance=false' 'numerical_acceptance=false' > SOURCE.txt
find bin scripts self-hosted stdlib examples native-input -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > MANIFEST.sha256
sha256sum --check --status MANIFEST.sha256
sha256sum SOURCE.txt MANIFEST.sha256 scripts/run_sio_test_suite.sh stdlib/stats/regression/nb2_diagnostics.sio examples/viz_nb_dispersion_diagnostic.sio
df -B1 /var/lib/cephfs-slurm/training
exec /home/devsounio/projects/sounio/sounio-forge submit corpus-smoke --source-path "$packet" --gpu off --cpu-cpus 2 --cpu-memory-mib 16384 --cpu-time 00:20:00 --expected-job-uid 1000 --expected-job-gid 1000 --submit-user sounio --artifact-root /var/lib/cephfs-slurm/training/sounio-foundry/runs --run-id nb2-native-profile-20261021
