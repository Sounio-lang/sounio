#!/usr/bin/env bash
set -euo pipefail
export PATH=/tmp/sounio-stable-api-bin:$PATH
commit=${1:?frozen source commit required}
[[ "$commit" =~ ^[a-f0-9]{40}$ ]]
packet=/home/devsounio/sounio-artifact-archive/prepared/bands-current-source-typecheck-20261020-v3
repo=/workspace/sounio-sprint-stats-viz-main7-bands-20261020
kubectl -n beagle exec sounio-workspace-control-0 -c workspace-ssh -- runuser -u openvscode-server -- git -C "$repo" diff --exit-code ea8c720823797638ec15887b4e0bafd29987cb0a "$commit" -- bin scripts self-hosted stdlib
test ! -e "$packet"
mkdir -p "$packet"
kubectl -n beagle exec sounio-workspace-control-0 -c workspace-ssh -- runuser -u openvscode-server -- git -C "$repo" archive "$commit" bin/souc bin/madaros bin/souc-linux-x86_64 bin/madaros-linux-x86_64.gz bin/madaros-linux-x86_64.sha256 scripts self-hosted stdlib examples/viz_physchem_demo examples/viz_lab examples/viz_lab_window examples/pbpk_dashboard examples/viz_sampled_uncertainty.sio examples/viz_hello examples/viz_html_export examples/viz_workbench examples/viz_nb_dispersion_diagnostic.sio | tar -xf - -C "$packet"
cp /tmp/bands-current-source-typecheck-dispatch-20261020-v3.sh "$packet/scripts/run_sio_test_suite.sh"
chmod +x "$packet/scripts/run_sio_test_suite.sh"
mkdir -p "$packet/reused-compiler"
cp /home/devsounio/sounio-artifact-archive/compiler-custody/bands-15772/{madaros-band-current.gz,COMPILER.sha256,ORIGIN.txt} "$packet/reused-compiler/"
cd "$packet"
test -x bin/souc-linux-x86_64
test "$(head -c 2 bin/souc-linux-x86_64)" != '#!'
printf '%s\n' "source_commit=$commit" 'purpose=attested-current-source-compiler-reuse-check-and-figure' 'compiler_origin_job=15772' 'compiler_origin_commit=ea8c720823797638ec15887b4e0bafd29987cb0a' 'compiler_input_trees_identical=bin,scripts,self-hosted,stdlib' 'compiler_tree_bin=8bbe6e3d99e337175127164d9841ffd41ecd004a' 'compiler_tree_scripts=686a86f184a67737f99b7bb614ebf252c21769e6' 'compiler_tree_self_hosted=7acf6997bc5eaf01a155c1051741c69c2acae571' 'compiler_tree_stdlib=d7c64d90ba29225cff8b55bfee271e9f19a25aa6' 'package_delta=scripts/run_sio_test_suite.sh' 'binary_allowlist=canonical_wrappers_primary_bootstrap_committed_madaros_gzip_and_digest' 'corpus_acceptance=false' 'numerical_acceptance=false' > SOURCE.txt
find bin scripts self-hosted stdlib examples reused-compiler -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > MANIFEST.sha256
sha256sum --check --status MANIFEST.sha256
sha256sum SOURCE.txt MANIFEST.sha256 scripts/run_sio_test_suite.sh stdlib/viz/uncertainty.sio stdlib/viz/epiviz.sio stdlib/viz/ir.sio stdlib/viz/viz_canvas.sio stdlib/viz/viz_html.sio stdlib/viz/workbench.sio
df -B1 /var/lib/cephfs-slurm/training
exec /home/devsounio/projects/sounio/sounio-forge submit corpus-smoke --source-path "$packet" --gpu off --cpu-cpus 2 --cpu-memory-mib 16384 --cpu-time 00:20:00 --expected-job-uid 1000 --expected-job-gid 1000 --submit-user sounio --artifact-root /var/lib/cephfs-slurm/training/sounio-foundry/runs --run-id bands-current-source-typecheck-20261020-v3
