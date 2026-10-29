#!/bin/bash
#SBATCH --job-name=loop-checkpoint-component
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/bidi-production05/results/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/bidi-production05/results/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
R="$B/compiler-resumed/bidi-production05"
S="$R/source"
P=/scratch/sounio-stdlib-inference-scale-20261024/stdlib-next-20261027
export PATH="$P/tools:$P/tooling/root/usr/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$P/tooling/root/usr/lib/x86_64-linux-gnu"
export SOUNIO_STDLIB_PATH="$S/self-hosted"
RAW="$B/root-integrated02/results/madaros"
ulimit -s524288;ulimit -c0
trap 'r=$?;echo "$r" > "$R/results/WORKER_STATUS";date -u > "$R/results/ENDED_UTC";exit "$r"' EXIT
mkdir -p "$R/results/tmp"
export TMPDIR="$R/results/tmp"
cd "$S"
sha256sum scripts/stdlib/runtime_fixtures.mjs scripts/stdlib/stdlib_e2e_diagnostics.mjs scripts/stdlib/run_stdlib_e2e.sh scripts/stdlib/test_runtime_fixture_artifacts.mjs tests/fixtures/wave4_bidi_runtime/official-corpus.json > "$R/results/INPUTS.sha256"
node --check scripts/stdlib/runtime_fixtures.mjs > "$R/results/runtime-syntax.stdout" 2> "$R/results/runtime-syntax.stderr"
node --check scripts/stdlib/stdlib_e2e_diagnostics.mjs > "$R/results/diagnostics-syntax.stdout" 2> "$R/results/diagnostics-syntax.stderr"
node scripts/stdlib/test_stdlib_e2e_diagnostics.mjs > "$R/results/diagnostics-controls.stdout" 2> "$R/results/diagnostics-controls.stderr"
node scripts/stdlib/test_runtime_fixture_artifacts.mjs > "$R/results/artifact-controls.stdout" 2> "$R/results/artifact-controls.stderr"
sha256sum -c "$R/results/INPUTS.sha256" > "$R/results/INPUTS.final.verify"
echo 0 > "$R/results/STATUS"
