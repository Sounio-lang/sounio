#!/bin/bash
#SBATCH --job-name=cs6-t23-u250-arb
#SBATCH --partition=gpu-orangefs
#SBATCH --account=lab
#SBATCH --qos=normal
#SBATCH --nodelist=gpuorangefs-multi-r740-proxmox
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00

set -euo pipefail

[[ $# -eq 5 ]] || { echo "usage: $0 WORKER WORKER_SHA WHEEL WHEEL_SHA OUTPUT" >&2; exit 2; }
worker=$1
worker_sha=$2
wheel=$3
wheel_sha=$4
output=$5
[[ $(sha256sum "$worker" | awk '{print $1}') == "$worker_sha" ]]
[[ $(sha256sum "$wheel" | awk '{print $1}') == "$wheel_sha" ]]
[[ $output == /orangefs/training/cs6-v7b-target23-u250-arb-* ]]

work=$(mktemp -d /tmp/cs6-v7b-target23-u250-arb.XXXXXXXX)
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/deps" "$output"
PIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install --no-index --no-deps \
  --target "$work/deps" "$wheel" > "$output/pip-install.txt" 2>&1
[[ $(PYTHONPATH="$work/deps" python3 -c 'import flint; print(flint.__version__)') == 0.8.0 ]]

run_leaf() {
  local index=$1 leaf=$2 ud=$3 ui=$4 sd=$5 si=$6
  local challenge binding
  challenge=$(printf 'target23-u250-arb-challenge-v1\0%s\0%s' "$SLURM_JOB_ID" "$leaf" | sha256sum | awk '{print $1}')
  binding=$(printf 'target23-u250-arb-binding-v1\0%s\0%s\0%s' "$worker_sha" "$wheel_sha" "$leaf" | sha256sum | awk '{print $1}')
  LEAF_INDEX="$index" LEAF_ID="$leaf" U_DEPTH="$ud" U_INDEX="$ui" \
    S_DEPTH="$sd" S_INDEX="$si" CHALLENGE="$challenge" BINDING="$binding" \
    PYTHONPATH="$work/deps" python3 -B - "$worker" > "$output/leaf-${index}.txt" <<'PY'
import importlib.util
import os
import sys

path = sys.argv[1]
spec = importlib.util.spec_from_file_location("target23_arb_worker", path)
worker = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(worker)
worker.LEAF_ID = os.environ["LEAF_ID"]
worker.U_DEPTH = int(os.environ["U_DEPTH"])
worker.U_INDEX = int(os.environ["U_INDEX"])
worker.S_DEPTH = int(os.environ["S_DEPTH"])
worker.S_INDEX = int(os.environ["S_INDEX"])
sys.argv = [path, os.environ["CHALLENGE"], os.environ["BINDING"]]
worker.main()
PY
}

run_leaf 319 U08-0000000221_S09-0000000325 8 221 9 325
run_leaf 331 U08-0000000223_S09-0000000325 8 223 9 325
run_leaf 329 U08-0000000222_S09-0000000325 8 222 9 325

printf 'SCHEMA=sounio.cs6.v7b-target23-u250-arb-recheck-execution.v1\nSLURM_JOB_ID=%s\nSLURM_NODE=%s\nWORKER_SHA256=%s\nPYTHON_FLINT_WHEEL_SHA256=%s\nARB_RECHECKS=3\nARB_RECHECK_EXECUTION_PASS=true\n' \
  "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST" "$worker_sha" "$wheel_sha" > "$output/execution.txt"
sha256sum "$output"/leaf-*.txt "$output/execution.txt" > "$output/files.sha256"
