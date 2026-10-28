#!/bin/bash
#SBATCH --job-name=stdlib-alias-converged-cli
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/alias-converged-cli01/results/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/alias-converged-cli01/results/slurm-%j.err
set -euo pipefail
R=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/alias-converged-cli01
B=/scratch/sounio-stdlib-wave3-20261027
RAW="$B/root-integrated02/results/madaros"
ORIGINAL="$B/compiler-resumed/receiver-raw-array01/source/tests/fixtures/wave3_receiver_dispatch/aliases_fields.sio"
P=/scratch/sounio-stdlib-inference-scale-20261024/stdlib-next-20261027
T="$P/tooling"
export PATH="$P/tools:$T/root/usr/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$T/root/usr/lib/x86_64-linux-gnu"
export MAGIC="$T/root/usr/lib/file/magic.mgc"
export SOUNIO_STDLIB_PATH="$B/compiler-resumed/receiver-raw-array01/source/stdlib"
ulimit -s524288
ulimit -c0
trap 'r=$?;echo "$r" > "$R/results/WORKER_STATUS";date -u > "$R/results/ENDED_UTC";exit "$r"' EXIT
printf '8abe891a66c71bdbeb784a9200d1ea395d4d5a55f0036a0daa51ad3cf0614932  %s\n0781ca5cfed395df4ff5b13df0314abbc4575c22088dec349e42bdaa4f19a990  %s\n' "$RAW" "$ORIGINAL" > "$R/results/INPUTS.sha256"
sha256sum -c "$R/results/INPUTS.sha256" > "$R/results/INPUTS.before.verify"
timeout 120 "$RAW" check "$ORIGINAL" > "$R/results/original-check.stdout" 2> "$R/results/original-check.stderr"
timeout 120 "$RAW" compile "$ORIGINAL" -o "$R/results/original.elf" > "$R/results/original-compile.stdout" 2> "$R/results/original-compile.stderr"
chmod +x "$R/results/original.elf"
timeout 15 "$R/results/original.elf" > "$R/results/original-native.stdout" 2> "$R/results/original-native.stderr"
perl -0777 -pe 's/if a\.get\(1\)!=18/if a.get(1)!=19/ or die "wrongvalue anchor"' "$ORIGINAL" > "$R/input/wrong-value.sio"
timeout 120 "$RAW" check "$R/input/wrong-value.sio" > "$R/results/wrong-check.stdout" 2> "$R/results/wrong-check.stderr"
timeout 120 "$RAW" compile "$R/input/wrong-value.sio" -o "$R/results/wrong.elf" > "$R/results/wrong-compile.stdout" 2> "$R/results/wrong-compile.stderr"
chmod +x "$R/results/wrong.elf"
status=0
timeout 15 "$R/results/wrong.elf" > "$R/results/wrong-native.stdout" 2> "$R/results/wrong-native.stderr" || status=$?
echo "$status" > "$R/results/WRONG_VALUE_STATUS"
test "$status" = 1
sha256sum "$R/results/original.elf" "$R/results/wrong.elf" > "$R/results/ELFS.sha256"
sha256sum -c "$R/results/INPUTS.sha256" > "$R/results/INPUTS.final.verify"
echo ALIASES_FIELDS_ORIGINAL_CONVERGED_CLI_ONLY_OK
