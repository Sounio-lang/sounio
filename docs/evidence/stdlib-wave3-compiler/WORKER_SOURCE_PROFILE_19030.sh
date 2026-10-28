#!/bin/bash
#SBATCH --job-name=stdlib-worker-frontend-v1
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/worker-source-v2/results/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/worker-source-v2/results/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed
R=$B/worker-source-v2
D=$R/results
S=$R/source
trap 'r=$?;echo "$r" > "$D/WORKER_STATUS";exit "$r"' EXIT
ulimit -s 524288
ulimit -c 0
cp -a "$B/worker-front-v1/source" "$S"
tar -xzf "$B/worker-source-v2-overlay.tar.gz" -C "$S"
cd "$S"
sha256sum -c WORKER_SOURCE_INPUTS.sha256 > "$D/FEATURE_INPUTS.verify"
find self-hosted tests/fixtures/wave4_worker_capability -type f -print0|sort -z|xargs -0 sha256sum > "$D/SOURCE.sha256"
sha256sum bin/souc-lean-single-x86_64 > "$D/SEED.sha256"
export SOUNIO_STDLIB_PATH="$S/self-hosted"
timeout 900 bin/souc-lean-single-x86_64 tests/fixtures/wave3_generic_enum/borrowed_enum_checker_driver.sio "$D/checker.elf" > "$D/checker-compile.stdout" 2>&1 &
checker_pid=$!
timeout 300 bin/souc-lean-single-x86_64 tests/fixtures/wave4_worker_capability/type_component.sio "$D/types.elf" > "$D/types-compile.stdout" 2>&1 &
types_pid=$!
wait "$checker_pid"
wait "$types_pid"
if grep -Eq '^error:|^error\\[|would have dropped|cannot compile an assignment' "$D/checker-compile.stdout" "$D/types-compile.stdout";then echo EXPLICIT_BOOTSTRAP_DIAGNOSTIC_REFUSED >&2;exit 1;fi
chmod +x "$D/checker.elf" "$D/types.elf"
timeout 30 "$D/types.elf" > "$D/types-native.stdout" 2>&1
grep -Fxq WORKER_OPAQUE_TYPE_COMPONENT_OK "$D/types-native.stdout"
printf 'case\texit_status\n' > "$D/WORKER_VERDICTS.tsv"
for n in good_i64 good_f64 good_transitive good_stack_refs good_recursive wrong_root_abi capture local_alias forged_cast raw_extract declared_alloc wrong_cap_abi forged_struct transitive_global transitive_indirect transitive_aggregate global_reference reference_cast;do
 rc=0;timeout 90 "$D/checker.elf" "tests/fixtures/wave4_worker_capability/$n.sio" > "$D/$n.stdout" 2>&1 ||rc=$?
 printf '%s\t%s\n' "$n" "$rc" >> "$D/WORKER_VERDICTS.tsv"
done
g++ -std=c++23 -O2 -Wall -Wextra -Werror -pedantic tests/fixtures/wave4_worker_capability/reference.cpp -o "$D/worker-reference"
"$D/worker-reference" "$D" > "$D/worker-reference.stdout"
printf 'case\texit_status\n' > "$D/BEFORE_VERDICTS.tsv"
for n in good_transitive good_stack_refs good_recursive transitive_global transitive_indirect transitive_aggregate global_reference reference_cast;do
 rc=0;timeout 90 "$B/worker-front-v1/results/checker.elf" "$S/tests/fixtures/wave4_worker_capability/$n.sio" > "$D/$n-before.stdout" 2>&1 ||rc=$?
 printf '%s\t%s\n' "$n" "$rc" >> "$D/BEFORE_VERDICTS.tsv"
done
mkdir -p "$R/controls"
tar -xzf "$B/borrowed-enum-v5-controls.tar.gz" -C "$R/controls"
cd "$R/controls"
sha256sum -c INPUTS.sha256 > "$D/CONTROLS.verify"
printf 'case\texit_status\n' > "$D/VERDICTS.tsv"
for input in returned_loan_*.sio borrowed_enum_*.sio;do
 [[ "$input" != *_driver.sio ]] ||continue
 n=$(basename "$input" .sio)
 rc=0;timeout 120 "$D/checker.elf" "$input" > "$D/$n.stdout" 2>&1 ||rc=$?
 printf '%s\t%s\n' "$n" "$rc" >> "$D/VERDICTS.tsv"
done
reference_source="$B/tuple-pattern-v3/source/tests/reference/wave3_returned_loan_matrix_reference.cpp"
sha256sum "$reference_source" > "$D/REFERENCE_SOURCE.sha256"
g++ -std=c++23 -O2 -Wall -Wextra -Werror -pedantic "$reference_source" -o "$D/reference"
"$D/reference" --kernel --verify "$D" > "$D/reference.stdout"
"$D/reference" --kernel --self-test "$D" > "$D/reference-wrong-output.stdout"
sha256sum -c INPUTS.sha256 > "$D/CONTROLS.final.verify"
cd "$S"
sha256sum -c WORKER_SOURCE_INPUTS.sha256 > "$D/FEATURE_INPUTS.final.verify"
sha256sum -c "$D/SOURCE.sha256" > "$D/SOURCE.final.verify"
sha256sum -c "$D/SEED.sha256" > "$D/SEED.final.verify"
sha256sum -c "$D/REFERENCE_SOURCE.sha256" > "$D/REFERENCE_SOURCE.final.verify"
sha256sum "$D/"*.elf > "$D/ELFS.sha256"
echo WORKER_SOURCE_PROFILE_AND_LOAN_REGRESSION_OK_NATIVE_CERTIFICATE_PENDING
