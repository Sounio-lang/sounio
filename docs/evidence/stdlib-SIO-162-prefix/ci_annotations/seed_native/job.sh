#!/bin/bash
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:15:00
#SBATCH --export=NIL
set -euo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
ulimit -s 524288
cd /scratch/sounio-stdlib-inference-scale-20261024/SIO162/source_annotations
out=../annotations_seed/retry
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
export SOUNIO_SOUC_ENGINE=lean_single
seed="$PWD/bin/souc-lean-single-x86_64"
printf '8956946def36004d11ec4eca0d20c75c1e7d3d0b17ee0f0a1fdda9cd6ec0e4cc  %s\n' "$seed" | sha256sum -c -
sha256sum "$seed" > "$out/compiler.sha256"
sha256sum tests/run-pass/stdlib_beta_uniform_baseline.sio tests/run-pass/stdlib_checked_distributions.sio stdlib/stats/checked_distributions.sio stdlib/stats/distributions.sio stdlib/special/checked_gamma_beta.sio > "$out/inputs.sha256"
for name in stdlib_beta_uniform_baseline stdlib_checked_distributions;do
 bin/souc check "tests/run-pass/$name.sio" > "$out/$name.check.stdout" 2> "$out/$name.check.stderr"
 if grep -Eq '^error:|error\[E' "$out/$name.check.stdout" "$out/$name.check.stderr";then exit 1;fi
 /usr/bin/time -v bin/souc compile "tests/run-pass/$name.sio" -o "$out/$name.elf" > "$out/$name.compile.stdout" 2> "$out/$name.compile.stderr"
 if grep -Eq '^error:|error\[E' "$out/$name.compile.stdout" "$out/$name.compile.stderr";then exit 1;fi
 chmod +x "$out/$name.elf";sha256sum "$out/$name.elf" > "$out/$name.elf.sha256"
 /usr/bin/time -v timeout 180 "$out/$name.elf" > "$out/$name.native.stdout" 2> "$out/$name.native.stderr"
done
grep -qx 'BETA_UNIFORM_BITS 4607182418800017408' "$out/stdlib_beta_uniform_baseline.native.stdout"
grep -qx CHECKED_DISTRIBUTIONS_SMOKE_PASS "$out/stdlib_checked_distributions.native.stdout"
echo NEWLY_SELECTED_SEED_NATIVE_PASS > "$out/gate.result"
