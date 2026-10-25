#!/bin/bash
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:25:00
#SBATCH --export=NIL
set -euo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
ulimit -s 524288
base=/scratch/sounio-stdlib-inference-scale-20261024/SIO167
out="$base/phase13"
export MADAROS_RAW_BIN=/scratch/sounio-stdlib-inference-scale-20261024/compiler_ci_c21/madaros.elf.ready
printf 'a292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3  %s\n' "$MADAROS_RAW_BIN" | sha256sum -c -
cd "$base/source11"
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
fixture="$base/source13/tests/run-pass/stdlib_noncentral_t_checked.sio"
SOUNIO_SOUC_ENGINE=lean_single bin/souc check "$fixture" > "$out/before.seed.stdout" 2> "$out/before.seed.stderr"
if grep -Eq '^error:|error\[E' "$out/before.seed.stdout" "$out/before.seed.stderr";then exit 1;fi
/usr/bin/time -v bin/souc compile "$fixture" -o "$out/before.elf" > "$out/before.compile.stdout" 2> "$out/before.compile.stderr"
if grep -Eq '^error:|error\[E' "$out/before.compile.stdout" "$out/before.compile.stderr";then exit 1;fi
chmod +x "$out/before.elf";sha256sum "$out/before.elf" stdlib/stats/checked_power.sio stdlib/stats/noncentral_t.sio "$fixture" > "$out/before.inputs.sha256"
set +e
/usr/bin/time -v timeout 120 "$out/before.elf" > "$out/before.native.stdout" 2> "$out/before.native.stderr"
code=$?;echo "$code" > "$out/before.native.exit"
set -e
[[ "$code" == 1 ]]
cd "$base/source13"
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
set +e
bash scripts/stdlib/noncentral_t_power_gate.sh "$out" > "$out/gate.stdout" 2> "$out/gate.stderr"
code=$?;echo "$code" > "$out/gate.exit";exit "$code"
