#!/bin/bash
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:05:00
#SBATCH --export=NIL
set -euo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
ulimit -s 524288
cd /scratch/sounio-stdlib-inference-scale-20261024/SIO167/source11
out=../legacy_module_c21
export MADAROS_RAW_BIN=/scratch/sounio-stdlib-inference-scale-20261024/compiler_ci_c21/madaros.elf.ready
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
sha256sum "$MADAROS_RAW_BIN" stdlib/stats/power.sio > "$out/inputs.sha256"
SOUNIO_SOUC_ENGINE=lean_single bin/souc check stdlib/stats/power.sio > "$out/seed.stdout" 2> "$out/seed.stderr"
if grep -Eq '^error:|error\[E' "$out/seed.stdout" "$out/seed.stderr";then exit 1;fi
/usr/bin/time -v bin/souc compile stdlib/stats/power.sio -o "$out/module.elf" > "$out/compile.stdout" 2> "$out/compile.stderr"
if grep -Eq '^error:|error\[E' "$out/compile.stdout" "$out/compile.stderr";then exit 1;fi
chmod +x "$out/module.elf";sha256sum "$out/module.elf" > "$out/module.elf.sha256"
/usr/bin/time -v timeout 120 "$out/module.elf" > "$out/native.stdout" 2> "$out/native.stderr"
grep -qx 'ALL PASS' "$out/native.stdout"
echo LEGACY_POWER_MODULE_PASS > "$out/gate.result"
