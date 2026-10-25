#!/bin/bash
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=2048M
#SBATCH --time=00:30:00
#SBATCH --export=NIL
set -euo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
ulimit -s 524288
cd /scratch/sounio-stdlib-inference-scale-20261024/SIO169/source14
OUT=/scratch/sounio-stdlib-inference-scale-20261024/SIO169/memory14
sha256sum -c SIO169_PHASE14.sha256 > "$OUT/packet.verify.stdout"
export MADAROS_RAW_BIN=/scratch/sounio-stdlib-inference-scale-20261024/compiler_ci_c21/madaros.elf.ready
printf 'a292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3  %s\n' "$MADAROS_RAW_BIN" | sha256sum -c -
printf '8956946def36004d11ec4eca0d20c75c1e7d3d0b17ee0f0a1fdda9cd6ec0e4cc  bin/souc-lean-single-x86_64\n' | sha256sum -c -
printf 'ccfc3110246b05607bebac716fb3bf9b86af957dc1246bf4a117b3563017e896  ../source13/stdlib/stats/checked_survival.sio\n' | sha256sum -c -
set +e
bash scripts/stdlib/survival_memory_gate.sh ../source13 "$OUT/results" > "$OUT/gate.stdout" 2> "$OUT/gate.stderr"
code=$?;echo "$code" > "$OUT/gate.exit"
exit "$code"
