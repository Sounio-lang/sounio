#!/usr/bin/env bash
# Same compiler/source/options/core; only canonical matrix-read implementation differs.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
out=${1:?fresh absolute output required}
baseline=${2:?frozen old dynamic_factorize.sio required}
[[ "$out" == /* && ! -e "$out" && -f "$baseline" && -x ${MADAROS_RAW_BIN:-} ]]
mkdir -p "$out/before-stdlib"
trap 'rc=$?; echo "$rc" > "$out/STATUS"' EXIT
cd "$root"
export SOUNIO_SOUC_ENGINE=madaros SOUNIO_MADAROS_NOCACHE=1
ulimit -s 524288
cp -a stdlib/. "$out/before-stdlib/"
cp "$baseline" "$out/before-stdlib/linalg/dynamic_factorize.sio"
sha256sum "$MADAROS_RAW_BIN" "$baseline" bin/souc bin/madaros tests/stdlib/bdf/compare_factor_memory.sh stdlib/linalg/dynamic_factorize.sio tests/stdlib/bdf/scale128.sio > "$out/INPUTS.sha256"
find stdlib -type f -print0 | sort -z | xargs -0 sha256sum >> "$out/INPUTS.sha256"
for variant in before after;do
 lib="$root/stdlib";[[ $variant == after ]] || lib="$out/before-stdlib"
 SOUNIO_STDLIB_PATH="$lib" bin/souc compile tests/stdlib/bdf/scale128.sio -o "$out/$variant.elf" > "$out/$variant.compile.stdout" 2> "$out/$variant.compile.stderr"
 chmod +x "$out/$variant.elf"
done
sha256sum "$out"/*.elf > "$out/ELFS.sha256"
# Slurm CPU_IDs can be abstract topology ids inside this container. Select a
# Linux CPU from the effective inherited affinity, and retain both records.
if [[ -n ${SLURM_JOB_ID:-} ]];then scontrol show job -d "$SLURM_JOB_ID" > "$out/SLURM_JOB.txt";fi
taskset -pc $$ > "$out/INHERITED_AFFINITY.txt"
cpu=$(sed 's/.*: //;s/[-,].*//' "$out/INHERITED_AFFINITY.txt")
printf 'cpu=%s\n' "$cpu" > "$out/AFFINITY.txt"
for pair in 1 2 3;do
 order='before after';[[ $pair != 2 ]] || order='after before'
 for variant in $order;do
  /usr/bin/time -f 'elapsed_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kib=%M\nexit_code=%x' -o "$out/pair$pair-$variant.resource" timeout 180 taskset -c "$cpu" "$out/$variant.elf" > "$out/pair$pair-$variant.stdout" 2> "$out/pair$pair-$variant.stderr"
 done
 cmp "$out/pair$pair-before.stdout" "$out/pair$pair-after.stdout"
done
sha256sum -c "$out/INPUTS.sha256" > "$out/INPUTS.verify"
echo BDF_FACTOR_MEMORY_COMPARE_PASS
