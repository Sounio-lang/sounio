#!/usr/bin/env bash
# Run only inside an allocated Linux CPU Slurm job. No performance claim.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
out=${1:?fresh absolute result directory}
[[ "$out" == /* && -x ${MADAROS_RAW_BIN:?raw compiler required} ]] || exit 64
mkdir "$out"
trap 'r=$?;echo "$r" >"$out/STATUS"' EXIT
cd "$root"
export SOUNIO_STDLIB_PATH="$root/stdlib" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
ulimit -s 524288
sha256sum "$MADAROS_RAW_BIN" >"$out/COMPILER.sha256"
find stdlib tests/stdlib/genomics_counts -type f -print0|sort -z|xargs -0 sha256sum >"$out/INPUTS.sha256"
sha256sum scripts/stdlib/genomics_token_bench_gate.sh tests/reference/genomics_token_bench_reference.cpp >>"$out/INPUTS.sha256"
step(){ local name=$1;shift;local rc=0;"$@" >"$out/$name.stdout" 2>"$out/$name.stderr"||rc=$?;printf '%s\t%s\n' "$name" "$rc" >>"$out/RESULTS.tsv";return "$rc"; }
step cpp-build g++ -std=c++23 -O2 -Wall -Wextra -Werror tests/reference/genomics_token_bench_reference.cpp -o "$out/reference"
sha256sum "$out/reference" >"$out/ELFS.sha256"
for mode in slot snapshot;do
 step "$mode.generate" "$out/reference" --generate tests/stdlib/genomics_counts/token_bench.sio "$mode"
 cp "$out/$mode.generate.stdout" "$out/$mode.sio"
 step "$mode.check" timeout 180 "$MADAROS_RAW_BIN" --check "$out/$mode.sio"
 step "$mode.compile" timeout 180 "$MADAROS_RAW_BIN" compile "$out/$mode.sio" -o "$out/$mode.elf"
 test -s "$out/$mode.elf";chmod +x "$out/$mode.elf"
 sha256sum "$out/$mode.elf" >>"$out/ELFS.sha256"
 step "$mode.native" timeout 300 "$out/$mode.elf"
 step "$mode.concordance" "$out/reference" --check "$out/$mode.native.stdout"
done
awk '$1=="TOKEN_BENCH"{$5=0}{print}' "$out/slot.native.stdout" >"$out/wrong-total.input"
sed '/^TOKEN_BENCH_PASS$/d' "$out/slot.native.stdout" >"$out/missing-marker.input"
cat "$out/slot.native.stdout" "$out/slot.native.stdout" >"$out/duplicate.input"
for control in wrong-total missing-marker duplicate;do
 rc=0;"$out/reference" --check "$out/$control.input" >"$out/$control.stdout" 2>"$out/$control.stderr"||rc=$?
 if [[ "$control" == wrong-total ]];then expected=8;else expected=9;fi
 printf '%s\t%s\t%s\n' "$control" "$expected" "$rc" >>"$out/CONTROLS.tsv"
 [[ "$rc" == "$expected" ]]
done
sha256sum -c "$out/INPUTS.sha256" >"$out/INPUTS.final.verify"
sha256sum -c "$out/COMPILER.sha256" >"$out/COMPILER.final.verify"
sha256sum -c "$out/ELFS.sha256" >"$out/ELFS.final.verify"
echo TOKEN_BENCH_GATE_PASS
