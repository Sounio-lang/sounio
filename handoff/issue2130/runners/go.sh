#!/usr/bin/env bash
set -uo pipefail
REMOTE='set -uo pipefail
W=/tmp/sounio-go-$$; mkdir -p "$W" && cd "$W" || exit 1
tar xzf - 2>/dev/null || exit 1
export SOUNIO_STDLIB_PATH="$W/stdlib"; export SOUNIO_BUILD_LOCK=/tmp/go-$$.lock
bash scripts/ci/build_modular_madaros.sh "$W/m.elf" >"$W/b.log" 2>&1 </dev/null || { echo BUILDFAIL; tail -12 "$W/b.log"; exit 1; }
echo "BUILT"
printf "  native-v2: "; "$W/m.elf" --native-v2-compile x509/gn_order.sio "$W/t.elf" >"$W/c.log" 2>&1 </dev/null
if [ -s "$W/t.elf" ]; then chmod +x "$W/t.elf"; timeout 60 "$W/t.elf" </dev/null 2>&1 | sed "s/^/    /"; else echo "COMPILEFAIL: $(grep -aoE "error[^\"]{0,70}" "$W/c.log"|head -1)"; fi
rm -f "$W/t.elf"
rm -rf "$W"'
cd "$1" && tar czf - self-hosted stdlib bin scripts x509 tests 2>/dev/null \
  | srun --partition=all --ntasks=1 --cpus-per-task=24 --time=01:30:00 bash -c "$REMOTE" 2>&1 | grep -vE 'TMPDIR|chdir|^srun:'
