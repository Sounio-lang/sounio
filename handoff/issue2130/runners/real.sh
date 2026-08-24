#!/usr/bin/env bash
set -uo pipefail
REMOTE='set -uo pipefail
W=/tmp/sounio-rs-$$; mkdir -p "$W" && cd "$W" || exit 1
tar xzf - 2>/dev/null || exit 1
export SOUNIO_STDLIB_PATH="$W/stdlib"; export SOUNIO_BUILD_LOCK=/tmp/rs-$$.lock
bash scripts/ci/build_modular_madaros.sh "$W/m.elf" >"$W/b.log" 2>&1 </dev/null || { echo BUILDFAIL; tail -12 "$W/b.log"; exit 1; }
echo "BUILT"
for mode in --native-v2-compile compile; do
  printf "  %-20s " "$mode"
  if [ "$mode" = compile ]; then "$W/m.elf" compile x509/real_shapes.sio -o "$W/t.elf" >"$W/c.log" 2>&1 </dev/null
  else "$W/m.elf" $mode x509/real_shapes.sio "$W/t.elf" >"$W/c.log" 2>&1 </dev/null; fi
  if [ -s "$W/t.elf" ]; then chmod +x "$W/t.elf"; timeout 60 "$W/t.elf" </dev/null >/dev/null 2>&1
    rc=$?; echo "exit=$rc  (0=todos os campos corretos; 1=ExtensionEntry 2=GeneralName 3=RdnEntry)"
  else echo "COMPILEFAIL: $(grep -aoE "error[^\"]{0,70}" "$W/c.log"|head -1)"; fi
  rm -f "$W/t.elf"
done
rm -rf "$W"'
cd "$1" && tar czf - self-hosted stdlib bin scripts x509 tests 2>/dev/null \
  | srun --partition=all --ntasks=1 --cpus-per-task=24 --time=01:30:00 bash -c "$REMOTE" 2>&1 | grep -vE 'TMPDIR|chdir|^srun:'
