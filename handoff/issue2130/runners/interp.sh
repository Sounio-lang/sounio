#!/usr/bin/env bash
set -uo pipefail
REMOTE='set -uo pipefail
W=/tmp/sounio-ip-$$; mkdir -p "$W" && cd "$W" || exit 1
tar xzf - 2>/dev/null || exit 1
export SOUNIO_STDLIB_PATH="$W/stdlib"; export SOUNIO_BUILD_LOCK=/tmp/ip-$$.lock
bash scripts/ci/build_modular_madaros.sh "$W/m.elf" >"$W/b.log" 2>&1 </dev/null || { echo BUILDFAIL; tail -5 "$W/b.log"; exit 1; }
echo "REMOTE: built"
echo "== interpolacao arr_20_512_8 (LIMPO) -> d_2fields (CORROMPIDO) =="
echo "   esperado limpo: 85 170 ...   corrompido: 170 ..."
run() { printf "  %-6s " "$1"; "$W/m.elf" --native-v2-compile "x509/$1.sio" "$W/t.elf" >"$W/c.log" 2>&1 </dev/null
  if [ -s "$W/t.elf" ]; then chmod +x "$W/t.elf"; echo "-> $(timeout 30 "$W/t.elf" 2>&1 </dev/null | tr "
" " ")"
  else echo "COMPILEFAIL: $(grep -aoE "error[^\"]{0,60}" "$W/c.log"|head -1)"; fi; rm -f "$W/t.elf"; }
for r in i0 i1 i2 i3 i4 i5 d_2fields; do run $r; done

rm -rf "$W"'
cd "$1" && tar czf - self-hosted stdlib bin scripts x509 2>/dev/null \
  | srun --partition=all --ntasks=1 --cpus-per-task=24 --time=01:30:00 bash -c "$REMOTE" 2>&1 | grep -vE 'TMPDIR|chdir|^srun:'
