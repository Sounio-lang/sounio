#!/usr/bin/env bash
set -uo pipefail
REMOTE='set -uo pipefail
W=/tmp/sounio-nm-$$; mkdir -p "$W" && cd "$W" || exit 1
tar xzf - 2>/dev/null || exit 1
export SOUNIO_STDLIB_PATH="$W/stdlib"; export SOUNIO_BUILD_LOCK=/tmp/nm-$$.lock
bash scripts/ci/build_modular_madaros.sh "$W/m.elf" >"$W/b.log" 2>&1 </dev/null || { echo BUILDFAIL; tail -5 "$W/b.log"; exit 1; }
echo "REMOTE: built"
echo "== hipotese: colisao por NOME. j4 corrompe, j3 limpa, so nomes diferem =="
run() { printf "  %-10s " "$1"; "$W/m.elf" --native-v2-compile "x509/$1.sio" "$W/t.elf" >"$W/c.log" 2>&1 </dev/null
  if [ -s "$W/t.elf" ]; then chmod +x "$W/t.elf"; echo "-> $(timeout 30 "$W/t.elf" 2>&1 </dev/null | tr "
" " ")"
  else echo "COMPILEFAIL: $(grep -aoE "error[^\"]{0,60}" "$W/c.log"|head -1)"; fi; rm -f "$W/t.elf"; }
echo "  k1=campos renomeados k2=locais k3=tipo/fn k4=INVERSO(limpo recebe nomes do corrompido) k5=ordem de print"
for r in j3 j4 k1 k2 k3 k4 k5; do run $r; done

rm -rf "$W"'
cd "$1" && tar czf - self-hosted stdlib bin scripts x509 2>/dev/null \
  | srun --partition=all --ntasks=1 --cpus-per-task=24 --time=01:30:00 bash -c "$REMOTE" 2>&1 | grep -vE 'TMPDIR|chdir|^srun:'
