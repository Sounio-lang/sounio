#!/usr/bin/env bash
set -uo pipefail
REMOTE='set -uo pipefail
W=/tmp/sounio-gg-$$; mkdir -p "$W" && cd "$W" || exit 1
tar xzf - 2>/dev/null || exit 1
export SOUNIO_STDLIB_PATH="$W/stdlib"; export SOUNIO_BUILD_LOCK=/tmp/gg-$$.lock
bash scripts/ci/build_modular_madaros.sh "$W/m.elf" >"$W/b.log" 2>&1 </dev/null || { echo BUILDFAIL; tail -15 "$W/b.log"; exit 1; }
echo "BUILT"
MADAROS_BIN="$W/m.elf" MADAROS_RAW_BIN="$W/m.elf" bash scripts/ci/madaros_source_to_elf_gate.sh >"$W/g.log" 2>&1 </dev/null
echo "GATE rc=$?"
grep -aE "FAIL|knowledge_field_shadow|PASS|ok$" "$W/g.log" | tail -12 | sed "s/^/  /"
rm -rf "$W"'
cd "$1" && tar czf - self-hosted stdlib bin scripts tests 2>/dev/null \
  | srun --partition=all --ntasks=1 --cpus-per-task=24 --time=01:30:00 bash -c "$REMOTE" 2>&1 | grep -vE 'TMPDIR|chdir|^srun:'
