#!/usr/bin/env bash
set -uo pipefail
REMOTE='set -uo pipefail
W=/tmp/sounio-mb-$$; mkdir -p "$W" && cd "$W" || exit 1
tar xzf - 2>/dev/null || exit 1
export SOUNIO_STDLIB_PATH="$W/stdlib"; export SOUNIO_BUILD_LOCK=/tmp/mb-$$.lock
bash scripts/ci/build_modular_madaros.sh "$W/m.elf" >"$W/b.log" 2>&1 </dev/null || { echo BUILDFAIL; tail -20 "$W/b.log"; exit 1; }
echo "BUILT"
run() { printf "  %-40s " "$1"; "$W/m.elf" --native-v2-compile "$2" "$W/t.elf" >"$W/c.log" 2>&1 </dev/null
  if [ -s "$W/t.elf" ]; then chmod +x "$W/t.elf"; echo "-> $(timeout 60 "$W/t.elf" 2>&1 </dev/null | tr "\n" " " | cut -c1-140)"
  else echo "COMPILEFAIL: $(grep -aoE "error[^\"]{0,70}" "$W/c.log"|head -1)"; fi; rm -f "$W/t.elf"; }
echo "== REPROS (correto: comeca com 85 29) =="
for r in repro8 repro32 repro32_small d_2fields k4; do run $r "x509/$r.sio"; done
echo "== TESTE DE REGRESSAO (esperado: oid=85,29 value=170,187 variance=7 confidence=9) =="
run knowledge_layout_shadows_user_field_name tests/run-pass/knowledge_layout_shadows_user_field_name.sio
echo "== KNOWN-FAILURE que deve passar agora =="
[ -f tests/known_failures/array_struct_u8_field_corruption_8elem.sio ] && run kf_8elem tests/known_failures/array_struct_u8_field_corruption_8elem.sio
echo "== RATCHET =="
MADAROS_BIN="$W/m.elf" bash scripts/ci/madaros_fixed_point_gate.sh >"$W/g.log" 2>&1 </dev/null; echo "  rc=$?"
grep -aoE "MADAROS_FIXED_POINT_[A-Z]+" "$W/g.log" | sort -u | sed "s/^/    /"
echo "== SUITE COMPLETA =="
timeout 5400 bash scripts/run_sio_test_suite.sh >"$W/s.log" 2>&1 </dev/null; echo "  rc=$?"
grep -aiE "Pass:|Fail:|Total:" "$W/s.log" | tail -5 | sed "s/^/    /"
echo "  --- FAIL-LIST-START ---"
grep -aoE "FAIL +[A-Za-z0-9_.]+" "$W/s.log" | awk "{print \$2}" | sort -u
echo "  --- FAIL-LIST-END ---"
rm -rf "$W"'
cd "$1" && tar czf - self-hosted stdlib bin scripts x509 tests examples docs 2>/dev/null \
  | srun --partition=all --ntasks=1 --cpus-per-task=24 --time=04:00:00 bash -c "$REMOTE" 2>&1 | grep -vE 'TMPDIR|chdir|^srun:'
