#!/usr/bin/env bash
# Real source-built compiler contract gate; negative ELFs are NEVER executed.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${SOUNIO_PRIVATE_COMPILER:?absolute source-built Stage2 required}"
: "${SOUNIO_PRIVATE_OUTPUT:?new absolute output directory required}"
[[ $SOUNIO_PRIVATE_COMPILER == /* && -x $SOUNIO_PRIVATE_COMPILER && $SOUNIO_PRIVATE_OUTPUT == /* ]]
[[ ! -e $SOUNIO_PRIVATE_OUTPUT ]]
mkdir -p "$SOUNIO_PRIVATE_OUTPUT"
out=$SOUNIO_PRIVATE_OUTPUT
export LC_ALL=C LANG=C
ulimit -s 524288
ulimit -c 0
sha256sum "$SOUNIO_PRIVATE_COMPILER" > "$out/COMPILER.sha256"
find self-hosted/compiler/lean_single.sio scripts/ci/private_field_borrow_gate.sh \
 tests/multimodule/private_field_borrow tests/compile-fail/private_field_borrow_checked.sio \
 tests/run-pass/private_field_borrow_public.sio stdlib -type f -print0 |
 sort -z | xargs -0 sha256sum > "$out/SOURCE.sha256"
printf 'target\tcase\tcompile\texpectation\n' > "$out/RESULTS.tsv"
compile() {
 local target=$1 name=$2 src=$3 rc=0
 timeout 120 "$SOUNIO_PRIVATE_COMPILER" "$src" "$out/$target.$name.elf" --target "$target" \
  > "$out/$target.$name.compile.stdout" 2> "$out/$target.$name.compile.stderr" || rc=$?
 printf '%s\n' "$rc" > "$out/$target.$name.compile.status"
 LAST_RC=$rc
 if [[ -s $out/$target.$name.elf ]];then sha256sum "$out/$target.$name.elf" > "$out/$target.$name.ELF.sha256";fi
}
refusal() {
 local target=$1 name=$2 src=$3
 grep -q '^//@ error-pattern: E259$' "$src"
 compile "$target" "$name" "$src"
 printf '%s\t%s\t%s\tE259\n' "$target" "$name" "$LAST_RC" >> "$out/RESULTS.tsv"
 [[ $LAST_RC == 1 ]]
 grep -q 'error\[E259\]' "$out/$target.$name.compile."{stdout,stderr}
 [[ ! -s $out/$target.$name.elf ]]
}
positive() {
 local target=$1 name=$2 src=$3 marker=$4
 compile "$target" "$name" "$src"
 printf '%s\t%s\t%s\tpublic-compatible\n' "$target" "$name" "$LAST_RC" >> "$out/RESULTS.tsv"
 [[ $LAST_RC == 0 && -s $out/$target.$name.elf ]]
 readelf -h "$out/$target.$name.elf" > "$out/$target.$name.header.txt"
 if [[ $target == x86_64-linux ]];then
  grep -q 'Machine:.*X86-64' "$out/$target.$name.header.txt"
  chmod +x "$out/$target.$name.elf"
  local rc=0
  timeout 30 "$out/$target.$name.elf" > "$out/$target.$name.native.stdout" 2> "$out/$target.$name.native.stderr" || rc=$?
  printf '%s\n' "$rc" > "$out/$target.$name.native.status"
  [[ $rc == 0 ]]
  grep -Fxq "$marker" "$out/$target.$name.native.stdout"
 else
  grep -q 'Machine:.*AArch64' "$out/$target.$name.header.txt"
 fi
}
for target in x86_64-linux aarch64-linux;do
 for src in tests/multimodule/private_field_borrow/*.sio;do
  grep -q '^//@ compile-fail$' "$src" || continue
  refusal "$target" "$(basename "$src" .sio)" "$src"
 done
 refusal "$target" checked_owner tests/compile-fail/private_field_borrow_checked.sio
 for name in read write forge;do
  refusal "$target" "legacy_$name" "tests/compile-fail/private_field_$name"_cross_module_1386.sio
 done
 positive "$target" multimodule_public tests/multimodule/private_field_borrow/positive.sio MULTIMODULE_PRIVATE_BORROW_PUBLIC_PASS
 positive "$target" main_public tests/run-pass/private_field_borrow_public.sio PRIVATE_FIELD_BORROW_PUBLIC_PASS
 positive "$target" legacy_nested_public tests/run-pass/lean_nested_deref_store_public.sio 'PASS lean_nested_deref_store_public'
 compile "$target" global_public tests/multimodule/private_field_borrow/global_public.sio
 if [[ $target == x86_64-linux ]];then
  [[ $LAST_RC == 0 && -s $out/$target.global_public.elf ]]
  printf '%s\tglobal_public\t0\tcompile-only-public\n' "$target" >> "$out/RESULTS.tsv"
 else
  # Preserved BEFORE limitation: A64 global field borrow is typed as a value.
  # Not a privacy pass, a positive runtime pass, a skip, or a repaired feature.
  if [[ $LAST_RC == 0 ]];then
   [[ -s $out/$target.global_public.elf ]]
   readelf -h "$out/$target.global_public.elf" > "$out/$target.global_public.header.txt"
   grep -q 'Machine:.*AArch64' "$out/$target.global_public.header.txt"
   printf '%s\tglobal_public\t0\tcompile-only-public-now-supported\n' "$target" >> "$out/RESULTS.tsv"
  else
   [[ $LAST_RC == 1 ]]
   grep -q 'error\[E001\].*Type mismatch' "$out/$target.global_public.compile."{stdout,stderr}
   printf '%s\tglobal_public\t1\tUNSUPPORTED_BEFORE_E001\n' "$target" >> "$out/RESULTS.tsv"
  fi
 fi
done
if [[ -n ${SOUNIO_PRIVATE_EXTRA172_ROOT:-} ]];then
 [[ $SOUNIO_PRIVATE_EXTRA172_ROOT == /* ]]
 src="$SOUNIO_PRIVATE_EXTRA172_ROOT/docs/evidence/stdlib-SIO-172/ci_repair_20261025/original_phonon_factor_private.sio"
 sha256sum "$src" > "$out/ORIGINAL172.sha256"
 find "$SOUNIO_PRIVATE_EXTRA172_ROOT/stdlib" -type f -print0 | sort -z | xargs -0 sha256sum > "$out/EXTRA172_LIB.sha256"
 saved=$PWD;cd "$SOUNIO_PRIVATE_EXTRA172_ROOT"
 # Raw lean_single resolves stdlib relative to cwd; the source stays byte-exact.
 compile x86_64-linux original_phonon "$src"
 [[ $LAST_RC == 1 ]]
 grep -q 'error\[E259\]' "$out/x86_64-linux.original_phonon.compile."{stdout,stderr}
 [[ ! -s $out/x86_64-linux.original_phonon.elf ]]
 cd "$saved"
 sha256sum -c "$out/ORIGINAL172.sha256" > "$out/ORIGINAL172.verify.txt"
 sha256sum -c "$out/EXTRA172_LIB.sha256" > "$out/EXTRA172_LIB.verify.txt"
fi
sha256sum -c "$out/SOURCE.sha256" > "$out/SOURCE.verify.txt"
sha256sum -c "$out/COMPILER.sha256" > "$out/COMPILER.verify.txt"
echo PRIVATE_FIELD_BORROW_GATE_PASS | tee "$out/FINAL.txt"
