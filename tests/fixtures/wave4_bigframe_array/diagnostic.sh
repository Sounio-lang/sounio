#!/usr/bin/env bash
# Collects diagnostics. Exit0 means completed custody, not that every case passed.
set -euo pipefail
[[ ${SLURM_JOB_ID:-} =~ ^[1-9][0-9]*$ ]] || { echo "Slurm required" >&2;exit 64; }
[[ $# == 3 && "$3" == /* && "$3" =~ ^/[A-Za-z0-9_./-]+$ ]] || exit 64
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
RAW=$(realpath -e "$1");STDLIB=$(realpath -e "$2");OUT=$(realpath -m "$3")
[[ -x "$RAW" && -f "$STDLIB/data/bigframe_ops.sio" && ! -e "$OUT" ]] || exit 64
mkdir -p "$(dirname "$OUT")";mkdir "$OUT"
finish(){ local r=$? changed=0;trap - EXIT
 for name in SOURCES RAW STDLIB INDEX_STDLIB SDK TOOLS EXECUTABLES;do
  if [[ -s "$OUT/$name.sha256" ]];then sha256sum -c "$OUT/$name.sha256" > "$OUT/$name.final.verify" 2>&1 || changed=1;fi
 done
 [[ "$changed" == 0 ]] || r=1
 printf '%s\n' "$r" > "$OUT/COLLECTION_STATUS"
 if [[ "$r" == 0 ]];then echo ARRAY_DIAGNOSTIC_COLLECTED;fi
 exit "$r"
}
trap finish EXIT
printf '%s\n' "slurm:$SLURM_JOB_ID" > "$OUT/EXECUTION_AUTHORITY"
printf '%s\n' 'Collection only; PHASES.tsv contains actual failures. No language-cast contract or production repair is asserted.' > "$OUT/SCOPE"
find "$HERE" -type f -print0|sort -z|xargs -0 sha256sum > "$OUT/SOURCES.sha256"
sha256sum "$RAW" > "$OUT/RAW.sha256"
find "$STDLIB" -type f -print0|sort -z|xargs -0 sha256sum > "$OUT/STDLIB.sha256"
awk -F '\t' 'NF!=4||$1==""||seen[$1]++{bad=1} END{if(NR!=19||bad)exit 1}' "$HERE/CASES.tsv"
while IFS=$'\t' read -r name file key library;do
 [[ "$name" =~ ^[A-Za-z0-9_]+$ && "$key" =~ ^[A-Za-z0-9_]+$ && "$library" =~ ^(original|index)$ ]]
 path=$(realpath -e "$HERE/$file");[[ "$path" == "$HERE/"* && -f "$path" ]]
done < "$HERE/CASES.tsv"
flags=(-std=c++23 -O2 -Wall -Wextra -Werror -static-libstdc++ -static-libgcc)
CXX=${CXX:-g++}
if [[ -n ${SOUNIO_GCC_SYSROOT:-} ]];then
 T=$(realpath -e "$SOUNIO_GCC_SYSROOT")
 CXX="$T/usr/bin/x86_64-linux-gnu-g++-13"
 flags=(--sysroot="$T" -B"$T/usr/libexec/gcc/x86_64-linux-gnu/13/" -B"$T/usr/bin/" -isystem "$T/usr/include/c++/13" -isystem "$T/usr/include/x86_64-linux-gnu/c++/13" "${flags[@]}")
 find "$T" -type f -print0|sort -z|xargs -0 sha256sum > "$OUT/SDK.sha256"
fi
(
 if [[ -n ${T:-} ]];then
  export PATH="$T/usr/bin:/usr/bin:/bin" LD_LIBRARY_PATH="$T/deps" COMPILER_PATH="$T/usr/libexec/gcc/x86_64-linux-gnu/13:$T/usr/bin" LIBRARY_PATH="$T/usr/lib/gcc/x86_64-linux-gnu/13:$T/usr/lib/x86_64-linux-gnu:$T/lib/x86_64-linux-gnu"
 fi
 command -v "$CXX" > "$OUT/CXX.path"
 "$CXX" --version > "$OUT/CXX.version"
 sha256sum "$(command -v "$CXX")" > "$OUT/TOOLS.sha256"
 "$CXX" "${flags[@]}" "$HERE/reference.cpp" -o "$OUT/reference"
)
sha256sum "$OUT/reference" > "$OUT/EXECUTABLES.sha256"
"$OUT/reference" --self-test > "$OUT/reference-self-test.stdout"
"$OUT/reference" --expected array > "$OUT/reference-good.stdout"
sed '2s/.*/0/' "$OUT/reference-good.stdout" > "$OUT/reference-wrong.stdout"
head -n2 "$OUT/reference-good.stdout" > "$OUT/reference-truncated.stdout"
for control in wrong truncated wrong-case;do
 file="$OUT/reference-$control.stdout";key=array
 if [[ "$control" == wrong-case ]];then file="$OUT/reference-good.stdout";key=word7;fi
 rc=0;"$OUT/reference" --verify "$key" "$file" > "$OUT/control-$control.log" 2>&1 ||rc=$?
 [[ "$rc" == 1 ]]
done
echo ACTUAL_ORACLE_REJECTED_THREE_WRONG_OUTPUTS > "$OUT/ORACLE_CONTROLS"
cp -a "$STDLIB" "$OUT/index-stdlib"
target=$(realpath -e "$OUT/index-stdlib/data/bigframe_ops.sio")
[[ "$target" == "$OUT/index-stdlib/data/bigframe_ops.sio" ]]
"$OUT/reference" --index-variant "$STDLIB/data/bigframe_ops.sio" "$target" > "$OUT/INDEX_VARIANT.log"
find "$OUT/index-stdlib" -type f -print0|sort -z|xargs -0 sha256sum > "$OUT/INDEX_STDLIB.sha256"
export SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
ulimit -s 524288;ulimit -c 0
while IFS=$'\t' read -r name file key library;do
 for mode in 0 1;do
  export SOUNIO_LOOP_REGIONS="$mode" SOUNIO_STDLIB_PATH="$STDLIB"
  if [[ "$library" == index ]];then export SOUNIO_STDLIB_PATH="$OUT/index-stdlib";fi
  rc=0;timeout 300 "$RAW" --check "$HERE/$file" > "$OUT/$name$mode.check.stdout" 2>&1 ||rc=$?
  printf '%s\t%s\tcheck\t%s\n' "$name" "$mode" "$rc" >> "$OUT/PHASES.tsv"
  rc=0;timeout 600 "$RAW" compile "$HERE/$file" -o "$OUT/$name$mode.elf" > "$OUT/$name$mode.compile.stdout" 2>&1 ||rc=$?
  printf '%s\t%s\tcompile\t%s\n' "$name" "$mode" "$rc" >> "$OUT/PHASES.tsv"
  if [[ "$rc" == 0 && -s "$OUT/$name$mode.elf" ]];then
   chmod +x "$OUT/$name$mode.elf";sha256sum "$OUT/$name$mode.elf" >> "$OUT/EXECUTABLES.sha256"
   rc=0;timeout 30 "$OUT/$name$mode.elf" > "$OUT/$name$mode.native.stdout" 2>&1 ||rc=$?
   printf '%s\t%s\tnative\t%s\n' "$name" "$mode" "$rc" >> "$OUT/PHASES.tsv"
   rc=0;"$OUT/reference" --verify "$key" "$OUT/$name$mode.native.stdout" > "$OUT/$name$mode.oracle.stdout" 2>&1 ||rc=$?
   printf '%s\t%s\toracle\t%s\n' "$name" "$mode" "$rc" >> "$OUT/PHASES.tsv"
  else
   printf '%s\t%s\tnative\tnot-run\n%s\t%s\toracle\tnot-run\n' "$name" "$mode" "$name" "$mode" >> "$OUT/PHASES.tsv"
  fi
 done
done < "$HERE/CASES.tsv"
[[ $(wc -l < "$OUT/PHASES.tsv") == 152 ]]
