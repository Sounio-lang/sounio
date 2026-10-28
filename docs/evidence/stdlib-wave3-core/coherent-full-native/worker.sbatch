#!/bin/bash
#SBATCH --job-name=stdlib-coherent-full-native
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/core/coherent-text-full02/results/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/core/coherent-text-full02/results/slurm-%j.err
set -euo pipefail
R=/scratch/sounio-stdlib-wave3-20261027/core/coherent-text-full02
B=/scratch/sounio-stdlib-wave3-20261027/core
F="$B/coherent-text-focused01"
P=/scratch/sounio-stdlib-inference-scale-20261024/stdlib-next-20261027
T="$P/tooling"
PLATFORM=/scratch/sounio-stdlib-wave3-20261027/services-tools-recovered/platform-abi-112/libstdlib_platform.so
ARCHIVE=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/coherent-text01/focused-source-elf-argv-receipts-19269-19270.tar.gz
export PATH="$P/tools:$T/root/usr/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$(dirname "$PLATFORM"):$T/root/usr/lib/x86_64-linux-gnu"
export MAGIC="$T/root/usr/lib/file/magic.mgc"
ulimit -s524288
ulimit -c0
trap 'r=$?;echo "$r" > "$R/results/WORKER_STATUS";date -u > "$R/results/ENDED_UTC";exit "$r"' EXIT
printf '%s\n' 'NATIVE COUPLED ELF REPLAY ONLY. Pristine ICU strict remains FAIL, recorded independently. No integrated whole-stdlib/generic compiler acceptance.' > "$R/results/SCOPE"
printf 'phase\texpected\tactual\n' > "$R/results/RESULTS.tsv"
phase(){ local name=$1 want=$2 got=0;shift 2;/usr/bin/time -v timeout 900 "$@" > "$R/results/$name.stdout" 2> "$R/results/$name.stderr" || got=$?;printf '%s\t%s\t%s\n' "$name" "$want" "$got" >> "$R/results/RESULTS.tsv";test "$want" = "$got";}
printf 'fcbaca9a807c7d264815edd920798d35a875dc4ff3f6edf28dd268e369681049  %s\n' "$F/results/SOURCE.sha256" > "$R/results/COMPILED_SOURCE_MANIFEST.sha256"
printf '19b5e632c3adb31942c181e22941c86e3258ec9d37de58548ce4e6182ba15442  %s\n' "$ARCHIVE" > "$R/results/FOCUSED_ARCHIVE.sha256"
printf 'de01597868c09725c2c26651d23203043c65fdc04a3e5e96f45dfe4a51d494c4  %s\n' "$PLATFORM" > "$R/results/PLATFORM.sha256"
for pin in COMPILED_SOURCE_MANIFEST FOCUSED_ARCHIVE PLATFORM;do sha256sum -c "$R/results/$pin.sha256" > "$R/results/$pin.before.verify";done
sha256sum -c "$F/results/SOURCE.sha256" > "$R/results/COMPILED_SOURCE.before.verify"
find "$R/input" -type f -print0|sort -z|xargs -0 sha256sum > "$R/results/INPUTS.sha256"
mkdir "$R/sealed"
tar -xzf "$ARCHIVE" -C "$R/sealed" coherent-text-focused01/coherent.elf coherent-text-focused01/bidi-corpus
ELF="$R/sealed/coherent-text-focused01/coherent.elf"
BIDI="$R/sealed/coherent-text-focused01/bidi-corpus"
chmod +x "$ELF"
printf '2bd52fdededc355138c796d6de92747d5a84c1e7764c9e10f6a69d9560048454  %s\n' "$ELF" > "$R/results/ELF.sha256"
sha256sum -c "$R/results/ELF.sha256" > "$R/results/ELF.before.verify"
read -r ELF_SHA _ < "$R/results/ELF.sha256"
printf '11c403171ad261c9c2b5ce7cf95946fd12f80deb5b68421aca368d701f625a09  %s\n' "$BIDI/CHUNKS.tsv" | sha256sum -c - > "$R/results/BIDI_MANIFEST.verify"
find "$BIDI" -type f -print0|sort -z|xargs -0 sha256sum > "$R/results/BIDI_BYTES.sha256"
printf 'name\tkind\tfiles\trows\tfixture_directory\tmanifest_sha256\n' > "$R/results/CORPORA.tsv"
verify_fixture(){
 local name=$1 kind=$2 files=$3 rows=$4 dir=$5 manifest_sha=$6 hashes=$7 hashes_sha=$8 hashes_root=$9
 printf '%s  %s\n%s  %s\n' "$manifest_sha" "$dir/MANIFEST.tsv" "$hashes_sha" "$hashes" > "$R/results/$name-PINS.sha256"
 sha256sum -c "$R/results/$name-PINS.sha256" > "$R/results/$name-PINS.before.verify"
 (cd "$hashes_root";sha256sum -c "$hashes") > "$R/results/$name-BYTES.before.verify"
 printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$kind" "$files" "$rows" "$dir" "$manifest_sha" >> "$R/results/CORPORA.tsv"
}
verify_fixture unicode 1 21971 11248160 "$B/restored/fixtures-exhaustive01/data" a872fecac523bbee18932592fdcb7fce9e932d79e8609508c07a97a447df46fd "$B/restored/fixtures-exhaustive01/FIXTURES.sha256" 86bf093265d707b93bda4e655b5ca94cdd130be7dc0608c281078f394a9df1ac "$B/restored/fixtures-exhaustive01"
verify_fixture uca 1 2307 1180750 "$B/uca-fixtures03/data" 0686152cf10c85f00576fa32fd87f93c8f6090af80923e75e9096e41c7a367bd "$B/uca-fixtures03/results/data.sha256" 100bb1d193628303bedab967769902e70d4e5ad7f643433f268c2a9cd5ae3e70 "$B/uca-fixtures03/data"
verify_fixture locale-default 2 763 390251 "$B/locale-fixtures02/defaults" 6f5e6003a0813992ff0fb970212206c5fa7665541e5100905e02352a6950b6ae "$B/locale-fixtures02/DEFAULTS.sha256" 3a9987638ea0bd81c3a486037477cd5e3a8e899fc7f34fd6a5a4ecf502bafe78 "$B/locale-fixtures02/defaults"
verify_fixture locale-options 2 4154 2126400 "$B/locale-fixtures02/options" eb1ebece158c358b29b32f7379151551b861593404bad55aad24ee75b79b382d "$B/locale-fixtures02/OPTIONS.sha256" 2c4e2f520f9e8b44504a5eec6b2c0f281823fd99cbef4028c1ab4043358d089d "$B/locale-fixtures02/options"
verify_fixture metadata 3 72 9190 "$B/locale-metadata-fixtures01/data" 852cf7a9ea503f88c63648fff37ae158f15299abe087e1295ece0166685e3b3f "$B/locale-metadata-fixtures01/DATA.sha256" 95031c9a90bfaa8742b3761c79f2ceb2da95e4f2452843421a080a1ac3fe2085 "$B/locale-metadata-fixtures01/data"
while IFS=: read -r name file;do
 phase "compile-$name" 0 g++ -std=c++23 -O2 -Wall -Wextra -Werror -Wno-misleading-indentation "$R/input/tools/unicode/$file" -o "$R/$name"
done <<'COMPILE'
unicode:run_chunks.cpp
unicode-collect:collect_chunks.cpp
locale:run_locale_chunks.cpp
locale-collect:collect_locale_chunks.cpp
metadata:run_locale_metadata_chunks.cpp
metadata-collect:collect_locale_metadata_chunks.cpp
bidi:run_bidi_coherent_chunks.cpp
COMPILE
find "$R" -maxdepth 1 -type f -executable -print0|sort -z|xargs -0 sha256sum > "$R/results/TOOLS.sha256"
mkdir -p "$R/bidi-probe/part0"
cp "$R/results/ELF.sha256" "$R/bidi-probe/part0/EXECUTABLE.sha256"
phase bidi-one-original 0 "$R/bidi" "$ELF" "$BIDI" "$R/bidi-probe/part0/chunks" 0 211 --coherent-kind 4
phase bidi-wrong-kind 64 "$R/bidi" "$ELF" "$BIDI" "$R/bidi-probe/forbidden-kind" 0 211 --coherent-kind 3
phase bidi-missing-argc 64 "$R/bidi" "$ELF" "$BIDI" "$R/bidi-probe/forbidden-argc" 0 211
test ! -e "$R/bidi-probe/forbidden-kind"
test ! -e "$R/bidi-probe/forbidden-argc"
phase bidi-incomplete-collect 1 "$R/bidi" --collect --coherent-kind 4 "$BIDI" "$ELF_SHA" "$R/bidi-probe/incomplete-summary" "$R/bidi-probe/part0/chunks"
phase bidi-duplicate-collect 1 "$R/bidi" --collect --coherent-kind 4 "$BIDI" "$ELF_SHA" "$R/bidi-probe/duplicate-summary" "$R/bidi-probe/part0/chunks" "$R/bidi-probe/part0/chunks"
for control in kind argc elf;do
 mkdir -p "$R/bidi-probe/tamper-$control"
 cp "$R/results/ELF.sha256" "$R/bidi-probe/tamper-$control/EXECUTABLE.sha256"
 cp -a "$R/bidi-probe/part0/chunks" "$R/bidi-probe/tamper-$control/chunks"
 case "$control" in
  kind) printf 'argc\t1\nkind\t3\n' > "$R/bidi-probe/tamper-$control/chunks/ELF_INVOCATION.tsv";;
  argc) printf 'argc\t2\nkind\t4\n' > "$R/bidi-probe/tamper-$control/chunks/ELF_INVOCATION.tsv";;
  elf) printf '%064d  wrong\n' 0 > "$R/bidi-probe/tamper-$control/EXECUTABLE.sha256";;
 esac
 phase "bidi-tamper-$control" 1 "$R/bidi" --collect --coherent-kind 4 "$BIDI" "$ELF_SHA" "$R/bidi-probe/forbidden-$control" "$R/bidi-probe/tamper-$control/chunks"
 test ! -e "$R/bidi-probe/forbidden-$control"
done
phase wrong-bidi-generate 0 "$F/bidi-generator" --wrong-level-control /scratch/sounio-stdlib-wave3-20261027/core/bidi-icu-complete03/candidate/upstream/ucd-18.0.0 "$R/wrong-bidi"
mkdir -p "$R/bidi-probe/wrong-level/data"
cp "$R/wrong-bidi/wrong-level.bin" "$R/bidi-probe/wrong-level/data/current.bin"
cd "$R/bidi-probe/wrong-level"
phase bidi-original-wrong-level 1 "$ELF" 4
grep -F BIDI_LEVEL_MISMATCH "$R/results/bidi-original-wrong-level.stdout"
test "$(grep -Fc BIDI_NATIVE_CONFORMANCE_OK "$R/results/bidi-original-wrong-level.stdout" || true)" = 0
cd "$R"
tail -n +2 "$R/results/CORPORA.tsv" | while IFS=$'\t' read -r name kind files rows dir manifest_sha;do
 case "$kind" in 1) tool=unicode;;2) tool=locale;;3) tool=metadata;;*) exit 1;;esac
 mkdir "$R/$name-parts"
 pids=""
 for part in 0 1 2 3 4 5 6 7;do
  parent="$R/$name-parts/part$part";mkdir "$parent"
  cp "$R/results/ELF.sha256" "$parent/EXECUTABLE.sha256"
  (/usr/bin/time -v "$R/$tool" "$ELF" "$dir" "$parent/chunks" "$part" 8 --coherent-kind "$kind" > "$parent/stdout" 2> "$parent/stderr";echo 0 > "$parent/OUTER_STATUS") &
  pids="$pids $!"
 done
 for pid in $pids;do wait "$pid";done
 phase "$name-complete-collect" 0 "$R/$tool-collect" --coherent-kind "$kind" "$dir" "$ELF_SHA" "$R/$name-summary" "$R/$name-parts"/part*/chunks
 grep -Fx "UNICODE_COMPLETE_MANIFEST_OK files=$files rows=$rows partitions=8" "$R/results/$name-complete-collect.stdout"
done
mkdir "$R/bidi-parts"
pids=""
for part in 0 1 2 3 4 5 6 7;do
 parent="$R/bidi-parts/part$part";mkdir "$parent"
 cp "$R/results/ELF.sha256" "$parent/EXECUTABLE.sha256"
 (/usr/bin/time -v "$R/bidi" "$ELF" "$BIDI" "$parent/chunks" "$part" 8 --coherent-kind 4 > "$parent/stdout" 2> "$parent/stderr";echo 0 > "$parent/OUTER_STATUS") &
 pids="$pids $!"
done
for pid in $pids;do wait "$pid";done
phase bidi-complete-collect 0 "$R/bidi" --collect --coherent-kind 4 "$BIDI" "$ELF_SHA" "$R/bidi-summary" "$R/bidi-parts"/part*/chunks
grep -Fx 'BIDI_COHERENT_FULL_NATIVE_OK files=211 cases=861948 kind=4 argc=1' "$R/results/bidi-complete-collect.stdout"
for pin in COMPILED_SOURCE_MANIFEST FOCUSED_ARCHIVE PLATFORM ELF INPUTS BIDI_BYTES TOOLS;do sha256sum -c "$R/results/$pin.sha256" > "$R/results/$pin.final.verify";done
sha256sum -c "$F/results/SOURCE.sha256" > "$R/results/COMPILED_SOURCE.final.verify"
while IFS=$'\t' read -r name kind files rows dir manifest_sha;do
 test "$name" != name || continue
 sha256sum -c "$R/results/$name-PINS.sha256" > "$R/results/$name-PINS.final.verify"
done < "$R/results/CORPORA.tsv"
(cd "$B/restored/fixtures-exhaustive01";sha256sum -c "$B/restored/fixtures-exhaustive01/FIXTURES.sha256") > "$R/results/unicode-BYTES.final.verify"
(cd "$B/uca-fixtures03/data";sha256sum -c "$B/uca-fixtures03/results/data.sha256") > "$R/results/uca-BYTES.final.verify"
(cd "$B/locale-fixtures02/defaults";sha256sum -c "$B/locale-fixtures02/DEFAULTS.sha256") > "$R/results/locale-default-BYTES.final.verify"
(cd "$B/locale-fixtures02/options";sha256sum -c "$B/locale-fixtures02/OPTIONS.sha256") > "$R/results/locale-options-BYTES.final.verify"
(cd "$B/locale-metadata-fixtures01/data";sha256sum -c "$B/locale-metadata-fixtures01/DATA.sha256") > "$R/results/metadata-BYTES.final.verify"
printf '1\n' > "$R/results/PRISTINE_STRICT_STATUS"
printf '%s\n' 'COHERENT_TEXT_FULL_NATIVE_ONLY_OK' > "$R/results/VERDICT"
echo COHERENT_TEXT_FULL_NATIVE_ONLY_OK
