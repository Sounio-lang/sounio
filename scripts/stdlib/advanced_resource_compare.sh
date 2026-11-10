#!/usr/bin/env bash
# Usage: advanced_resource_compare.sh BASE_SOURCE CANDIDATE_SOURCE MADAROS_ELF OUTPUT
# Three paired same-compiler native observations; no input mutation/cache build.
# STDLIB_RESOURCE_COMPARE_CXX_SOURCE may locate a standalone copied analyzer.
set -euo pipefail
fail(){ echo "RESOURCE_COMPARE_REFUSE $*" >&2; exit 70; }
[[ $# == 4 && -n ${SLURM_JOB_ID:-} && $(id -u) == 1000 ]] || fail 'requires Slurm UID1000 and four arguments'
base=$(realpath "$1"); candidate=$(realpath "$2"); compiler=$(realpath "$3")
[[ $4 == /* ]] || fail 'output must be absolute'
out=$(realpath -m "$4")
[[ $base != "$candidate" ]] || fail 'distinct source snapshots required'
for src in "$base" "$candidate"; do
 [[ -d $src/stdlib && ! -w $src ]] || fail "source root must be effectively read-only: $src"
 [[ -z $(find "$src" \( -type f -o -type d \) -writable -print -quit) ]] || fail "writable source input: $src"
 [[ -z $(find "$src/stdlib" -type l -print -quit) ]] || fail 'stdlib symlinks require explicit custody review'
 case "$out/" in "$src/"*) fail 'output must be outside both source snapshots';; esac
done
[[ -f $compiler && -r $compiler && -x $compiler ]] || fail 'compiler unreadable/unexecutable'
[[ $(head -c 4 "$compiler") == $'\177ELF' ]] || fail 'compiler must be raw ELF, not resolver script'
[[ ! -e $out ]] || fail 'refusing existing output; preserve prior attempts'
script=$(realpath "${BASH_SOURCE[0]}")
script_root=$(cd "$(dirname "$script")/../.." && pwd)
cxx_source=${STDLIB_RESOURCE_COMPARE_CXX_SOURCE:-$script_root/tests/reference/stdlib_resource_compare.cpp}
cxx_source=$(realpath "$cxx_source")
[[ -f $cxx_source ]] || fail 'standalone C++ analyzer source missing'
mkdir -p "$out/tools" "$out/tmp" "$out/home"
cp "$script" "$out/tools/advanced_resource_compare.sh"
cp "$cxx_source" "$out/tools/stdlib_resource_compare.cpp"
/usr/bin/time --version > "$out/TIME_VERSION.txt"
g++ --version > "$out/CXX_VERSION.txt"
printf 'slurm_job_id=%s\nuid=%s\nbase=%s\ncandidate=%s\ncompiler=%s\n' "$SLURM_JOB_ID" "$(id -u)" "$base" "$candidate" "$compiler" > "$out/REQUEST.txt"
ulimit -s 524288
export LC_ALL=C
fixtures=(tests/run-pass/stdlib_deep_dynamic_qr.sio tests/run-pass/stdlib_deep_stats_count_glm.sio)
kernels=(qr count_glm)
markers=(STDLIB_DEEP_DYNAMIC_QR_PASS 'DEEP COUNT GLM PASS')
for fixture in "${fixtures[@]}"; do cmp "$base/$fixture" "$candidate/$fixture" || fail "fixture bytes differ: $fixture"; done
inventory(){ (cd "$1" && find . -type f -print0 | sort -z | xargs -0 sha256sum); }
inventory "$base" > "$out/before.INPUTS.pre.sha256"
inventory "$candidate" > "$out/after.INPUTS.pre.sha256"
sha256sum "$compiler" > "$out/COMPILER.pre.sha256"
verified=0
verify_inputs(){
 inventory "$base" > "$out/before.INPUTS.post.sha256"
 inventory "$candidate" > "$out/after.INPUTS.post.sha256"
 sha256sum "$compiler" > "$out/COMPILER.post.sha256"
 cmp "$out/before.INPUTS.pre.sha256" "$out/before.INPUTS.post.sha256" &&
 cmp "$out/after.INPUTS.pre.sha256" "$out/after.INPUTS.post.sha256" &&
 cmp "$out/COMPILER.pre.sha256" "$out/COMPILER.post.sha256"
}
finish(){ local rc=$?;trap - EXIT;if [[ $verified != 1 ]]; then verify_inputs || rc=70;fi;exit "$rc"; }
trap finish EXIT
env -i PATH=/usr/bin:/bin HOME="$out/home" "$compiler" --version > "$out/COMPILER_VERSION.txt" 2>&1
grep -Fq 'Madaros' "$out/COMPILER_VERSION.txt" || fail 'compiler identity banner mismatch'
g++ -std=c++23 -O2 -Wall -Wextra -pedantic "$out/tools/stdlib_resource_compare.cpp" -o "$out/tools/stdlib_resource_compare" > "$out/tools/compile.stdout" 2> "$out/tools/compile.stderr"
# Timing includes child exec/check plus this batch-driver overhead. Every native
# execution must retain its marker and exact whole stdout, including all numbers.
cat > "$out/tools/batch.sh" <<'DRIVER'
#!/usr/bin/env bash
set -euo pipefail
elf=$1; count=$2; marker=$3; golden=$4; stem=$5
: > "$stem.stderr"
for ((rep=0; rep<count; rep++)); do
 "$elf" > "$stem.current.stdout" 2>> "$stem.stderr"
 grep -Fq -- "$marker" "$stem.current.stdout"
 cmp "$golden" "$stem.current.stdout"
done
DRIVER
chmod +x "$out/tools/batch.sh"
for v in before after; do
 src=$base; [[ $v != after ]] || src=$candidate
 mkdir "$out/$v"
 for k in 0 1; do
  kernel=${kernels[$k]}; fixture=${fixtures[$k]}; stem="$out/$v/$kernel"
  # Raw current-source Madaros: explicit build subcommand, no bin resolver.
  (cd "$src" && env -i PATH=/usr/bin:/bin HOME="$out/home" TMPDIR="$out/tmp" MADAROS_RAW_BIN="$compiler" SOUNIO_STDLIB_PATH="$src/stdlib" SOUNIO_SOUC_ENGINE=madaros SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1 timeout --kill-after=20 600 "$compiler" build "$src/$fixture" --target native -o "$stem.elf") > "$stem.compile.stdout" 2> "$stem.compile.stderr"
  [[ -s $stem.elf ]] || fail "compiler produced no ELF: $v $kernel"
  chmod +x "$stem.elf"
  sha256sum "$stem.elf" "$src/$fixture" > "$stem.IDENTITIES.sha256"
  (cd "$src" && env -i PATH=/usr/bin:/bin HOME="$out/home" SOUNIO_STDLIB_PATH="$src/stdlib" "$stem.elf") > "$stem.warmup.stdout" 2> "$stem.warmup.stderr"
  grep -Fq -- "${markers[$k]}" "$stem.warmup.stdout" || fail 'missing native warmup completion marker'
  if [[ $v == after ]]; then cmp "$out/before/$kernel.warmup.stdout" "$stem.warmup.stdout" || fail 'native stdout changed';fi
 done
done
sha256sum "$out"/{before,after}/{qr,count_glm}.elf > "$out/NATIVE.pre.sha256"
header=$'kernel\tvariant\ttrial\tbatch\twall_s\tuser_s\tsystem_s\tmax_rss_kib\texit_status\tstdout_sha256\telf_sha256\tcompiler_sha256\tfixture_sha256'
printf '%s\n' "$header" > "$out/METRICS.tsv"
printf '%s\n' "$header" > "$out/CALIBRATION.tsv"
compiler_sha=$(cut -d' ' -f1 "$out/COMPILER.pre.sha256")
measure(){
 local kernel=$1 v=$2 trial=$3 batch=$4 destination=$5 k=0
 [[ $kernel != count_glm ]] || k=1
 local stem="$out/$v/${kernel}.t${trial}.b${batch}" rc=0 src="$base"
 [[ $v != after ]] || src=$candidate
 (cd "$src" && env -i PATH=/usr/bin:/bin HOME="$out/home" SOUNIO_STDLIB_PATH="$src/stdlib" /usr/bin/time -f '%e\t%U\t%S\t%M\t%x' -o "$stem.time" timeout --kill-after=10 900 bash "$out/tools/batch.sh" "$out/$v/$kernel.elf" "$batch" "${markers[$k]}" "$out/before/$kernel.warmup.stdout" "$stem") || rc=$?
 ((rc==0)) || fail "native batch failed: $kernel $v trial$trial batch$batch rc$rc"
 local wall user system rss exit_status
 IFS=$'\t' read -r wall user system rss exit_status < "$stem.time"
 [[ $exit_status == 0 ]] || fail 'GNU time reports failed execution'
 local stdout_sha elf_sha fixture_sha
 stdout_sha=$(sha256sum "$stem.current.stdout");stdout_sha=${stdout_sha%% *}
 elf_sha=$(sha256sum "$out/$v/$kernel.elf");elf_sha=${elf_sha%% *}
 fixture_sha=$(sha256sum "$base/${fixtures[$k]}");fixture_sha=${fixture_sha%% *}
 printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$kernel" "$v" "$trial" "$batch" "$wall" "$user" "$system" "$rss" "$exit_status" "$stdout_sha" "$elf_sha" "$compiler_sha" "$fixture_sha" >> "$destination"
}
batch=25
while :; do
 measure count_glm before 0 "$batch" "$out/CALIBRATION.tsv"
 measure count_glm after 0 "$batch" "$out/CALIBRATION.tsv"
 IFS=$'\t' read -r bw _ < "$out/before/count_glm.t0.b${batch}.time"
 IFS=$'\t' read -r aw _ < "$out/after/count_glm.t0.b${batch}.time"
 next=$("$out/tools/stdlib_resource_compare" --next-batch "$bw" "$aw" "$batch")
 [[ $next != "$batch" ]] || break
 batch=$next
done
printf '%s\n' "$batch" > "$out/COUNT_GLM_BATCH.txt"
for kernel in qr count_glm; do
 b=1;[[ $kernel != count_glm ]] || b=$batch
 for trial in 1 2 3; do
  if [[ $trial == 2 ]]; then order=(after before);else order=(before after);fi
  for v in "${order[@]}"; do measure "$kernel" "$v" "$trial" "$b" "$out/METRICS.tsv";done
 done
done
verify_inputs || fail 'source/compiler identity changed during comparison'
sha256sum "$out"/{before,after}/{qr,count_glm}.elf > "$out/NATIVE.post.sha256"
cmp "$out/NATIVE.pre.sha256" "$out/NATIVE.post.sha256" || fail 'native ELF changed during comparison'
verified=1
"$out/tools/stdlib_resource_compare" "$out/METRICS.tsv" > "$out/REPORT.tsv" 2> "$out/REPORT.stderr"
(cd "$out" && find . -type f ! -name FILES.sha256 ! -name READY -print0 | sort -z | xargs -0 sha256sum > FILES.sha256)
printf 'RESOURCE_COMPARE_PASS\n' > "$out/READY"
cat "$out/REPORT.tsv"
