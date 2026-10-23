#!/usr/bin/env bash
set -euo pipefail
umask 077
run_root=/scratch/sounio-nb2-residual-bars-20261023/run1
src=$run_root/source
out=$run_root/artifacts
stage=identity
trap 'status=$?; printf "last_stage=%s\nexit_status=%s\n" "$stage" "$status" > "$out/DISPATCH_STATUS.txt"' EXIT
[[ $(id -u) == 1000 && $(id -g) == 1000 && ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || exit 65
[[ $(realpath -e "$run_root") == "$run_root" && $(realpath -e "$src") == "$src" ]] || exit 65
printf 'uid=%s\ngid=%s\njob_id=%s\nnode=%s\n' "$(id -u)" "$(id -g)" "$SLURM_JOB_ID" "$(hostname)" > "$out/IDENTITY.txt"
export SOUNIO_STDLIB_PATH="$src/stdlib" TMPDIR="$run_root/tmp"
export SOUNIO_MADAROS_CACHE="$run_root/cache" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
export SOUNIO_BUILD_LOCK="$run_root/build.lock"
ulimit -s 524288
cd "$src"
stage=source_integrity
sha256sum --check --status "$run_root/SOURCE_FILES.sha256"
sha256sum "$run_root/job.sh" "$run_root/dispatch.sh" > "$out/CONTROL_IDENTITIES.sha256"
stage=current_source_build
/usr/bin/time -f 'elapsed_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kib=%M\nexit_code=%x' -o "$out/BUILD_RESOURCE.txt" bash scripts/ci/build_modular_madaros.sh "$out/madaros.elf" > "$out/build.stdout" 2> "$out/build.stderr"
sha256sum "$out/madaros.elf" > "$out/COMPILER.sha256"
gzip -n -c "$out/madaros.elf" > "$out/madaros.elf.gz"
export MADAROS_RAW_BIN="$out/madaros.elf" SOUNIO_SOUC_ENGINE=madaros
printf 'kind\tsource\texit_status\n' > "$out/RESULTS.tsv"
stage=typechecks
checks_ok=true
for source in stdlib/math/logarithmic.sio stdlib/stats/regression/nb2_diagnostics.sio stdlib/stats/regression/nb2_deviance.sio stdlib/viz/bar.sio stdlib/viz/ir.sio stdlib/viz/viz_html.sio stdlib/io/sci_print.sio examples/viz_nb2_residual_bars.sio examples/viz_nb2_deviance.sio; do
 name=${source//\//_}
 set +e
 "$src/bin/souc" check "$source" > "$out/checks/$name.log" 2>&1
 status=$?
 set -e
 printf 'typecheck\t%s\t%s\n' "$source" "$status" >> "$out/RESULTS.tsv"
 if [[ "$status" != 0 ]]; then checks_ok=false; fi
done
[[ "$checks_ok" == true ]] || exit 1
for demo in viz_nb2_residual_bars viz_nb2_deviance; do
 stage="compile_$demo"
 demo_out="$out/native/$demo"
 mkdir "$demo_out"
 "$src/bin/souc" compile "examples/$demo.sio" -o "$demo_out/program.elf" > "$demo_out/compile.log" 2>&1
 sha256sum "$demo_out/program.elf" "examples/$demo.sio" > "$demo_out/IDENTITIES.sha256"
 stage="execute_$demo"
 set +e
 /usr/bin/time -f 'elapsed_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kib=%M\nexit_code=%x' -o "$demo_out/RESOURCE.txt" "$demo_out/program.elf" > "$demo_out/native.stdout" 2> "$demo_out/native.stderr"
 status=$?
 set -e
 printf 'native\texamples/%s.sio\t%s\n' "$demo" "$status" >> "$out/RESULTS.tsv"
 sha256sum "$demo_out/native.stdout" "$demo_out/native.stderr" >> "$demo_out/IDENTITIES.sha256"
 gzip -n -c "$demo_out/program.elf" > "$demo_out/program.elf.gz"
 [[ "$status" == 0 ]] || exit "$status"
done
stage=endpoint_integrity
sha256sum --check --status "$run_root/SOURCE_FILES.sha256"
sha256sum --check --status "$out/COMPILER.sha256"
printf 'source_endpoints_equal=true\ncompiler_build_exit=0\ntypechecks=9\nnative_examples=2\ntests_run=false\nnumerical_concordance=false\ntraining_quota_writes=none_by_job_recipe\n' > "$out/RESULT.txt"
stage=complete
