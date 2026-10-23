#!/usr/bin/env bash
set -euo pipefail
umask 077
prior=/scratch/sounio-nb2-residual-bars-20261023/run1
run_root=/scratch/sounio-nb2-residual-bars-20261023/run2
src=$prior/source
out=$run_root/artifacts
stage=identity
trap 'status=$?; printf "last_stage=%s\nexit_status=%s\n" "$stage" "$status" > "$out/DISPATCH_STATUS.txt"' EXIT
[[ $(id -u) == 1000 && $(id -g) == 1000 && ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || exit 65
printf 'uid=%s\ngid=%s\njob_id=%s\nnode=%s\n' "$(id -u)" "$(id -g)" "$SLURM_JOB_ID" "$(hostname)" > "$out/IDENTITY.txt"
export SOUNIO_STDLIB_PATH="$src/stdlib" TMPDIR="$run_root/tmp"
export SOUNIO_MADAROS_CACHE="$run_root/cache" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
export SOUNIO_BUILD_LOCK="$run_root/build.lock" MADAROS_RAW_BIN="$prior/artifacts/madaros.elf" SOUNIO_SOUC_ENGINE=madaros
ulimit -s 524288
cd "$src"
stage=input_integrity
sha256sum --check --status "$prior/SOURCE_FILES.sha256"
sha256sum --check --status "$prior/artifacts/COMPILER.sha256"
(cd "$run_root" && sha256sum --check --status CONSUMER_FILES.sha256)
sha256sum "$run_root/job.sh" "$run_root/dispatch.sh" > "$out/CONTROL_IDENTITIES.sha256"
printf 'kind\tsource\texit_status\n' > "$out/RESULTS.tsv"
for demo in viz_nb2_residual_bars viz_nb2_deviance; do
 source="$run_root/examples/$demo.sio"
 if [[ "$demo" == viz_nb2_deviance ]]; then source="$src/examples/$demo.sio"; fi
 demo_out="$out/native/$demo"
 mkdir "$demo_out"
 stage="typecheck_$demo"
 "$src/bin/souc" check "$source" > "$demo_out/typecheck.log" 2>&1
 printf 'typecheck\t%s\t0\n' "$source" >> "$out/RESULTS.tsv"
 stage="compile_$demo"
 "$src/bin/souc" compile "$source" -o "$demo_out/program.elf" > "$demo_out/compile.log" 2>&1
 sha256sum "$demo_out/program.elf" "$source" > "$demo_out/IDENTITIES.sha256"
 stage="execute_$demo"
 set +e
 /usr/bin/time -f 'elapsed_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kib=%M\nexit_code=%x' -o "$demo_out/RESOURCE.txt" "$demo_out/program.elf" > "$demo_out/native.stdout" 2> "$demo_out/native.stderr"
 status=$?
 set -e
 printf 'native\t%s\t%s\n' "$source" "$status" >> "$out/RESULTS.tsv"
 sha256sum "$demo_out/native.stdout" "$demo_out/native.stderr" >> "$demo_out/IDENTITIES.sha256"
 gzip -n -c "$demo_out/program.elf" > "$demo_out/program.elf.gz"
 [[ "$status" == 0 ]] || exit "$status"
done
stage=endpoint_integrity
sha256sum --check --status "$prior/SOURCE_FILES.sha256"
sha256sum --check --status "$prior/artifacts/COMPILER.sha256"
(cd "$run_root" && sha256sum --check --status CONSUMER_FILES.sha256)
printf 'compiler_build_job=16017\ncompiler_rebuilt_in_retry=false\ncompiler_source_commit=99082d7bcc404ef308347c25b2e67954ddf4c793\nconsumer_source_commit=13c825f7a7b3fbc78c7c25754d212d244b09a1d2\nmanifest_listed_inputs_equal=true\ntypechecks=2\nnative_examples=2\ntests_run=false\nnumerical_concordance=false\n' > "$out/RESULT.txt"
stage=complete
