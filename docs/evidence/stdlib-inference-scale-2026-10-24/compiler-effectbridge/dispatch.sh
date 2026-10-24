#!/bin/bash
set -euo pipefail
trap 'rc=$?; if [[ $rc != 0 ]]; then printf "FAILED %s\n" "$rc" > /scratch/sounio-stdlib-inference-scale-20261024/compiler_effectbridge01/artifacts/STATUS; fi' EXIT
root=/scratch/sounio-stdlib-inference-scale-20261024/compiler_effectbridge01
cd "$root/source"
export TMPDIR="$root/tmp" SOUNIO_BUILD_LOCK="$root/build.lock" SOUNIO_MADAROS_CACHE="$root/cache" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1 SOUNIO_STDLIB_PATH="$root/source/stdlib" SOUNIO_SOUC_ENGINE=madaros
ulimit -s 524288
find bin scripts self-hosted stdlib -type f -print0 | sort -z | xargs -0 sha256sum > "$root/INPUTS.sha256"
unset MADAROS_RAW_BIN
/usr/bin/time -v -o "$root/artifacts/BUILD_RESOURCE.txt" bash scripts/ci/build_modular_madaros.sh "$root/artifacts/madaros.elf" > "$root/artifacts/build.stdout" 2> "$root/artifacts/build.stderr"
sha256sum "$root/artifacts/madaros.elf" > "$root/artifacts/COMPILER.sha256"
export MADAROS_RAW_BIN="$root/artifacts/madaros.elf"
printf 'kind\tsource\texit_status\n' > "$root/artifacts/RESULTS.tsv"
for src in tests/run-pass/stdlib_deep_linear_borrow.sio tests/run-pass/stdlib_deep_callback_binding.sio; do
 name=$(basename "$src" .sio)
 bin/souc check "$src" > "$root/artifacts/$name.check.log" 2>&1
 chmod +x "$root/artifacts/$name.elf"
 bin/souc compile "$src" -o "$root/artifacts/$name.elf" > "$root/artifacts/$name.compile.log" 2>&1
 chmod +x "$root/artifacts/$name.elf"
 "$root/artifacts/$name.elf" > "$root/artifacts/$name.stdout" 2> "$root/artifacts/$name.stderr"
 printf 'native\t%s\t0\n' "$src" >> "$root/artifacts/RESULTS.tsv"
done
for src in tests/compile-fail/stdlib_deep_callback_private*.sio tests/compile-fail/stdlib_deep_buffer*.sio tests/compile-fail/madaros_module_qualified_call_private_fn.sio; do
 name=$(basename "$src" .sio); rc=0
 bin/souc check "$src" > "$root/artifacts/$name.check.log" 2>&1 || rc=$?
 printf 'negative\t%s\t%s\n' "$src" "$rc" >> "$root/artifacts/RESULTS.tsv"
 [[ $rc == 1 ]] || exit 1
 if [[ "$name" == *callback* || "$name" == madaros_module* ]]; then grep -q E175 "$root/artifacts/$name.check.log"; fi
done
echo COMPLETE > "$root/artifacts/STATUS"
