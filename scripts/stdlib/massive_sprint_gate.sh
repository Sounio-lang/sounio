#!/usr/bin/env bash
# Linux x86-64 native integration gate. Invoke on a Slurm CPU worker.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
out=${STDLIB_MASSIVE_OUT:?set STDLIB_MASSIVE_OUT to a fresh absolute directory}
[[ "$out" == /* && ! -e "$out" ]] || { echo "output must be a fresh absolute directory" >&2; exit 64; }
raw=${MADAROS_RAW_BIN:?identify the source-built raw compiler}
[[ -x "$raw" && "$(uname -sm)" == "Linux x86_64" ]] || exit 64
mkdir -p "$out"
export SOUNIO_STDLIB_PATH="$root/stdlib" SOUNIO_SOUC_ENGINE=madaros
export SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1 SOUNIO_REQUIRE_COMMITTED_MADAROS=0
export LD_LIBRARY_PATH="$out${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ulimit -s 524288
ulimit -c 0
find stdlib tests/stdlib/massive tests/stdlib/sync tests/reference examples/atomic_demo.sio examples/stdlib_science_workflow.sio examples/stdlib_science_report_check.sio docs/evidence/stdlib-massive-next docs/evidence/stdlib-massive-2026-10-26 runtime scripts/stdlib tools/science_boundary -type f -print0 | sort -z | xargs -0 sha256sum > "$out/INPUTS.sha256"
sha256sum "$raw" bin/madaros bin/souc > "$out/COMPILER.sha256"
g++ --version > "$out/CXX.txt"
printf 'phase\tsource\texit_status\n' > "$out/RESULTS.tsv"
fail=0
record() {
 local phase=$1 source=$2; shift 2
 local rc=0
 "$@" > "$out/$phase.stdout" 2> "$out/$phase.stderr" || rc=$?
 printf '%s\t%s\t%s\n' "$phase" "$source" "$rc" >> "$out/RESULTS.tsv"
 if [[ "$rc" != 0 ]]; then fail=1; fi
 return "$rc"
}
record platform-build runtime/stdlib_platform.cpp g++ -std=c++23 -O2 -Wall -Wextra -Werror -shared -fPIC -pthread runtime/stdlib_platform.cpp -o "$out/libstdlib_platform.so" || true
record callback-abi-build tests/reference/stdlib_callback_abi_reference.cpp g++ -std=c++23 -O2 -Wall -Wextra -Werror -pthread tests/reference/stdlib_callback_abi_reference.cpp runtime/stdlib_platform.cpp -o "$out/callback-abi" || true
record callback-abi tests/reference/stdlib_callback_abi_reference.cpp timeout 30 "$out/callback-abi" || true
if [[ -f "$out/callback-abi" ]];then sha256sum "$out/callback-abi" >> "$out/EXECUTABLES.sha256";else fail=1;fi
record platform-reference-build tests/reference/stdlib_platform_reference.cpp g++ -std=c++23 -O2 -Wall -Wextra -Werror -pthread tests/reference/stdlib_platform_reference.cpp -L"$out" -lstdlib_platform -o "$out/platform-reference" || true
record platform-reference tests/reference/stdlib_platform_reference.cpp timeout 60 "$out/platform-reference" || true
record channel-reference-build tests/reference/stdlib_channel_reference.cpp g++ -std=c++23 -O2 -Wall -Wextra -Werror -pthread tests/reference/stdlib_channel_reference.cpp -L"$out" -lstdlib_platform -o "$out/channel-reference" || true
record channel-reference tests/reference/stdlib_channel_reference.cpp timeout 120 "$out/channel-reference" || true
for ref in file csv_numeric socket_ready vcf_header; do
 record "$ref-reference-build" "tests/reference/stdlib_${ref}_reference.cpp" g++ -std=c++23 -O2 -Wall -Wextra -Werror -pthread "tests/reference/stdlib_${ref}_reference.cpp" -L"$out" -lstdlib_platform -o "$out/$ref-reference" || true
 record "$ref-reference" "tests/reference/stdlib_${ref}_reference.cpp" timeout 120 "$out/$ref-reference" || true
 if [[ -f "$out/$ref-reference" ]];then sha256sum "$out/$ref-reference" >> "$out/EXECUTABLES.sha256";else fail=1;fi
done
record complex-reference-build tests/reference/stdlib_complex_canonical_reference.cpp g++ -std=c++23 -O2 -Wall -Wextra -Werror tests/reference/stdlib_complex_canonical_reference.cpp -o "$out/complex-reference" || true
if [[ -f "$out/complex-reference" ]];then sha256sum "$out/complex-reference" >> "$out/EXECUTABLES.sha256";else fail=1;fi
record workflow-reference-build tests/reference/stdlib_science_workflow_reference.cpp g++ -std=c++23 -O2 -Wall -Wextra -Werror tests/reference/stdlib_science_workflow_reference.cpp -o "$out/workflow-reference" || true
if [[ -f "$out/workflow-reference" ]];then sha256sum "$out/workflow-reference" >> "$out/EXECUTABLES.sha256";else fail=1;fi
record oracle-build tests/reference/stdlib_hyper_metrics.cpp g++ -std=c++23 -O2 -Wall -Wextra -Werror tests/reference/stdlib_hyper_metrics.cpp -o "$out/hyper-oracle" || true
for artifact in libstdlib_platform.so platform-reference channel-reference hyper-oracle; do
 if [[ -f "$out/$artifact" ]]; then sha256sum "$out/$artifact" >> "$out/EXECUTABLES.sha256"; else fail=1;fi
done
record vcf-reference-build tests/reference/stdlib_vcf_reference.cpp g++ -std=c++23 -O2 -Wall -Wextra -Werror tests/reference/stdlib_vcf_reference.cpp -o "$out/vcf-reference" || true
record vcf-reference tests/reference/stdlib_vcf_reference.cpp timeout 30 "$out/vcf-reference" || true
if [[ -f "$out/vcf-reference" ]]; then sha256sum "$out/vcf-reference" >> "$out/EXECUTABLES.sha256";fi
export STDLIB_HYPER_ORACLE="$out/hyper-oracle"
record operational-controls scripts/stdlib node --test scripts/stdlib/test_hyper_gate.mjs scripts/stdlib/test_run_full_suite.mjs scripts/stdlib/test_reliability_status.mjs scripts/stdlib/test_stdlib_e2e_diagnostics.mjs scripts/stdlib/test_emit_stdlib_e2e_result.mjs tools/science_boundary/core_test.mjs tools/science_boundary/closure_test.mjs tools/science_boundary/attestor_test.mjs tools/science_boundary/policy_claims_test.mjs || true
for src in tests/stdlib/massive/test_{complex_canonical,vcf_header,bytes,dynamic_collections,dynamic_heap_set,binary,platform,channel,native_file,native_csv,time,vcf,json_dom,dynamic_fasta,dynamic_gff,csv_checked_numeric,poll_executor,poll_socket,png_checked}.sio tests/stdlib/sync/test_{atomic,mutex}_e2e.sio examples/atomic_demo.sio tests/stdlib/math/test_core_e2e.sio tests/stdlib/complex/test_complex.sio tests/stdlib/complex/test_complex_deep_stdlib.sio tests/stdlib/regex/test_regex.sio tests/stdlib/profile/test_profiler_e2e.sio; do
 name=$(basename "$src" .sio)
 record "$name-check" "$src" timeout 180 bin/souc check "$src" || continue
 record "$name-compile" "$src" timeout 180 bin/souc compile "$src" -o "$out/$name.elf" || continue
 sha256sum "$out/$name.elf" >> "$out/EXECUTABLES.sha256"
 record "$name-native" "$src" timeout 120 "$out/$name.elf" || continue
done


record complex-concordance tests/reference/stdlib_complex_canonical_reference.cpp timeout 60 "$out/complex-reference" "$out/test_complex_canonical-native.stdout" || true

for kind in workflow report_check;do
 src="examples/stdlib_science_$kind.sio"
 record "$kind-check" "$src" timeout 180 bin/souc check "$src" || continue
 record "$kind-compile" "$src" timeout 180 bin/souc compile "$src" -o "$out/$kind.elf" || continue
 sha256sum "$out/$kind.elf" >> "$out/EXECUTABLES.sha256"
done
record scientific-workflow scripts/stdlib/science_workflow_gate.mjs node scripts/stdlib/science_workflow_gate.mjs --out-dir "$out/science-artifacts" --producer "$out/workflow.elf" --checker "$out/report_check.elf" --reference "$out/workflow-reference" || true

for control in private linear; do
 src="tests/stdlib/massive/byte_buffer_${control}_probe.sio"
 rc=0
 timeout 180 bin/souc check "$src" > "$out/$control-negative.stdout" 2> "$out/$control-negative.stderr" || rc=$?
 expected=E259
 if [[ "$control" == linear ]]; then expected=E039; fi
 result=0
 if [[ "$rc" == 0 ]] || ! grep -q "$expected" "$out/$control-negative."{stdout,stderr}; then result=1;fail=1;fi
 printf '%s\t%s\t%s\n' "$control-negative" "$src" "$result" >> "$out/RESULTS.tsv"
done
before=docs/evidence/stdlib-massive-2026-10-26/binary_before.sio
if record binary-before-compile "$before" timeout 180 bin/souc compile "$before" -o "$out/binary-before.elf"; then
 rc=0
 timeout 30 "$out/binary-before.elf" > "$out/binary-before-native.stdout" 2> "$out/binary-before-native.stderr" || rc=$?
 result=0
 if [[ "$rc" != 1 ]]; then result=1;fail=1;fi
 printf 'binary-before-expected-failure\t%s\t%s\n' "$before" "$result" >> "$out/RESULTS.tsv"
 echo "$rc" > "$out/binary-before-native.exit"
fi

sha256sum -c "$out/INPUTS.sha256" > "$out/INPUTS.final.verify" || fail=1
sha256sum -c "$out/COMPILER.sha256" > "$out/COMPILER.final.verify" || fail=1
sha256sum -c "$out/EXECUTABLES.sha256" > "$out/EXECUTABLES.final.verify" || fail=1
echo "$fail" > "$out/STATUS"
if [[ "$fail" == 0 ]]; then echo STDLIB_MASSIVE_GATE_OK; else echo STDLIB_MASSIVE_GATE_FAIL; fi
exit "$fail"
