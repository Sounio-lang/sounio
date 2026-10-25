#!/usr/bin/env bash
# Operational protocol tests only: the C++ fixture copies an ELF; no Sounio
# numerical source is compiled or scientific result recalculated.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=$(mktemp -d "${TMPDIR:-/tmp}/sounio-native-phases.XXXXXX")
T=$WORK/tree
mkdir -p "$T/scripts/dev" "$T/scripts/ci" "$T/scripts/lib" "$T/tests/run-pass" "$T/stdlib"
trap 'code=$?;printf "%s\n" "$code" > "$WORK/STATUS";echo "operational_receipts=$WORK"' EXIT
cp "$ROOT/scripts/dev/run_sio_test_suite_v2.sh" "$T/scripts/dev/"
ln -s run_sio_test_suite_v2.sh "$T/scripts/dev/run_sio_test_suite.sh"
cp "$ROOT/scripts/run_sio_test_suite.sh" "$T/scripts/"
cp "$ROOT/scripts/ci/souc-native-wrapper.sh" "$T/scripts/ci/"
cp "$ROOT/scripts/lib/resolve_souc.sh" "$ROOT/scripts/lib/native_test_phases.sh" "$T/scripts/lib/"
cat > "$WORK/fixture.cpp" <<'CPP'
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
namespace f=std::filesystem;
void line(const char* key,const std::string& value) {
 const char* p=std::getenv(key);if(!p)return;
 int fd=open(p,O_WRONLY|O_CREAT|O_APPEND,0600);
 if(fd<0)std::exit(96);
 const auto text=value+"\n";if(write(fd,text.data(),text.size())!=static_cast<ssize_t>(text.size()))std::exit(97);close(fd);
}
int main(int argc,char**argv) {
 std::ifstream cfg(std::string(argv[0])+".mode");std::string kind;
 if(std::getline(cfg,kind)) {
  cfg.close();f::remove(std::string(argv[0])+".mode");
  line("STUB_NATIVE_LOG",kind+" "+argv[0]);
  if(kind=="native_timeout")std::this_thread::sleep_for(std::chrono::seconds(35));
  if(kind=="native_failure"){std::cout<<"PHASE_EXPECTED_FAILURE\n";return 7;}
  if(kind=="marker_missing"){std::cout<<"NATIVE_WITHOUT_EXPECTED_MARKER\n";return 0;}
  if(kind=="mutate") {
   f::remove(argv[0]);f::copy_file(std::getenv("STUB_COMPILER_BIN"),argv[0]);
   std::ofstream out(argv[0],std::ios::binary|std::ios::app);out.put('x');
  }
  std::cout<<"PHASE_NATIVE_OK\n";return 0;
 }
 if(argc==2&&std::string(argv[1])=="--version"){std::cout<<"Madaros v0.80.0 -- the Sounio self-hosted compiler\n";return 0;}
 if(argc!=4||std::string(argv[2])!="-o")return 93;
 kind=f::path(argv[1]).stem().string();
 line("STUB_CALL_LOG",kind+" "+argv[3]);
 if(kind=="compile_timeout")std::this_thread::sleep_for(std::chrono::seconds(35));
 if(kind=="missing")return 0;
 if(kind=="badelf"){std::ofstream out(argv[3]);out<<"#!/bin/sh\necho BADSCRIPT_EXEC >> \"$STUB_NATIVE_LOG\"\n";return 0;}
 f::copy_file(argv[0],argv[3],f::copy_options::overwrite_existing);
 const auto runtime=(kind=="xfail"||kind=="xfail_wrong")?"native_failure":kind=="xfail_timeout"?"native_timeout":kind=="compiler_marker"?"marker_missing":kind;
 std::ofstream(std::string(argv[3])+".mode")<<runtime<<"\n";
 if(kind=="source_mutate"){std::ofstream(argv[1],std::ios::app)<<"// changed during synthetic compile\n";}
 if(kind=="error0"){std::cerr<<"error[E900]: synthetic compiler diagnostic\n";return 0;}
 if(kind=="bad_rc"){std::cerr<<"compiler deliberately returns7 after emitting ELF\n";return 7;}
 if(kind=="compiler_marker")std::cout<<"PHASE_NATIVE_OK\n";
 return 0;
}
CPP
g++ -std=c++23 -O2 -Wall -Wextra -Werror "$WORK/fixture.cpp" -o "$WORK/fixture-compiler"
sha256sum "$WORK/fixture.cpp" "$WORK/fixture-compiler" > "$WORK/FIXTURE.sha256"
compiler_sha=$(sha256sum "$WORK/fixture-compiler" | cut -d' ' -f1)
printf 'case\tharness_rc\tcompile_calls\tnative_calls\n' > "$WORK/RESULTS.tsv"
make_case() {
 local name=$1;shift
 { printf '%s\n' '//@ run-pass' '//@ requires: madaros' '//@ expect-stdout: PHASE_NATIVE_OK';printf '%s\n' "$@";printf '%s\n' '// SYNTHETIC_METADATA_ONLY fixture; not scientific Sounio.'; } > "$T/tests/run-pass/$name.sio"
}
for name in good missing badelf bad_rc error0 compiler_marker marker_missing mutate source_mutate concurrent0 concurrent1 concurrent2 concurrent3;do make_case "$name";done
make_case compile_timeout
make_case native_timeout '//@ timeout: 1'
make_case xfail '//@ known-failure' '//@ xfail-exit: 7' '//@ xfail-expect: PHASE_EXPECTED_FAILURE'
make_case xfail_wrong '//@ known-failure' '//@ xfail-exit: 8' '//@ xfail-expect: PHASE_EXPECTED_FAILURE'
make_case xfail_timeout '//@ timeout: 1' '//@ known-failure' '//@ xfail-exit: 124' '//@ xfail-expect: phase=native'
invoke() {
 local label=$1 mode=$2 expected=$3;shift 3
 local dir=$WORK/$label;mkdir -p "$dir/phases" "$dir/results"
 printf '%s\n' "$@" > "$dir/test-list.txt"
 local rc=0
 env -u SOUNIO_TEST_NATIVE_BIN -u SOUC_BIN -u SOUC_NATIVE_BIN \
  SOUNIO_MADAROS_AVAILABLE=1 SOUNIO_SOUC_RAW_MODE=modular \
  SOUNIO_TEST_SOUC_BIN="$WORK/fixture-compiler" \
  SOUNIO_TEST_NATIVE_PHASES="$mode" \
  SOUNIO_TEST_PHASE_COMPILER="$WORK/fixture-compiler" \
  SOUNIO_TEST_PHASE_COMPILER_SHA256="$compiler_sha" \
  SOUNIO_TEST_NATIVE_PHASE_OUT="$dir/phases" SOUNIO_TEST_RESULTS_DIR="$dir/results" \
  STUB_CALL_LOG="$dir/compile.calls" STUB_NATIVE_LOG="$dir/native.calls" \
  STUB_COMPILER_BIN="$WORK/fixture-compiler" \
  bash "$T/scripts/run_sio_test_suite.sh" --test-list "$dir/test-list.txt" --jobs 4 \
   > "$dir/harness.stdout" 2> "$dir/harness.stderr" || rc=$?
 [[ $rc == "$expected" ]] || { cat "$dir/harness.stdout" "$dir/harness.stderr";echo "wrong harness rc: $label expected=$expected actual=$rc" >&2;exit 1; }
 local calls=0 natives=0
 [[ ! -f $dir/compile.calls ]] || calls=$(wc -l < "$dir/compile.calls")
 [[ ! -f $dir/native.calls ]] || natives=$(wc -l < "$dir/native.calls")
 printf '%s\t%s\t%s\t%s\n' "$label" "$rc" "$calls" "$natives" >> "$WORK/RESULTS.tsv"
}
# Default contract stays on wrapper run and compiles exactly once.
invoke legacy-good 0 0 tests/run-pass/good.sio
# Before control: the unchanged wrapper masks compiler rc7 if an ELF was emitted.
invoke legacy-rc-mask 0 0 tests/run-pass/bad_rc.sio
invoke split-good changed-tests-v1 0 tests/run-pass/good.sio
for name in missing badelf bad_rc error0 compiler_marker marker_missing mutate source_mutate;do
 invoke "$name" changed-tests-v1 1 "tests/run-pass/$name.sio"
done
invoke xfail changed-tests-v1 0 tests/run-pass/xfail.sio
invoke xfail-wrong changed-tests-v1 1 tests/run-pass/xfail_wrong.sio
invoke native-timeout changed-tests-v1 1 tests/run-pass/native_timeout.sio
invoke xfail-timeout changed-tests-v1 0 tests/run-pass/xfail_timeout.sio
invoke concurrent changed-tests-v1 0 tests/run-pass/concurrent0.sio tests/run-pass/concurrent1.sio tests/run-pass/concurrent2.sio tests/run-pass/concurrent3.sio
invoke compile-timeout changed-tests-v1 1 tests/run-pass/compile_timeout.sio
for label in missing badelf bad_rc error0 source_mutate compile-timeout;do
 [[ ! -s $WORK/$label/native.calls ]] || { echo "invalid compile was executed: $label" >&2;exit 1; }
done
for label in legacy-good legacy-rc-mask split-good missing badelf bad_rc error0 compiler_marker marker_missing mutate source_mutate xfail xfail-wrong native-timeout xfail-timeout compile-timeout;do
 [[ $(wc -l < "$WORK/$label/compile.calls") == 1 ]]
done
[[ $(wc -l < "$WORK/concurrent/compile.calls") == 4 && $(wc -l < "$WORK/concurrent/native.calls") == 4 ]]
[[ $(find "$WORK/concurrent/phases" -mindepth 1 -maxdepth 1 -type d | wc -l) == 4 ]]
grep -q 'phase=native' "$WORK/native-timeout/results/result_1.json"
grep -q 'phase=compile' "$WORK/compile-timeout/results/result_1.json"
grep -q '"status":"xfail"' "$WORK/xfail/results/result_1.json"
grep -q '"status":"xfail"' "$WORK/xfail-timeout/results/result_1.json"
# Explicit unsupported modes and wrong compiler identities refuse before compile.
mkdir -p "$WORK/refusal/phases"
for mode in unsupported changed-tests-v1;do
 rc=0
 env SOUNIO_TEST_SOUC_BIN="$WORK/fixture-compiler" SOUNIO_MADAROS_AVAILABLE=1 \
  SOUNIO_SOUC_RAW_MODE=modular SOUNIO_TEST_NATIVE_PHASES="$mode" \
  SOUNIO_TEST_PHASE_COMPILER="$WORK/fixture-compiler" SOUNIO_TEST_PHASE_COMPILER_SHA256="$(printf '%064d' 0)" \
  SOUNIO_TEST_NATIVE_PHASE_OUT="$WORK/refusal/phases" STUB_CALL_LOG="$WORK/refusal/compile.calls" \
  bash "$T/scripts/run_sio_test_suite.sh" --test-list "$WORK/split-good/test-list.txt" > "$WORK/refusal/$mode.stdout" 2> "$WORK/refusal/$mode.stderr" || rc=$?
 [[ $rc == 2 ]]
done
# A correct compiler digest does not authorize an unsupported compiler mode.
rc=0
env SOUNIO_TEST_SOUC_BIN="$WORK/fixture-compiler" SOUNIO_MADAROS_AVAILABLE=1 \
 SOUNIO_SOUC_RAW_MODE=lean SOUNIO_TEST_NATIVE_PHASES=changed-tests-v1 \
 SOUNIO_TEST_PHASE_COMPILER="$WORK/fixture-compiler" SOUNIO_TEST_PHASE_COMPILER_SHA256="$compiler_sha" \
 SOUNIO_TEST_NATIVE_PHASE_OUT="$WORK/refusal/phases" STUB_CALL_LOG="$WORK/refusal/compile.calls" \
 bash "$T/scripts/run_sio_test_suite.sh" --test-list "$WORK/split-good/test-list.txt" > "$WORK/refusal/raw-mode.stdout" 2> "$WORK/refusal/raw-mode.stderr" || rc=$?
[[ $rc == 2 ]]
grep -q requires_explicit_raw_modular "$WORK/refusal/raw-mode.stderr"
cp "$WORK/fixture-compiler" "$WORK/alien-compiler"
rc=0
env SOUNIO_TEST_SOUC_BIN="$WORK/fixture-compiler" SOUNIO_MADAROS_AVAILABLE=1 \
 SOUNIO_SOUC_RAW_MODE=modular SOUNIO_TEST_NATIVE_PHASES=changed-tests-v1 \
 SOUNIO_TEST_PHASE_COMPILER="$WORK/alien-compiler" SOUNIO_TEST_PHASE_COMPILER_SHA256="$compiler_sha" \
 SOUNIO_TEST_NATIVE_PHASE_OUT="$WORK/refusal/phases" STUB_CALL_LOG="$WORK/refusal/compile.calls" \
 bash "$T/scripts/run_sio_test_suite.sh" --test-list "$WORK/split-good/test-list.txt" > "$WORK/refusal/binding.stdout" 2> "$WORK/refusal/binding.stderr" || rc=$?
[[ $rc == 2 ]]
grep -q compiler_binding_mismatch "$WORK/refusal/binding.stderr"
[[ ! -s $WORK/refusal/compile.calls ]]
[[ -z $(find "$WORK/legacy-good/phases" -mindepth 1 -print -quit) ]]
sha256sum -c "$WORK/FIXTURE.sha256" > "$WORK/FIXTURE.verify"
echo 'MADAROS_NATIVE_PHASE_OPERATIONAL_SELFTEST_PASS'
cat "$WORK/RESULTS.tsv"
