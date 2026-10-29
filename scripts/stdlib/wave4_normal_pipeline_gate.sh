#!/usr/bin/env bash
# Certified workers and bounded normal-probability CSV/Arrow pipeline.
set -euo pipefail
NORMAL_ORACLE_ONLY=${NORMAL_ORACLE_ONLY:-0}
NORMAL_ABORT_ONLY=${NORMAL_ABORT_ONLY:-0}
[[ $NORMAL_ORACLE_ONLY != 1 || $NORMAL_ABORT_ONLY != 1 ]] || exit 64
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
[[ $# == 1 && "$1" == /* && "$1" =~ ^/[A-Za-z0-9_./-]+$ ]] || exit 64
OUT=$(realpath -m -- "$1")
[[ ! -e "$OUT" ]] || { echo "fresh output required" >&2;exit 64; }
if [[ ${SLURM_JOB_ID:-} =~ ^[1-9][0-9]*$ ]];then AUTHORITY="slurm:$SLURM_JOB_ID"
elif [[ ${GITHUB_ACTIONS:-} == true && ${GITHUB_RUN_ID:-} =~ ^[1-9][0-9]*$ && ${GITHUB_JOB:-} =~ ^[A-Za-z0-9_.-]+$ ]];then AUTHORITY="github-actions:$GITHUB_RUN_ID:$GITHUB_JOB"
else echo "scheduled execution required" >&2;exit 64;fi
: "${MADAROS_RAW_BIN:?qualified raw compiler}"
: "${ARROW_CPP_ROOT:?Arrow C++26 SDK usr prefix}"
RAW=$(realpath -e "$MADAROS_RAW_BIN")
SDK=$(realpath -e "$ARROW_CPP_ROOT")
[[ -x "$RAW" && -f "$SDK/include/arrow/api.h" ]] || exit 64
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
mkdir "$OUT/inputs" "$OUT/runs"
printf '%s\n' "$AUTHORITY" > "$OUT/EXECUTION_AUTHORITY"
CXX=${CXX:-g++}
TIME_BIN=${TIME_BIN:-/usr/bin/time}
[[ "$TIME_BIN" == /* && -x "$TIME_BIN" ]] || exit 64
WRAPPER="$ROOT/scripts/ci/souc-native-wrapper.sh"
export SOUC_NATIVE_BIN="$RAW" SOUNIO_SOUC_BIN="$RAW" SOUNIO_SOUC_RAW_MODE=modular SOUNIO_STDLIB_PATH="$ROOT/stdlib"
export SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
export LD_LIBRARY_PATH="$OUT:$SDK/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ulimit -s 524288
ulimit -c 0
finish(){
 local r=$? changed=0
 trap - EXIT
 for name in SOURCES TOOLS EXECUTABLES INPUTS DYNAMIC ARTIFACTS SDK_HEADERS;do
  if [[ -s "$OUT/$name.sha256" ]];then sha256sum -c "$OUT/$name.sha256" > "$OUT/$name.final.verify" 2>&1 || changed=1;fi
 done
 [[ $changed == 0 ]] || r=1
 printf '%s\n' "$r" > "$OUT/STATUS"
 if [[ "$r" == 0 ]];then
  if [[ "$NORMAL_ORACLE_ONLY" == 1 ]];then echo WAVE4_NORMAL_ORACLE_ONLY_OK
  elif [[ "$NORMAL_ABORT_ONLY" == 1 ]];then echo WAVE4_ARROW_ABORT_ONLY_OK
  else echo WAVE4_NORMAL_PIPELINE_GATE_OK;fi
 fi
 exit "$r"
}
trap finish EXIT
(
 cd "$ROOT"
 find stdlib tests/stdlib/wave3-native-pool tests/stdlib/wave4-science-pipeline -type f -print0
 printf '%s\0' tests/stdlib/arrow/test_arrow_native_stream.sio tests/stdlib/massive/test_platform.sio tests/stdlib/massive/test_csv_stream.sio tests/stdlib/massive/test_deadline_socket.sio runtime/stdlib_platform.cpp scripts/stdlib/wave3_native_pool_gate.sh scripts/ci/souc-native-wrapper.sh tests/reference/wave3_native_pool_reference.cpp tests/reference/wave3_adc_pipeline_reference.cpp tests/reference/wave4_normal_pipeline_reference.cpp scripts/stdlib/wave4_normal_pipeline_gate.sh
) | sort -zu | while IFS= read -r -d '' p;do sha256sum "$ROOT/$p";done > "$OUT/SOURCES.sha256"
sha256sum "$RAW" "$TIME_BIN" "$(command -v "$CXX")" "$SDK/lib/x86_64-linux-gnu/libarrow.so.2600.0.0" > "$OUT/TOOLS.sha256"
"$CXX" --version > "$OUT/CXX_VERSION"
printf 'case\tphase\tstatus\n' > "$OUT/RESULTS.tsv"
: > "$OUT/EXECUTABLES.sha256"
: > "$OUT/INPUTS.sha256"
phase(){
 local name=$1 kind=$2 seconds=$3 r=0
 shift 3
 timeout "$seconds" "$@" > "$OUT/$name.$kind.stdout" 2> "$OUT/$name.$kind.stderr" || r=$?
 printf '%s\t%s\t%s\n' "$name" "$kind" "$r" >> "$OUT/RESULTS.tsv"
 return "$r"
}
compile_sio(){
 local name=$1 source=$2
 sha256sum "$source" >> "$OUT/INPUTS.sha256"
 phase "$name" check 300 bash "$WRAPPER" check "$source"
 grep -Fxq 'check: OK' "$OUT/$name.check.stdout"
 phase "$name" compile 600 bash "$WRAPPER" compile "$source" -o "$OUT/$name.elf"
 [[ -s "$OUT/$name.elf" && $(od -An -tx1 -N4 "$OUT/$name.elf"|tr -d ' \n') == 7f454c46 ]]
 chmod +x "$OUT/$name.elf"
 sha256sum "$OUT/$name.elf" >> "$OUT/EXECUTABLES.sha256"
}

# Seal the compiler driver/backend/linker and SDK headers before compiling.
for tool in "$(command -v "$CXX")" "$("$CXX" -print-prog-name=cc1plus)" "$("$CXX" -print-prog-name=collect2)" "$("$CXX" -print-prog-name=as)" "$("$CXX" -print-prog-name=ld)" "$("$CXX" -print-file-name=libstdc++.so)" "$("$CXX" -print-file-name=libgcc_s.so.1)";do
 if [[ "$tool" != */* ]];then tool=$(command -v "$tool");fi
 tool=$(realpath -e "$tool")
 sha256sum "$tool" >> "$OUT/TOOLS.sha256"
 if [[ $(od -An -tx1 -N4 "$tool"|tr -d ' \n') == 7f454c46 ]];then
  ldd "$tool" >> "$OUT/CXX_TOOLS.ldd"
 elif [[ $(head -c 2 "$tool") == '#!' ]];then
  printf 'HASHED_SCRIPT_DRIVER %s\n' "$tool" >> "$OUT/CXX_TOOLS.ldd"
 else echo "unsupported compiler tool format: $tool" >&2;exit 1;fi
done
! grep -q 'not found' "$OUT/CXX_TOOLS.ldd"
awk '/=> \//{print $3} /^[[:space:]]*\//{print $1}' "$OUT/CXX_TOOLS.ldd" | sort -u | xargs -r sha256sum > "$OUT/DYNAMIC.sha256"
find "$SDK/include" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/SDK_HEADERS.sha256"
dynamic_manifest(){
 local executable=$1 name=$2
 ldd "$executable" > "$OUT/$name.ldd"
 ! grep -q 'not found' "$OUT/$name.ldd"
 awk '/=> \//{print $3} /^[[:space:]]*\//{print $1}' "$OUT/$name.ldd" | sort -u | xargs -r sha256sum >> "$OUT/DYNAMIC.sha256"
}


phase normal-reference build 180 "$CXX" -std=c++23 -O2 -Wall -Wextra -Werror -pthread -isystem "$SDK/include" "$ROOT/tests/reference/wave4_normal_pipeline_reference.cpp" -L"$SDK/lib/x86_64-linux-gnu" -larrow -o "$OUT/normal-reference"
sha256sum "$OUT/normal-reference" >> "$OUT/EXECUTABLES.sha256"
dynamic_manifest "$OUT/normal-reference" normal-reference
phase oracle-positive verify 120 "$OUT/normal-reference" oracle-fixture "$OUT/runs/oracle" 257
sha256sum "$OUT/runs/oracle/input.csv" >> "$OUT/INPUTS.sha256"
for mutation in wrong-value wrong-order;do
 phase "oracle-$mutation" generate 60 "$OUT/normal-reference" mutate "$OUT/runs/oracle/oracle.arrow" "$OUT/oracle-$mutation.arrow" "$mutation"
 r=0
 phase "oracle-$mutation" judge-reject 60 "$OUT/normal-reference" verify "$OUT/runs/oracle/input.csv" "$OUT/oracle-$mutation.arrow" || r=$?
 [[ $r == 1 ]]
done
for mutation in truncated trailing no-eos extra-eos;do
 cp "$OUT/runs/oracle/oracle.arrow" "$OUT/oracle-$mutation.arrow"
 case "$mutation" in
 truncated) truncate -s -1 "$OUT/oracle-$mutation.arrow";;
 trailing) printf x >> "$OUT/oracle-$mutation.arrow";;
 no-eos) truncate -s -8 "$OUT/oracle-$mutation.arrow";;
 extra-eos) printf '\377\377\377\377\0\0\0\0' >> "$OUT/oracle-$mutation.arrow";;
 esac
 r=0
 phase "oracle-$mutation" judge-reject 60 "$OUT/normal-reference" verify "$OUT/runs/oracle/input.csv" "$OUT/oracle-$mutation.arrow" || r=$?
 [[ $r == 1 ]]
done
if [[ $NORMAL_ORACLE_ONLY == 1 ]];then
 (find "$OUT/runs" "$OUT/inputs" -type f -print0;find "$OUT" -maxdepth 1 -name "*.arrow" -type f -print0) | sort -zu | xargs -0 sha256sum > "$OUT/ARTIFACTS.sha256"
 printf '%s\n' ORACLE_ONLY_NO_SOUNIO_PIPELINE_QUALIFICATION > "$OUT/PARTIAL"
 exit 0
fi
phase runtime build 180 "$CXX" -std=c++23 -O2 -Wall -Wextra -Werror -shared -fPIC -pthread "$ROOT/runtime/stdlib_platform.cpp" -o "$OUT/libstdlib_platform.so"
sha256sum "$OUT/libstdlib_platform.so" >> "$OUT/EXECUTABLES.sha256"
cat > "$OUT/inputs/spawn_trace.cpp" <<'CPP'
#include <pthread.h>
#include <dlfcn.h>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
static bool target_process(){
 const char* target=std::getenv("SOUNIO_SPAWN_TARGET");
 if(!target)return false;
 char path[4096];const auto n=::readlink("/proc/self/exe",path,sizeof(path)-1);
 if(n<0||static_cast<std::size_t>(n)>=sizeof(path)-1)return false;
 path[n]=0;return std::strcmp(path,target)==0;
}
static void event(char value){
 if(!target_process())return;
 if(const char* path=std::getenv("SOUNIO_SPAWN_TRACE")){
  const int fd=::open(path,O_WRONLY|O_CREAT|O_APPEND|O_CLOEXEC,0600);
  if(fd>=0){const auto ignored=::write(fd,&value,1);(void)ignored;::close(fd);}
 }
}
__attribute__((constructor))static void loaded(){event('L');}
extern "C" int pthread_create(pthread_t* id,const pthread_attr_t* attr,void*(*start)(void*),void* arg){
 using Fn=int(*)(pthread_t*,const pthread_attr_t*,void*(*)(void*),void*);
 static Fn real=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"pthread_create"));
 if(!real)return 38;
 const int rc=real(id,attr,start,arg);
 if(rc==0)event('S');
 return rc;
}
CPP
sha256sum "$OUT/inputs/spawn_trace.cpp" >> "$OUT/INPUTS.sha256"
phase spawn-trace build 60 "$CXX" -std=c++23 -O2 -Wall -Wextra -Werror -shared -fPIC -pthread "$OUT/inputs/spawn_trace.cpp" -ldl -o "$OUT/spawn-trace.so"
sha256sum "$OUT/spawn-trace.so" >> "$OUT/EXECUTABLES.sha256"

trace_check(){
 local path=$1 expected=$2
 [[ -f "$path" && $(tr -cd L < "$path") == L && $(tr -cd S < "$path"|wc -c) -eq "$expected" && $(tr -d LS < "$path"|wc -c) -eq 0 ]]
}
: > "$OUT/inputs/trace-empty"
printf L > "$OUT/inputs/trace-valid-zero"
printf LSSSS > "$OUT/inputs/trace-valid-four"
printf SSSS > "$OUT/inputs/trace-no-load"
printf LLSSSS > "$OUT/inputs/trace-two-loads"
printf LX > "$OUT/inputs/trace-unknown"
trace_check "$OUT/inputs/trace-valid-zero" 0
trace_check "$OUT/inputs/trace-valid-four" 4
for wrong in trace-empty trace-no-load trace-two-loads trace-unknown;do
 wanted=0
 case "$wrong" in trace-no-load|trace-two-loads) wanted=4;;esac
 if trace_check "$OUT/inputs/$wrong" "$wanted";then echo "spawn verifier accepted $wrong" >&2;exit 1;fi
done
printf 'spawn-trace-controls\tverify\t0\n' >> "$OUT/RESULTS.tsv"



# Explicit abort keeps stream validity independent from cleanup errno.
cat > "$OUT/inputs/close_eio.cpp" <<'CPP'
#include <dlfcn.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
extern "C" int close(int fd){
 using Fn=int(*)(int);static Fn real=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"close"));
 if(!real){errno=ENOSYS;return -1;}
 char proc[64],path[4096];std::snprintf(proc,sizeof(proc),"/proc/self/fd/%d",fd);
 const auto n=::readlink(proc,path,sizeof(path)-1);
 bool target=false;
 if(n>0){path[n]=0;const char* slash=std::strrchr(path,'/');const char* name=std::getenv("SOUNIO_FAIL_CLOSE_NAME");target=std::strcmp(slash?slash+1:path,name?name:"close-error.arrow")==0;}
 const int rc=real(fd);if(rc!=0||!target)return rc;
 if(::fcntl(fd,F_GETFD)!=-1||errno!=EBADF){errno=EPROTO;return -1;}
 constexpr char marker[]="INJECTED_CLOSE_EIO_DESCRIPTOR_CLOSED\n";
 const auto ignored=::write(STDERR_FILENO,marker,sizeof(marker)-1);(void)ignored;
 errno=EIO;return -1;
}
CPP
sha256sum "$OUT/inputs/close_eio.cpp" >> "$OUT/INPUTS.sha256"
phase close-eio build 60 "$CXX" -std=c++23 -O2 -Wall -Wextra -Werror -shared -fPIC "$OUT/inputs/close_eio.cpp" -ldl -o "$OUT/close-eio.so"
sha256sum "$OUT/close-eio.so" >> "$OUT/EXECUTABLES.sha256"
compile_sio arrow-abort "$ROOT/tests/stdlib/wave4-science-pipeline/arrow_abort.sio"
sed 's/expected_close:i64=0/expected_close:i64=5/' "$ROOT/tests/stdlib/wave4-science-pipeline/arrow_abort.sio" > "$OUT/inputs/arrow-abort-eio.sio"
compile_sio arrow-abort-eio "$OUT/inputs/arrow-abort-eio.sio"
for variant in normal eio;do
 mkdir "$OUT/runs/arrow-abort-$variant"
 if [[ $variant == normal ]];then
  (cd "$OUT/runs/arrow-abort-$variant";phase arrow-abort native 120 "$OUT/arrow-abort.elf")
  [[ $(head -1 "$OUT/arrow-abort.native.stdout") == 0 ]]
 else
  (cd "$OUT/runs/arrow-abort-$variant";phase arrow-abort-eio native 120 env LD_PRELOAD="$OUT/close-eio.so" "$OUT/arrow-abort-eio.elf")
  [[ $(head -1 "$OUT/arrow-abort-eio.native.stdout") == 5 ]]
  grep -Fxq INJECTED_CLOSE_EIO_DESCRIPTOR_CLOSED "$OUT/arrow-abort-eio.native.stderr"
 fi
 for incomplete in strict-unfinished abort-unopened abort-begun close-error;do
  [[ -f "$OUT/runs/arrow-abort-$variant/$incomplete.arrow" ]]
  [[ $(tail -c 8 "$OUT/runs/arrow-abort-$variant/$incomplete.arrow"|od -An -tx1|tr -d ' \n') != ffffffff00000000 ]]
 done
 [[ $(tail -c 8 "$OUT/runs/arrow-abort-$variant/strict-finished.arrow"|od -An -tx1|tr -d ' \n') == ffffffff00000000 ]]
done
mkdir "$OUT/runs/original-arrow"
sed "s|/tmp/sounio_wave3_arrow_|$OUT/runs/original-arrow/original_|g" "$ROOT/tests/stdlib/arrow/test_arrow_native_stream.sio" > "$OUT/inputs/original-arrow.sio"
sed "s|$OUT/runs/original-arrow/original_|/tmp/sounio_wave3_arrow_|g" "$OUT/inputs/original-arrow.sio" > "$OUT/inputs/original-arrow.reversed"
cmp "$ROOT/tests/stdlib/arrow/test_arrow_native_stream.sio" "$OUT/inputs/original-arrow.reversed"
compile_sio original-arrow "$OUT/inputs/original-arrow.sio"
phase original-arrow native 120 "$OUT/original-arrow.elf"
grep -Fxq ARROW_NATIVE_STREAM_OK "$OUT/original-arrow.native.stdout"
if [[ $NORMAL_ABORT_ONLY == 1 ]];then
 find "$OUT/runs" "$OUT/inputs" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/ARTIFACTS.sha256"
 printf '%s\n' ARROW_ABORT_COMPONENT_NO_PUBLIC_NORMAL_PIPELINE_QUALIFICATION > "$OUT/PARTIAL"
 exit 0
fi

# Every rejection must be semantic. Unsupported intrinsic names, parser errors,
# timeouts, crashes, or a leftover executable cannot qualify a security control.
worker_refusal(){
 local name=$1 source=$2 kind r log
 cp "$source" "$OUT/inputs/$name.sio"
 sha256sum "$OUT/inputs/$name.sio" >> "$OUT/INPUTS.sha256"
 for kind in check compile;do
  r=0
  if [[ $kind == check ]];then
   phase "$name" worker-check-refusal 300 bash "$WRAPPER" check "$OUT/inputs/$name.sio" || r=$?
   log="$OUT/$name.worker-check-refusal"
  else
   phase "$name" worker-compile-refusal 600 bash "$WRAPPER" compile "$OUT/inputs/$name.sio" -o "$OUT/$name.elf" || r=$?
   log="$OUT/$name.worker-compile-refusal"
  fi
  [[ $r == 1 && ! -e "$OUT/$name.elf" ]]
  cat "$log.stdout" "$log.stderr" > "$log.diagnostics"
  ! grep -Eq 'E137|PARSE_REFUSED|parse error|unknown identifier|undefined identifier' "$log.diagnostics"
  grep -Eq 'WorkerFn (capability|source closure) refused:|expected WorkerFn(I64|F64)' "$log.diagnostics"
 done
}
worker_refusals=0
for fixture in "$ROOT"/tests/stdlib/wave4-science-pipeline/refuse_*.sio.in;do
 [[ -s $fixture ]]
 worker_refusal "$(basename "$fixture" .sio.in)" "$fixture"
 worker_refusals=$((worker_refusals+1))
done
[[ $worker_refusals -ge 14 ]]
printf '%s\n' "$worker_refusals" > "$OUT/WORKER_REFUSAL_COUNT"

compile_sio thread "$ROOT/tests/stdlib/wave4-science-pipeline/thread.sio"
phase thread native 120 "$OUT/thread.elf"
grep -Fxq NATIVE_CERTIFIED_THREAD_OK "$OUT/thread.native.stdout"
compile_sio platform "$ROOT/tests/stdlib/massive/test_platform.sio"
phase platform native 120 "$OUT/platform.elf"
grep -Fxq MASSIVE_PLATFORM_OK "$OUT/platform.native.stdout"

compile_sio pool_f64 "$ROOT/tests/stdlib/wave4-science-pipeline/pool_f64.sio"
phase pool_f64 native 120 "$OUT/pool_f64.elf"
grep -Fxq NATIVE_CERTIFIED_F64_POOL_OK "$OUT/pool_f64.native.stdout"
compile_sio pool_i64 "$ROOT/tests/stdlib/wave3-native-pool/pool.sio"
phase pool_i64 native 120 "$OUT/pool_i64.elf"
grep -Fxq NATIVE_THREAD_POOL_OK "$OUT/pool_i64.native.stdout"
cp "$ROOT/tests/stdlib/wave4-science-pipeline/normal_driver.sio.in" "$OUT/inputs/driver.original.sio"
printf 'variant\tparallel\tworkers_per_kernel\tchunk\n' > "$OUT/SUBSTITUTIONS.tsv"
for variant in serial one two;do
 parallel=true;workers=1;chunk=3
 case "$variant" in serial) parallel=false;chunk=4096;;two) workers=2;chunk=17;;esac
 sed -e "s/parallel:true/parallel:$parallel/" -e "s/native_pool_reservation(\\&p,1,16)/native_pool_reservation(\\&p,$workers,16)/" -e "s/workers_per_kernel:1/workers_per_kernel:$workers/" -e "s/read_chunk_bytes:17/read_chunk_bytes:$chunk/" "$OUT/inputs/driver.original.sio" > "$OUT/inputs/$variant.sio"
 printf '%s\t%s\t%s\t%s\n' "$variant" "$parallel" "$workers" "$chunk" >> "$OUT/SUBSTITUTIONS.tsv"
 compile_sio "$variant" "$OUT/inputs/$variant.sio"
done
run_file(){
 local name=$1 elf=$2 count=$3 spawned=$4
 mkdir "$OUT/runs/$name"
 phase "$name" generate 60 "$OUT/normal-reference" generate "$OUT/runs/$name" "$count"
 sha256sum "$OUT/runs/$name/input.csv" >> "$OUT/INPUTS.sha256"
 : > "$OUT/$name.spawns"
 (cd "$OUT/runs/$name";phase "$name" native 600 env LD_PRELOAD="$OUT/spawn-trace.so" SOUNIO_SPAWN_TRACE="$OUT/$name.spawns" SOUNIO_SPAWN_TARGET="$elf" "$TIME_BIN" -v -o "$OUT/$name.resource" "$elf")
 trace_check "$OUT/$name.spawns" "$spawned"
 grep -Fxq NORMAL_PIPELINE_RESOURCES_CLOSED "$OUT/$name.native.stdout"
 grep -Fxq NORMAL_CSV_ARROW_OK "$OUT/$name.native.stdout"
 phase "$name" verify 120 "$OUT/normal-reference" verify "$OUT/runs/$name/input.csv" "$OUT/runs/$name/output.arrow"
}
run_file serial-small "$OUT/serial.elf" 257 0
run_file one-small "$OUT/one.elf" 257 4
run_file two-small "$OUT/two.elf" 257 8
cmp "$OUT/runs/serial-small/output.arrow" "$OUT/runs/one-small/output.arrow"
cmp "$OUT/runs/one-small/output.arrow" "$OUT/runs/two-small/output.arrow"
run_file serial-scale "$OUT/serial.elf" 1000000 0
run_file one-scale "$OUT/one.elf" 1000000 4
cmp "$OUT/runs/serial-scale/input.csv" "$OUT/runs/one-scale/input.csv"
cmp "$OUT/runs/serial-scale/output.arrow" "$OUT/runs/one-scale/output.arrow"
for mutation in wrong-value wrong-order;do
 phase "$mutation" generate 60 "$OUT/normal-reference" mutate "$OUT/runs/one-small/output.arrow" "$OUT/$mutation.arrow" "$mutation"
 r=0
 phase "$mutation" judge-reject 60 "$OUT/normal-reference" verify "$OUT/runs/one-small/input.csv" "$OUT/$mutation.arrow" || r=$?
 [[ $r == 1 ]]
done
negative_input(){
 local name=$1 expected=$2 contents=$3 r=0
 mkdir "$OUT/runs/$name"
 printf '%b' "$contents" > "$OUT/runs/$name/input.csv"
 sha256sum "$OUT/runs/$name/input.csv" >> "$OUT/INPUTS.sha256"
 (cd "$OUT/runs/$name";phase "$name" native-refusal 120 "$OUT/two.elf") || r=$?
 [[ $r == "$expected" ]]
 [[ $(head -1 "$OUT/$name.native-refusal.stdout") == "$expected" && $(sed -n '2p' "$OUT/$name.native-refusal.stdout") == 0 ]]
 [[ $(tail -c 8 "$OUT/runs/$name/output.arrow"|od -An -tx1|tr -d ' \n') != ffffffff00000000 ]]
 grep -Fxq NORMAL_PIPELINE_RESOURCES_CLOSED "$OUT/$name.native-refusal.stdout"
 ! grep -Fxq NORMAL_CSV_ARROW_OK "$OUT/$name.native-refusal.stdout"
}
negative_input header 22 'id,z_milli\n0,1\n'
negative_input columns 22 'sample_id,z_milli\n0,1,2\n'
negative_input order 22 'sample_id,z_milli\n1,1\n'
negative_input fraction 22 'sample_id,z_milli\n0,0.5\n'
negative_input exponent 22 'sample_id,z_milli\n0,1e3\n'
negative_input sign 22 'sample_id,z_milli\n0,+\n'
negative_input range 34 'sample_id,z_milli\n0,40001\n'
negative_input min-range 34 'sample_id,z_milli\n0,-40001\n'
negative_input overflow 34 'sample_id,z_milli\n0,9223372036854775808\n'
negative_input utf8 84 'sample_id,z_milli\n0,\377\n'
negative_input quoted-eof 22 'sample_id,z_milli\n0,"2'
negative_input maxrow 105 "sample_id,z_milli\n0,$(printf '%0130d' 1)\n"

mkdir "$OUT/runs/input-close-error"
printf 'sample_id,z_milli\n0,"2' > "$OUT/runs/input-close-error/input.csv"
sha256sum "$OUT/runs/input-close-error/input.csv" >> "$OUT/INPUTS.sha256"
r=0
(cd "$OUT/runs/input-close-error";phase input-close-error native-refusal 120 env LD_PRELOAD="$OUT/close-eio.so" SOUNIO_FAIL_CLOSE_NAME=output.arrow "$OUT/two.elf") || r=$?
[[ $r == 22 && $(sed -n '1p' "$OUT/input-close-error.native-refusal.stdout") == 22 && $(sed -n '2p' "$OUT/input-close-error.native-refusal.stdout") == 5 ]]
grep -Fxq NORMAL_PIPELINE_RESOURCES_CLOSED "$OUT/input-close-error.native-refusal.stdout"
grep -Fxq INJECTED_CLOSE_EIO_DESCRIPTOR_CLOSED "$OUT/input-close-error.native-refusal.stderr"
[[ $(tail -c 8 "$OUT/runs/input-close-error/output.arrow"|od -An -tx1|tr -d ' \n') != ffffffff00000000 ]]

cp "$ROOT/tests/stdlib/wave4-science-pipeline/normal_tcp_driver.sio.in" "$OUT/inputs/tcp.sio"
compile_sio tcp "$OUT/inputs/tcp.sio"
tcp_run(){
 local name=$1 elf=$2 input=$3 chunk=$4 delay=$5 mode=$6 expected=$7 port='' producer_rc=0 peer_rc=0
 mkdir "$OUT/runs/$name"
 cp "$input" "$OUT/runs/$name/input.csv"
 sha256sum "$OUT/runs/$name/input.csv" >> "$OUT/INPUTS.sha256"
 : > "$OUT/$name.spawns"
 (cd "$OUT/runs/$name";phase "$name" native 600 env LD_PRELOAD="$OUT/spawn-trace.so" SOUNIO_SPAWN_TRACE="$OUT/$name.spawns" SOUNIO_SPAWN_TARGET="$elf" "$TIME_BIN" -v -o "$OUT/$name.resource" "$elf") &
 local producer=$!
 for attempt in {1..250};do
  if grep -Fxq TCP_LISTENING "$OUT/$name.native.stdout" 2>/dev/null;then
   port=$(sed -n '2p' "$OUT/$name.native.stdout")
   [[ $port =~ ^[0-9]+$ ]] && break
  fi
  if ! kill -0 "$producer" 2>/dev/null;then break;fi
  sleep 0.02
 done
 if [[ ! $port =~ ^[0-9]+$ ]];then wait "$producer" || true;echo "missing TCP port" >&2;return 1;fi
 phase "$name" peer 600 "$OUT/normal-reference" peer "$port" "$OUT/runs/$name/input.csv" "$chunk" "$delay" "$mode" || peer_rc=$?
 wait "$producer" || producer_rc=$?
 trace_check "$OUT/$name.spawns" 4
 if [[ $expected == 0 ]];then
  [[ $producer_rc == 0 && $peer_rc == 0 ]]
  grep -Fxq NORMAL_TCP_ARROW_OK "$OUT/$name.native.stdout"
  phase "$name" verify 120 "$OUT/normal-reference" verify "$OUT/runs/$name/input.csv" "$OUT/runs/$name/output.arrow"
 else
  [[ $producer_rc == "$expected" ]]
  [[ $(sed -n '3p' "$OUT/$name.native.stdout") == "$expected" && $(sed -n '4p' "$OUT/$name.native.stdout") == 0 ]]
  [[ $(tail -c 8 "$OUT/runs/$name/output.arrow"|od -An -tx1|tr -d ' \n') != ffffffff00000000 ]]
  grep -Fxq NORMAL_TCP_RESOURCES_CLOSED "$OUT/$name.native.stdout"
  ! grep -Fxq NORMAL_TCP_ARROW_OK "$OUT/$name.native.stdout"
  if [[ $peer_rc != 0 ]];then [[ $peer_rc == 1 ]];grep -Eq 'errno=(32|104)' "$OUT/$name.peer.stderr";fi
 fi
}
tcp_run tcp-small "$OUT/tcp.elf" "$OUT/runs/one-small/input.csv" 4096 0 normal 0
tcp_run tcp-short "$OUT/tcp.elf" "$OUT/runs/one-small/input.csv" 1 1 slow 0
tcp_run tcp-scale "$OUT/tcp.elf" "$OUT/runs/one-scale/input.csv" 4096 0 normal 0
cmp "$OUT/runs/one-small/output.arrow" "$OUT/runs/tcp-small/output.arrow"
cmp "$OUT/runs/one-small/output.arrow" "$OUT/runs/tcp-short/output.arrow"
cmp "$OUT/runs/one-scale/output.arrow" "$OUT/runs/tcp-scale/output.arrow"
sed 's/value+180000000000/value+10000000/' "$OUT/inputs/tcp.sio" > "$OUT/inputs/tcp-deadline.sio"
compile_sio tcp-deadline "$OUT/inputs/tcp-deadline.sio"
tcp_run tcp-deadline "$OUT/tcp-deadline.elf" "$OUT/runs/one-small/input.csv" 4096 250 hold 110
sed 's/deadline,33554432/deadline,16/' "$OUT/inputs/tcp.sio" > "$OUT/inputs/tcp-budget.sio"
compile_sio tcp-budget "$OUT/inputs/tcp-budget.sio"
tcp_run tcp-budget "$OUT/tcp-budget.elf" "$OUT/runs/one-small/input.csv" 4096 0 normal 105
printf 'sample_id,z_milli\n0,"12' > "$OUT/inputs/tcp-truncated.csv"
tcp_run tcp-truncated "$OUT/tcp.elf" "$OUT/inputs/tcp-truncated.csv" 1 0 normal 22
phase serial-resource-bound verify 60 "$OUT/normal-reference" resources "$OUT/serial-small.resource" "$OUT/serial-scale.resource" "$OUT/serial-scale.resource"
phase parallel-resource-bound verify 60 "$OUT/normal-reference" resources "$OUT/one-small.resource" "$OUT/one-scale.resource" "$OUT/tcp-scale.resource"
printf 'Maximum resident set size (kbytes): 1048576\n' > "$OUT/fake-growth.resource"
r=0
phase resource-bound judge-reject 60 "$OUT/normal-reference" resources "$OUT/one-small.resource" "$OUT/fake-growth.resource" "$OUT/tcp-scale.resource" || r=$?
[[ $r == 1 ]]
(find "$OUT/inputs" "$OUT/runs" -type f -print0;find "$OUT" -maxdepth 1 -name "*.arrow" -type f -print0) | sort -zu | xargs -0 sha256sum > "$OUT/ARTIFACTS.sha256"
sha256sum "$OUT/SUBSTITUTIONS.tsv" >> "$OUT/INPUTS.sha256"
printf '%s\n' WAVE4_NORMAL_PIPELINE_COMPLETE > "$OUT/COMPLETE"
