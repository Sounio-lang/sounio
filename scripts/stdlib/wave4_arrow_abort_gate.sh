#!/usr/bin/env bash
# Explicit Arrow abort versus strict finish/close, with real close fault injection.
set -euo pipefail
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
 [[ "$r" != 0 ]] || echo WAVE4_ARROW_ABORT_GATE_OK
 exit "$r"
}
trap finish EXIT
(
 cd "$ROOT"
 find stdlib -type f -print0
 printf '%s\0' tests/stdlib/wave4-science-pipeline/arrow_abort.sio tests/stdlib/arrow/test_arrow_native_stream.sio runtime/stdlib_platform.cpp scripts/stdlib/wave4_arrow_abort_gate.sh scripts/ci/souc-native-wrapper.sh
) | sort -zu | while IFS= read -r -d '' p;do sha256sum "$ROOT/$p";done > "$OUT/SOURCES.sha256"
sha256sum "$RAW" "$(command -v "$CXX")" "$SDK/lib/x86_64-linux-gnu/libarrow.so.2600.0.0" > "$OUT/TOOLS.sha256"
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
 ldd "$tool" >> "$OUT/CXX_TOOLS.ldd"
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


phase runtime build 180 "$CXX" -std=c++23 -O2 -Wall -Wextra -Werror -shared -fPIC -pthread "$ROOT/runtime/stdlib_platform.cpp" -o "$OUT/libstdlib_platform.so"
sha256sum "$OUT/libstdlib_platform.so" >> "$OUT/EXECUTABLES.sha256"
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
find "$OUT/runs" "$OUT/inputs" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/ARTIFACTS.sha256"
