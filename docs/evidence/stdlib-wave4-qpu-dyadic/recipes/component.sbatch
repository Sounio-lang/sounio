#!/bin/bash
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027
R=$B/root-qpu-dyadic01
D=$R/receipts
mkdir "$D"
finish(){ r=$?;trap - EXIT;for m in SOURCE SDK RAW;do if [[ -s "$D/$m.sha256" ]];then sha256sum -c "$D/$m.sha256" >"$D/$m.final.verify" 2>&1||r=1;fi;done;echo "$r" >"$D/STATUS";exit "$r"; }
trap finish EXIT
export QPU_GCC_ROOT=$B/services/bigframe-abi07/gcc13-private/root
T=$QPU_GCC_ROOT
export PATH="$T/usr/bin:/usr/bin:/bin" LD_LIBRARY_PATH="$T/deps:$B/services/wide-counts33/input/dso"
export COMPILER_PATH="$T/usr/libexec/gcc/x86_64-linux-gnu/13:$T/usr/bin"
export LIBRARY_PATH="$T/usr/lib/gcc/x86_64-linux-gnu/13:$T/usr/lib/x86_64-linux-gnu:$T/lib/x86_64-linux-gnu"
export SOUNIO_STDLIB_PATH="$B/services/quantum-pair-library28/after/stdlib"
export NODE_BIN="$B/services-tools-recovered/tools/node"
cat >"$R/cxx" <<'SH'
#!/bin/bash
T=$QPU_GCC_ROOT
exec "$T/usr/bin/x86_64-linux-gnu-g++-13" --sysroot="$T" -B"$T/usr/libexec/gcc/x86_64-linux-gnu/13/" -B"$T/usr/bin/" -isystem "$T/usr/include/c++/13" -isystem "$T/usr/include/x86_64-linux-gnu/c++/13" -static-libstdc++ -static-libgcc "$@"
SH
chmod +x "$R/cxx"
export CXX="$R/cxx"
find "$R/input" "$SOUNIO_STDLIB_PATH" -type f -print0|sort -z|xargs -0 sha256sum >"$D/SOURCE.sha256"
find "$T" -type f -print0|sort -z|xargs -0 sha256sum >"$D/SDK.sha256"
sha256sum "$R/cxx" "$NODE_BIN" "$B/services/wide-counts33/input/dso/libstdlib_platform.so" >>"$D/SDK.sha256"
cp "$B/root-integrated02/RAW_READY" "$D/RAW.sha256"
sha256sum -c "$D/RAW.sha256" >"$D/RAW.before.verify"
bash "$R/input/scripts/stdlib/wave4_qpu_dyadic_gate.sh" "$R/gate" "$B/root-integrated02/results/madaros" >"$D/gate.stdout" 2>"$D/gate.stderr"
echo QPU_DYADIC_COMPONENT_QUALIFIED
