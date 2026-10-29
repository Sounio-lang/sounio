#!/bin/bash
set -euo pipefail
R=/scratch/sounio-stdlib-wave3-20261027/root-integrated04
S=$R/source;D=$R/preflight-v4
mkdir "$D"
P=/scratch/sounio-stdlib-inference-scale-20261024/stdlib-next-20261027;T=$P/tooling
export PATH="$P/tools:$T/root/usr/bin:/usr/bin:/bin" LD_LIBRARY_PATH="$T/root/usr/lib/x86_64-linux-gnu" MAGIC="$T/root/usr/lib/file/magic.mgc"
source "$R/tools-env-v2.sh"
ulimit -s 524288;ulimit -c 0
finish(){ r=$?;trap - EXIT;(cd "$S";sha256sum -c "$R/INITIAL_SOURCE.sha256") >"$D/SOURCE.final.verify" 2>&1||r=1;sha256sum -c "$D/INPUTS.sha256" >"$D/INPUTS.final.verify" 2>&1||r=1;echo "$r" >"$D/STATUS";exit "$r"; }
trap finish EXIT
cd "$S"
sha256sum -c "$R/INITIAL_SOURCE.sha256" >"$D/SOURCE.before.verify"
test "$(cat "$R/SOURCE_HEAD")" = ccc137cf642353502f39a1457b2e28cf5af147ea
test ! -e self-hosted/check/storage_types.sio
if grep -q 'TyStorage' self-hosted/check/types.sio >"$D/STORAGE_MUST_BE_ABSENT";then exit 1;fi
printf '%s  %s\n' e50736a7556ff271839b924e3336f92a5ad4a4f773b704437d25c531232be104 self-hosted/native/worker_proof.sio 673c2c0f364f7edb63f84b4f12aef3ed3607fbd82e370a4b8c1c3d9aa3355783 self-hosted/ir/loop_checkpoint_elision.sio 0e370fddc5c85147bc5e8215f4cc1e6b8ac4adecf8f172d3fc28db5a36ac1ac7 self-hosted/native/codegen_x86_linux.sio >"$D/QUALIFIED_COMPONENTS.sha256"
sha256sum -c "$D/QUALIFIED_COMPONENTS.sha256" >"$D/QUALIFIED_COMPONENTS.verify"
printf 'eaa61d90e8d2d1897f545636c06496f8b0791fbbf206dca5bd4fa4151926ac60  %s\n' "$R/capacity.patch" >"$D/CAPACITY.sha256"
sha256sum -c "$D/CAPACITY.sha256" >"$D/CAPACITY.verify"
git apply --check "$R/capacity.patch" >"$D/CAPACITY_APPLICABLE.verify" 2>&1
sha256sum "$R/capacity.patch" "$R/build-v3.sbatch" "$R/loader-v3.sbatch" "$0" "$R/source.tar" >"$D/INPUTS.sha256"
getconf GNU_LIBC_VERSION >"$D/LIBC_VERSION"
test "$(cat "$D/LIBC_VERSION")" = 'glibc 2.39'
find "$G" -type f -print0|sort -z|xargs -0 sha256sum >"$D/PRIVATE_SDK.sha256"
sha256sum "$P/tools/time" "$R/private-tools-v2/g++" "$R/tools-env-v2.sh" "$R/private-runtime/libgcc_s.so.1" "$R/private-runtime/libstdc++.so.6" >>"$D/PRIVATE_SDK.sha256"
"$CXX" --version >"$D/CXX_VERSION"
ldd "$G/usr/bin/x86_64-linux-gnu-g++-13" >"$D/CXX.ldd"
printf '#include <iostream>\nint main(){std::cout<<"PRIVATE_CPP23_READY\\n";}\n' >"$D/tool-smoke.cpp"
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror "$D/tool-smoke.cpp" -o "$D/tool-smoke"
"$D/tool-smoke" >"$D/tool-smoke.stdout"
grep -Fxq PRIVATE_CPP23_READY "$D/tool-smoke.stdout"
sha256sum -c "$D/PRIVATE_SDK.sha256" >"$D/PRIVATE_SDK.verify"
/scratch/sounio-stdlib-inference-scale-20261024/stdlib-next-20261027/tools/time -v bash "$R/tool-overlays/loop_checkpoint_elision_gate.sh" "$S" /scratch/sounio-stdlib-wave3-20261027/root-integrated02/results/madaros /scratch/sounio-stdlib-wave3-20261027/loop-regions-prototype03/source/bin/souc-lean-single-x86_64 "$D/loop-component" >"$D/loop-component.stdout" 2>"$D/loop-component.stderr"
sha256sum -c "$D/PRIVATE_SDK.sha256" >"$D/PRIVATE_SDK.final.verify"
echo FOCUSED_COMPONENT_AND_INPUTS_READY >"$R/PREFLIGHT_READY"
