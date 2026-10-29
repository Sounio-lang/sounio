#!/bin/bash
set -euo pipefail
R=/scratch/sounio-stdlib-wave3-20261027/root-integrated03
S=$R/source;D=$R/results
test -s "$R/PREFLIGHT_READY"
mkdir "$D"
P=/scratch/sounio-stdlib-inference-scale-20261024/stdlib-next-20261027
T=$P/tooling
export PATH="$P/tools:$T/root/usr/bin:/usr/bin:/bin" LD_LIBRARY_PATH="$T/root/usr/lib/x86_64-linux-gnu" MAGIC="$T/root/usr/lib/file/magic.mgc"
ulimit -s 524288;ulimit -c 0
finish(){ r=$?;trap - EXIT;echo "$r" >"$D/BUILD_STATUS";date -u >"$D/ENDED_UTC";exit "$r"; }
trap finish EXIT
cd "$S"
sha256sum -c "$R/INITIAL_SOURCE.sha256" >"$D/PREFLIGHT_SOURCE.verify"
printf 'eaa61d90e8d2d1897f545636c06496f8b0791fbbf206dca5bd4fa4151926ac60  %s\n' "$R/capacity.patch" >"$D/CAPACITY.sha256"
sha256sum -c "$D/CAPACITY.sha256" >"$D/CAPACITY.verify"
git apply --check "$R/capacity.patch" >"$D/CAPACITY_APPLICABLE.verify" 2>&1
git apply "$R/capacity.patch"
cp self-hosted/compiler/main.sio "$D/main.before.sio"
test "$(grep -cFx 'use parser::ast::*' self-hosted/compiler/main.sio)" == 1
test "$(grep -cFx 'fn main() with IO, Mut, Panic, Div, Alloc {' self-hosted/compiler/main.sio)" == 1
sed -i '/^use parser::ast::\*$/a\use native::worker_proof::{worker_lower_backend_opt_in}' self-hosted/compiler/main.sio
sed -i '/^fn main() with IO, Mut, Panic, Div, Alloc {$/a\    worker_lower_backend_opt_in()' self-hosted/compiler/main.sio
diff -u "$D/main.before.sio" self-hosted/compiler/main.sio >"$D/WORKER_OPT_IN.patch" || [[ $? == 1 ]]
test "$(grep -cFx '    worker_lower_backend_opt_in()' self-hosted/compiler/main.sio)" == 1
mkdir -p bin
cp /scratch/sounio-stdlib-wave3-20261027/loop-regions-prototype03/source/bin/souc-lean-single-x86_64 bin/
printf '9cfb6a970d73718b09091331dc514d015e3d4e6f402ee756cdc7f7b0cc55110a  bin/souc-lean-single-x86_64\n' >"$D/SEED.sha256"
sha256sum -c "$D/SEED.sha256" >"$D/SEED.before.verify"
find self-hosted stdlib scripts tests runtime -type f -print0|sort -z|xargs -0 sha256sum >"$D/SOURCE.sha256"
sha256sum "$D/WORKER_OPT_IN.patch" >>"$D/CAPACITY.sha256"
export SOUNIO_STDLIB_PATH="$S/stdlib" SOUNIO_BUILD_LOCK="$R/build.lock" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1 SOUNIO_LOOP_REGIONS=0
/usr/bin/time -v env SOUC_BIN="$S/bin/souc-lean-single-x86_64" bash scripts/ci/build_modular_madaros.sh "$D/madaros" >"$D/build.stdout" 2>"$D/build.stderr"
if grep -E '^error:|^error\[|Error:|field initializer mismatch|would have dropped|cannot compile an assignment' "$D/build.stdout" "$D/build.stderr" >"$D/EXPLICIT_ERRORS";then exit 1;fi
test -s "$D/madaros"
test "$(od -An -tx1 -N4 "$D/madaros"|tr -d ' \n')" == 7f454c46
chmod +x "$D/madaros"
sha256sum "$D/madaros" >"$D/COMPILER.sha256"
sha256sum -c "$D/SOURCE.sha256" >"$D/SOURCE.final.verify"
sha256sum -c "$D/SEED.sha256" >"$D/SEED.final.verify"
sha256sum -c "$D/CAPACITY.sha256" >"$D/CAPACITY.final.verify"
sha256sum -c "$D/COMPILER.sha256" >"$D/COMPILER.final.verify"
cp "$D/COMPILER.sha256" "$R/RAW_READY"
echo RAW_BUILT_INTEGRATED_RUNTIME_AND_CORPUS_PENDING
