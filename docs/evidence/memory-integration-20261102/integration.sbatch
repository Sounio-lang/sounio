#!/bin/bash
#SBATCH -J sounio-memory-integrate
#SBATCH -p cpu-ops
#SBATCH -w cpuops-t560-proxmox
#SBATCH -n1
#SBATCH -c4
#SBATCH --mem=12G
#SBATCH -t 00:18:00
#SBATCH -o /scratch/sounio-memory-integration-20261102/integration-%j.log
set -euo pipefail
ulimit -s 1048576
cd /scratch/sounio-memory-integration-20261102/source
run=/scratch/sounio-memory-integration-20261102/run-$SLURM_JOB_ID
mkdir -p "$run"
tc=/scratch/amrex6090-sm75-86-build-20261101/deps/toolchain
export PATH="$tc/usr/bin:$PATH"
export COMPILER_PATH="$tc/usr/libexec/gcc/x86_64-linux-gnu/13:$tc/usr/bin"
export LIBRARY_PATH="$tc/usr/lib/x86_64-linux-gnu:$tc/usr/lib/gcc/x86_64-linux-gnu/13"
export LD_LIBRARY_PATH="$tc/usr/lib/x86_64-linux-gnu"
export CPLUS_INCLUDE_PATH="$tc/usr/include/c++/13:$tc/usr/include/x86_64-linux-gnu/c++/13:$tc/usr/lib/gcc/x86_64-linux-gnu/13/include:$tc/usr/include:$tc/usr/include/x86_64-linux-gnu"
export SOUNIO_BUILD_LOCK="$run/build.lock"
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
unset SOUC_BIN SOUNIO_SOUC_BIN MADAROS_BIN SOUNIO_MADAROS_BIN || true
sha256sum self-hosted/compiler/main.sio self-hosted/native/avx512.sio self-hosted/native/codegen_x86_linux.sio tests/native/bytewidth_active_backend.sio
if ! timeout --kill-after=10s 750s bash scripts/ci/build_modular_madaros.sh "$run/madaros" > "$run/build.log" 2>&1;then tail -25 "$run/build.log";exit 2;fi
sha256sum "$run/madaros"
export SOUNIO_MADAROS_BIN="$run/madaros"
for src in self-hosted/native/avx512.sio tests/native/bytewidth_active_backend.sio;do
 timeout --kill-after=5s 60s bin/souc check "$src" > "$run/$(basename "$src").check.log" 2>&1
done
timeout --kill-after=5s 90s bin/souc compile tests/native/bytewidth_active_backend.sio -o "$run/bytewidth.elf" > "$run/bytewidth-compile.log" 2>&1
chmod +x "$run/bytewidth.elf"
timeout 30 "$run/bytewidth.elf" > "$run/bytewidth.log"
"$tc/usr/bin/g++" -std=c++23 -O2 -Wall -Wextra -Werror docs/evidence/bytewidth-active-20261102/oracle.cpp -o "$run/byte-oracle"
"$run/byte-oracle" "$run/bytewidth.log"
timeout --kill-after=5s 90s bin/souc compile docs/evidence/bytewidth-active-20261102/word-mutant.sio -o "$run/word-mutant.elf" > "$run/word-mutant-compile.log" 2>&1
chmod +x "$run/word-mutant.elf"
set +e
timeout 30 "$run/word-mutant.elf" > "$run/word-mutant.log"
rc=$?
"$run/byte-oracle" "$run/word-mutant.log"
orc=$?
set -e
test "$rc" -eq 2 && test "$orc" -eq 1
echo ACTIVE_SOURCE_BYTEWIDTH_AND_NEGATIVE_PASS
# Direct emitter unit uses the pinned bootstrap, separately from default Madaros.
seed=/scratch/sounio-memory-contract-20261102/bin/souc-lean-single-x86_64
sha256sum "$seed"
cat self-hosted/native/avx512.sio docs/evidence/memory-contract-20261102/probe-main.sio > "$run/emitter.sio"
timeout 90 "$seed" "$run/emitter.sio" "$run/emitter.elf" > "$run/emitter-build.log" 2>&1
chmod +x "$run/emitter.elf"
timeout 40 "$run/emitter.elf" > "$run/emitter.log"
tail -3 "$run/emitter.log"
"$tc/usr/bin/g++" -std=c++23 -O2 -Wall -Wextra -Werror docs/evidence/memory-contract-20261102/check.cpp -o "$run/encoding-check"
"$run/encoding-check" docs/evidence/memory-contract-20261102/oracle.txt "$run/emitter.log" native
printf 'CASE malformed\n' > "$run/malformed.log"
set +e
"$run/encoding-check" docs/evidence/memory-contract-20261102/oracle.txt "$run/malformed.log" bytes
rc=$?
set -e
test "$rc" -eq 6
echo MALFORMED_CASE_REJECTED
sed '/^    \/\/ RSP\/R12 select/,+3d' self-hosted/native/avx512.sio > "$run/sib-mutant.sio"
cat docs/evidence/memory-contract-20261102/probe-main.sio >> "$run/sib-mutant.sio"
timeout 90 "$seed" "$run/sib-mutant.sio" "$run/sib-mutant.elf" > "$run/sib-mutant-build.log" 2>&1
chmod +x "$run/sib-mutant.elf"
set +e
timeout 40 "$run/sib-mutant.elf" > "$run/sib-mutant.log"
rc=$?
set -e
test "$rc" -eq 1
grep -F 'FAIL: T73 disp32 SIB full bytes' "$run/sib-mutant.log"
echo INTEGRATION_VALIDATION_PASS
