#!/bin/bash
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:20:00
#SBATCH --export=NIL
set -u
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
ulimit -s 524288
cd /scratch/sounio-stdlib-inference-scale-20261024/SIO167/source08
export MADAROS_RAW_BIN=/scratch/sounio-stdlib-inference-scale-20261024/compiler_effectbridge01/artifacts/madaros.elf
export SOUNIO_STDLIB_PATH=$PWD/stdlib
out=../phase08
sha256sum stdlib/stats/{noncentral_t,checked_power,power}.sio stdlib/special/checked_gamma_beta.sio tests/reference/noncentral_t*cpp "$MADAROS_RAW_BIN" > "$out/inputs.sha256"
run_case(){
 name=$1; source=$2
 SOUNIO_SOUC_ENGINE=lean_single bin/souc check "$source" > "$out/$name.seed.stdout" 2> "$out/$name.seed.stderr"
 if grep -Eq '^error:|error\[E' "$out/$name.seed.stdout" "$out/$name.seed.stderr";then return 1;fi
 bin/souc check "$source" > "$out/$name.check.stdout" 2> "$out/$name.check.stderr" || return $?
 bin/souc compile "$source" -o "$out/$name.elf" > "$out/$name.compile.stdout" 2> "$out/$name.compile.stderr" || return $?
 chmod +x "$out/$name.elf"
 sha256sum "$out/$name.elf" > "$out/$name.elf.sha256"
 /usr/bin/time -v timeout 240 "$out/$name.elf" > "$out/$name.native.stdout" 2> "$out/$name.native.stderr"
}
g++ -std=c++23 -O2 tests/reference/noncentral_t_reference.cpp -o "$out/nct_reference"
echo $? > "$out/nct.reference.compile.exit"
"$out/nct_reference" --generate > "$out/nct.generated.sio"
run_case nct "$out/nct.generated.sio";echo $? > "$out/nct.run.exit"
"$out/nct_reference" "$out/nct.native.stdout" > "$out/nct.verify.stdout" 2> "$out/nct.verify.stderr";echo $? > "$out/nct.verify.exit"
g++ -std=c++23 -O2 tests/reference/noncentral_t_power_reference.cpp -o "$out/power_reference"
echo $? > "$out/power.reference.compile.exit"
"$out/power_reference" --generate > "$out/power.generated.sio"
run_case power "$out/power.generated.sio";echo $? > "$out/power.run.exit"
"$out/power_reference" "$out/power.native.stdout" > "$out/power.verify.stdout" 2> "$out/power.verify.stderr";echo $? > "$out/power.verify.exit"
"$out/power_reference" --generate-legacy > "$out/legacy.generated.sio"
run_case legacy_after "$out/legacy.generated.sio";echo $? > "$out/legacy.after.run.exit"
"$out/power_reference" --legacy "$out/legacy_after.native.stdout" > "$out/legacy.after.verify.stdout" 2> "$out/legacy.after.verify.stderr";echo $? > "$out/legacy.after.verify.exit"
cd ../source05
export SOUNIO_STDLIB_PATH=$PWD/stdlib
run_case legacy_before "$out/legacy.generated.sio";echo $? > "$out/legacy.before.run.exit"
"$out/power_reference" --legacy "$out/legacy_before.native.stdout" > "$out/legacy.before.verify.stdout" 2> "$out/legacy.before.verify.stderr";echo $? > "$out/legacy.before.verify.exit"
