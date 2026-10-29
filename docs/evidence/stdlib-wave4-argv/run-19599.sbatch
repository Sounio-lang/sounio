#!/bin/bash
#SBATCH --job-name=stdlib-literal-argv-contract
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/argv-contract39/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/argv-contract39/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027;R=$B/services/argv-contract39;S=$R/input;D=$R/results;U=$B/root-integrated02
export PATH=$B/services-tools-recovered/tools:/usr/bin:/bin
export SOUNIO_REPO_HARD_NO_RUST=1 SOUNIO_LOOP_REGIONS=1 SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
ulimit -s 524288;ulimit -c 0
finish(){ r=$?;trap - EXIT;sha256sum -c "$D/INPUTS.sha256" >"$D/INPUTS.final.verify" 2>&1||r=1;sha256sum -c "$D/COMPILER.sha256" >"$D/COMPILER.final.verify" 2>&1||r=1;echo "$r" >"$D/STATUS";exit "$r";};trap finish EXIT
find "$S" -type f -print0|sort -z|xargs -0 sha256sum >"$D/INPUTS.sha256"
cp "$U/RAW_READY" "$D/COMPILER.sha256";sha256sum -c "$D/COMPILER.sha256" >"$D/COMPILER.before.verify"
cd "$S"
bash -n scripts/stdlib/run_stdlib_e2e.sh
for t in argv_contract_gate.mjs test_stdlib_compile_fail.mjs test_stdlib_runtime_refusal.mjs test_stdlib_silent_refusal.mjs test_stdlib_e2e_diagnostics.mjs test_run_full_suite.mjs test_emit_stdlib_e2e_result.mjs;do
 node "scripts/stdlib/$t" >"$D/$t.stdout" 2>&1
 echo "$t PASS"
done
cat >"$R/raw-wrapper" <<'SH'
#!/bin/bash
set -euo pipefail
RAW=/scratch/sounio-stdlib-wave3-20261027/root-integrated02/results/madaros
if [[ "$1" == check ]];then shift;exec "$RAW" --check "$@";fi
exec "$RAW" "$@"
SH
chmod +x "$R/raw-wrapper";export SOUC_BIN="$R/raw-wrapper"
# A full original source archive is kept for both canonical-runner executions.
cp "$R/before/scripts/stdlib/run_stdlib_e2e.sh" "$R/old-runner"
cp "$R/before/scripts/stdlib/stdlib_e2e_diagnostics.mjs" "$R/old-diagnostics"
cp -a "$S/." "$R/before/"
cp "$R/old-runner" "$R/before/scripts/stdlib/run_stdlib_e2e.sh"
cp "$R/old-diagnostics" "$R/before/scripts/stdlib/stdlib_e2e_diagnostics.mjs"
testpath=tests/stdlib/hardware/test_pireus_material_engine_admission.sio
export SOUNIO_STDLIB_PATH="$R/before/stdlib"
set +e
bash "$R/before/scripts/stdlib/run_stdlib_e2e.sh" --filter "$testpath" --json-out "$D/material.before.json" --artifacts-dir "$R/material.before.capture" >"$D/material.before.stdout" 2>&1
rc=$?
set -e
[[ $rc == 1 ]]
node -e 'const j=require(process.argv[1]);if(j.totals.total!==1||j.totals.fail!==1||j.results[0].exit_code!==2)process.exit(1)' "$D/material.before.json"
export SOUNIO_STDLIB_PATH="$S/stdlib"
bash scripts/stdlib/run_stdlib_e2e.sh --filter "$testpath" --json-out "$D/material.after.json" --artifacts-dir "$R/material.after.capture" >"$D/material.after.stdout" 2>&1
node -e 'const j=require(process.argv[1]);if(j.totals.total!==1||j.totals.pass!==1||j.totals.skip!==0)process.exit(1)' "$D/material.after.json"
# Explicit no-argv control, same witness and compiler; no known-failure annotation changed.
timeout 1200 "$U/results/madaros" compile "$testpath" -o "$D/material.control.elf" >"$D/material.control.compile.stdout" 2>&1
chmod +x "$D/material.control.elf"
set +e
timeout 30 "$D/material.control.elf" >"$D/material.noargv.stdout" 2>&1
rc=$?
set -e
[[ $rc == 2 ]];echo "$rc" >"$D/material.noargv.status"
printf '%s\n' 'ARGV_CONTRACT_AND_REAL_MATERIAL_OK'
