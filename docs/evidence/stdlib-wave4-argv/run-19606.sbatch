#!/bin/bash
#SBATCH --job-name=stdlib-frozen-argv-material
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/argv-final42/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/argv-final42/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027;R=$B/services/argv-final42;S=$R/input;D=$R/results;U=$B/root-integrated02
export PATH=$B/services-tools-recovered/tools:/usr/bin:/bin
export SOUNIO_REPO_HARD_NO_RUST=1 SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1 SOUNIO_STDLIB_PATH="$S/stdlib" SOUC_BIN=/scratch/sounio-stdlib-wave3-20261027/services/argv-contract39/raw-wrapper
ulimit -s 524288;ulimit -c 0
finish(){ r=$?;trap - EXIT;for n in INPUTS COMPILER EXECUTABLES;do if [[ -s "$D/$n.sha256" ]];then sha256sum -c "$D/$n.sha256" >"$D/$n.final.verify" 2>&1||r=1;fi;done;echo "$r" >"$D/STATUS";exit "$r";};trap finish EXIT
cd "$S"
find "$S" -type f -print0|sort -z|xargs -0 sha256sum >"$D/INPUTS.sha256"
cp "$U/RAW_READY" "$D/COMPILER.sha256";sha256sum "$SOUC_BIN" >>"$D/COMPILER.sha256"
for t in argv_contract_gate.mjs test_stdlib_compile_fail.mjs test_stdlib_runtime_refusal.mjs test_stdlib_silent_refusal.mjs test_stdlib_e2e_diagnostics.mjs test_run_full_suite.mjs test_emit_stdlib_e2e_result.mjs;do node "scripts/stdlib/$t" >"$D/$t.stdout" 2>&1;echo "$t PASS";done
node - "$S" >"$D/frozen-inputs.verify" <<'JS'
const fs=require('fs'),path=require('path'),crypto=require('crypto'),root=process.argv[2];
const ledger=JSON.parse(fs.readFileSync(root+'/tests/fixtures/wave4_argv/FROZEN_PARENTS.json'));
const args=fs.readFileSync(root+'/tests/stdlib/hardware/test_pireus_material_engine_admission.sio.args','utf8').trimEnd().split('\n');
if(ledger.parents.length!==12||args.length!==12)throw Error('parent count');
for(const p of ledger.parents){if(args[p.index]!==p.argv_path)throw Error('order/path mismatch');const sha=crypto.createHash('sha256').update(fs.readFileSync(path.resolve(root,args[p.index]))).digest('hex');if(sha!==p.expected_sha256)throw Error('parent mismatch '+p.index);console.log(p.index+' '+sha+' '+args[p.index]);}
console.log('ALL_TWELVE_ORDERED_FROZEN_INPUTS_OK');
JS
for mode in 0 1;do
 export SOUNIO_LOOP_REGIONS="$mode"
 bash scripts/stdlib/run_stdlib_e2e.sh --filter tests/stdlib/hardware/test_pireus_material_engine_admission.sio --json-out "$D/material-$mode.json" --artifacts-dir "$R/material-$mode.capture" >"$D/material-$mode.stdout" 2>&1
 node -e 'const j=require(process.argv[1]);if(j.totals.total!==1||j.totals.pass!==1||j.totals.skip!==0)process.exit(1)' "$D/material-$mode.json"
done
timeout 1200 "$U/results/madaros" compile tests/stdlib/hardware/test_pireus_material_engine_admission.sio -o "$D/material.elf" >"$D/control.compile.stdout" 2>&1
chmod +x "$D/material.elf";sha256sum "$D/material.elf" >"$D/EXECUTABLES.sha256"
set +e
timeout 30 "$D/material.elf" >"$D/noargv.stdout" 2>&1
r=$?
set -e
[[ $r == 2 ]];echo "$r" >"$D/noargv.status"
mapfile -t current <"$R/control.original.args"
set +e
timeout 30 "$D/material.elf" "${current[@]}" >"$D/current-docs.stdout" 2>&1
r=$?
set -e
[[ $r == 1 ]];grep -Fq 'pireus material engine admission test failed:' "$D/current-docs.stdout";echo "$r" >"$D/current-docs.status"
echo FROZEN_MATERIAL_ORIGINAL_BOTH_MODES_AND_CURRENT_DOCS_NEGATIVE_OK
