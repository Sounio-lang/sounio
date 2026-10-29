#!/bin/bash
#SBATCH --job-name=pireus-five-exact-elf-inputs
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:10:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/pireus-staging43/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/pireus-staging43/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027;R=$B/services/pireus-staging43;D=$R/results;BASE=$B/whole-integrated01
export PATH=$B/services-tools-recovered/tools:/usr/bin:/bin
export LD_LIBRARY_PATH="$BASE/results/native" SOUNIO_LOOP_REGIONS=1
ulimit -s 524288;ulimit -c 0
finish(){ r=$?;trap - EXIT;for n in INPUTS TOOLS ELFS WITNESSES;do if [[ -s "$D/$n.sha256" ]];then sha256sum -c "$D/$n.sha256" >"$D/$n.final.verify" 2>&1||r=1;fi;done;echo "$r" >"$D/STATUS";exit "$r";};trap finish EXIT
find "$R/complete" "$R/incomplete" -type f -print0|sort -z|xargs -0 sha256sum >"$D/INPUTS.sha256"
sha256sum "$BASE/results/native/libstdlib_platform.so" "$B/root-integrated01/results/madaros" "$B/services-tools-recovered/tools/node" /usr/bin/time >"$D/TOOLS.sha256"
node - "$R" >"$D/PARENTS.verify" <<'JS'
const fs=require('fs'),crypto=require('crypto'),path=require('path'),r=process.argv[2],j=JSON.parse(fs.readFileSync(r+'/PARENTS.json'));if(j.parents.length!==22)throw Error('empty/incomplete metadata selection');for(const p of j.parents){const f=path.join(r,'complete',p.path);if(crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex')!==p.expected_sha256)throw Error('parent mismatch '+p.path);console.log(p.module+' '+p.field+' '+p.expected_sha256+' '+p.path);}if(fs.existsSync(r+'/incomplete/docs')||fs.existsSync(r+'/incomplete/tools'))throw Error('control unexpectedly has docs/tools');
JS
for n in multiprobe_block_certification operator_autogenesis operator_discovery_engine operator_novelty_frontier operator_seed_kernel;do
 old="$BASE/results/retained/tests/stdlib/hardware/test_pireus_$n.sio.elf"
 sha256sum -c "$old.sha256" >>"$D/ORIGINAL_ELFS.verify"
 cmp "$old" "$R/elf/$n.elf"
 sha256sum "$old" "$R/elf/$n.elf" >>"$D/ELFS.sha256"
 src=tests/stdlib/hardware/test_pireus_$n.sio
 cmp "$BASE/source/$src" "$R/complete/$src";cmp "$BASE/source/$src" "$R/incomplete/$src"
 sha256sum "$BASE/source/$src" "$R/complete/$src" "$R/incomplete/$src" >>"$D/WITNESSES.sha256"
done
run_case(){
 local n="$1" seconds="$2" expected="$3" rc=0
 cd "$R/incomplete"
 /usr/bin/time -v -o "$D/$n.before.resource" timeout "$seconds" "$R/elf/$n.elf" >"$D/$n.before.stdout" 2>&1||rc=$?
 echo "$rc" >"$D/$n.before.status";[[ $rc == "$expected" ]]||return 1
 cd "$R/complete";rc=0
 /usr/bin/time -v -o "$D/$n.after.resource" timeout "$seconds" "$R/elf/$n.elf" >"$D/$n.after.stdout" 2>&1||rc=$?
 echo "$rc" >"$D/$n.after.status"
 [[ $rc == 0 ]]
}
run_case multiprobe_block_certification 1200 3 & p1=$!
run_case operator_autogenesis 30 1 & p2=$!
run_case operator_discovery_engine 3600 1 & p3=$!
run_case operator_novelty_frontier 3600 1 & p4=$!
run_case operator_seed_kernel 30 1 & p5=$!
rc=0
for pid in "$p1" "$p2" "$p3" "$p4" "$p5";do wait "$pid"||rc=1;done
printf '%s\n' 'Same retained compiler11332 ELFs from19392; source/witness/expected hashes unchanged; only runtime cwd input closure changes. No18timeout reruns, no new compiler.'
exit "$rc"
