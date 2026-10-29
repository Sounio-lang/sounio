#!/bin/bash
set -euo pipefail
R=/scratch/sounio-stdlib-wave3-20261027/root-integrated02
S=$R/source;D=$R/preflight
mkdir "$D"
export PATH=/usr/bin:/bin SOUNIO_LOOP_REGIONS=0
ulimit -s 524288;ulimit -c 0
finish(){ r=$?;trap - EXIT;sha256sum -c "$R/INITIAL_SOURCE.sha256" >"$D/SOURCE.final.verify" 2>&1||r=1;sha256sum -c "$D/TOOLS.sha256" >"$D/TOOLS.final.verify" 2>&1||r=1;echo "$r" >"$D/STATUS";exit "$r"; }
trap finish EXIT
cd "$S"
SEED=/scratch/sounio-stdlib-wave3-20261027/loop-regions-prototype03/source/bin/souc-lean-single-x86_64
RAW=/scratch/sounio-stdlib-wave3-20261027/root-integrated01/results/madaros
sha256sum "$SEED" "$RAW" "$R/capacity.patch" "$0" >"$D/TOOLS.sha256"
sha256sum -c "$R/INITIAL_SOURCE.sha256" >"$D/SOURCE.before.verify"
printf '9cfb6a970d73718b09091331dc514d015e3d4e6f402ee756cdc7f7b0cc55110a  %s\n' "$SEED" >"$D/EXPECTED_SEED.sha256"
sha256sum -c "$D/EXPECTED_SEED.sha256" >"$D/EXPECTED_SEED.verify"
git apply --check "$R/capacity.patch" >"$D/CAPACITY_APPLICABLE.verify" 2>&1
bash tests/fixtures/wave4_worker_capability/components/gate.sh "$D/worker" "$SEED" "$RAW" >"$D/worker.stdout" 2>&1
bash tests/fixtures/wave4_borrow_capacity/gate_checker_component.sh "$SEED" "$D/checker" >"$D/checker.stdout" 2>&1
while IFS=$'\t' read -r name before after;do
 [[ -n "$name" ]]
 rc=0
 timeout 120 "$D/checker/checker.elf" "tests/fixtures/wave4_constant_scope/$name.sio" >"$D/$name.stdout" 2>&1||rc=$?
 [[ "$rc" == "$after" ]]
 grep -Fxq "ACTUAL_CHECKER_VERDICT $after" "$D/$name.stdout"
 if [[ "$after" == 1 ]];then grep -Fq 'error[E056]' "$D/$name.stdout";fi
 printf '%s\t%s\n' "$name" "$rc" >>"$D/CONSTANTS.tsv"
done <tests/fixtures/wave4_constant_scope/CASES.tsv
bash tests/fixtures/wave4_logical_ownership/gate_existing.sh "$D/checker/checker.elf" /scratch/sounio-stdlib-wave3-20261027/services/logical-ownership03/focused/before "$D/logical" >"$D/logical.stdout" 2>&1
export SOUNIO_STDLIB_PATH="$S/self-hosted"
timeout 600 "$SEED" tests/fixtures/wave4_alias_canonicalization/plain_driver.sio.in "$D/alias.elf" >"$D/alias.compile.stdout" 2>&1
! grep -Eq '^error:|^error\[|would have dropped|cannot compile an assignment' "$D/alias.compile.stdout"
test -s "$D/alias.elf";chmod +x "$D/alias.elf"
sha256sum "$D/alias.elf" >"$D/ALIAS_ELF.sha256"
bash scripts/compiler/alias_canonical_component_gate.sh "$D/alias.elf" tests/fixtures/wave4_alias_canonicalization "$D/alias-matrix" >"$D/alias.stdout" 2>&1
sha256sum -c "$R/INITIAL_SOURCE.sha256" >"$D/SOURCE.ready.verify"
sha256sum -c "$D/TOOLS.sha256" >"$D/TOOLS.ready.verify"
printf 'ROOT_INTEGRATED02_PREFLIGHT_OK\n' >"$R/PREFLIGHT_READY"
