#!/bin/bash
set -euo pipefail
R=/scratch/sounio-stdlib-wave3-20261027/root-integrated03
S=$R/source;D=$R/preflight
mkdir "$D"
export PATH=/usr/bin:/bin SOUNIO_LOOP_REGIONS=0 SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
ulimit -s 524288;ulimit -c 0
finish(){ r=$?;trap - EXIT;(cd "$S";sha256sum -c "$R/INITIAL_SOURCE.sha256") >"$D/SOURCE.final.verify" 2>&1||r=1;sha256sum -c "$D/TOOLS.sha256" >"$D/TOOLS.final.verify" 2>&1||r=1;echo "$r" >"$D/STATUS";exit "$r"; }
trap finish EXIT
cd "$S"
SEED=/scratch/sounio-stdlib-wave3-20261027/loop-regions-prototype03/source/bin/souc-lean-single-x86_64
RAW=/scratch/sounio-stdlib-wave3-20261027/root-integrated02/results/madaros
printf 'e6926020dd215096b24859cf81530f255985ccbe\n' > "$D/EXPECTED_HEAD"
cmp "$D/EXPECTED_HEAD" "$R/SOURCE_HEAD"
test ! -e self-hosted/check/storage_types.sio
! grep -q 'TyStorage' self-hosted/check/types.sio
! grep -q 'worker_lower_backend_opt_in' self-hosted/compiler/main.sio
sha256sum "$SEED" "$RAW" "$R/capacity.patch" "$R/build.sbatch" "$R/loader.sbatch" "$0" >"$D/TOOLS.sha256"
printf '9cfb6a970d73718b09091331dc514d015e3d4e6f402ee756cdc7f7b0cc55110a  %s\n8abe891a66c71bdbeb784a9200d1ea395d4d5a55f0036a0daa51ad3cf0614932  %s\ne07fd67b46f3abef9309cdb97dfd73b039d8c211ac9755d962c8a6242b90f2db  %s\n' "$SEED" "$RAW" "$S/self-hosted/native/worker_proof.sio" >"$D/EXPECTED_TOOLS.sha256"
sha256sum -c "$D/EXPECTED_TOOLS.sha256" >"$D/EXPECTED_TOOLS.verify"
sha256sum -c "$R/INITIAL_SOURCE.sha256" >"$D/SOURCE.before.verify"
# Compiler sources differ from raw02's original source only in the new proof.
grep '  self-hosted/' /scratch/sounio-stdlib-wave3-20261027/root-integrated02/INITIAL_SOURCE.sha256 | grep -v '  self-hosted/native/worker_proof.sio$' > "$D/UNCHANGED_COMPILER.sha256"
sha256sum -c "$D/UNCHANGED_COMPILER.sha256" > "$D/UNCHANGED_COMPILER.verify"
git apply --check "$R/capacity.patch" >"$D/CAPACITY_APPLICABLE.verify" 2>&1
bash tests/fixtures/wave4_worker_capability/components/gate.sh "$D/worker" "$SEED" "$RAW" >"$D/worker.stdout" 2>&1
grep -Fxq 'WORKER_REGION_ENVELOPES_OK zero_unique_dominant=1 frame_token_canonical=1 mutations_refused=18' "$D/worker/results/ir.native.stdout"
grep -Fxq WORKER_COMPONENTS_OK_PUBLIC_MAIN_CLOSED_BFS_AND_REAL_CALLBACKS_PENDING "$D/worker.stdout"
sha256sum -c "$D/TOOLS.sha256" >"$D/TOOLS.ready.verify"
printf 'ROOT_INTEGRATED03_PREFLIGHT_OK\n' >"$R/PREFLIGHT_READY"
